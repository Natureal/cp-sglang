from collections import defaultdict
import random
import logging
import time
import heapq
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import torch

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.predictor.lrb import LRBReuseDistancePredictor
from sglang.srt.predictor.popu import POPUPredictor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_ts = 0
        self.pred = 0
        self.pred_valid = 0

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        # GUARD algorithm specific attributes
        self.guarded = False           # whether the node is guarded in current phase

        # predictor
        self.freq = 1

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_ts < other.last_access_ts

def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i

class GuardRadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        waiting_queue_cache: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable

        self.waiting_queue_cache = waiting_queue_cache

        self.algo_type = "lru"
        self.degrade_to_lru = True

        #self.predictor = POPUPredictor()
        #self.predictor = LRUPredictor()
        #self.predictor = PLECOPredictor()
        self.predictor = LRBReuseDistancePredictor()

        self.current_ts = 0

        self.evicted_in_phase = set()
        self.rand_evict_budget = 0

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        
        # GUARD algorithm specific state
        self.U: Set[TreeNode] = set()  # unrequested old pages in current phase
        self.current_phase = 0
        self.current_request_key: Optional[List[int]] = None  # track current request
        
        self.reset()

        self.timestamp = 1

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        
        # Reset GUARD state
        self.U.clear()
        self.current_phase = 0
        self.current_request_key = None
        self.current_phase_err_num = 0
        self.phase_relax_times = 10
    
    def _predict(self, nodes: List[TreeNode]):
        node_to_pred = []
        addresses = []
        for node in nodes:
            if node.pred_valid == 0:
                node_to_pred.append(node)
                addresses.append(hash(tuple(node.key)))

        if len(node_to_pred) > 0:
            preds = self.predictor.predict(addresses)
            for i in range(len(node_to_pred)):
                node = node_to_pred[i]
                if preds[i] == 2**62:
                    node.pred = -node.last_access_ts + preds[i]
                else:
                    node.pred = node.last_access_ts + preds[i]
                node.pred_valid = 1

    def match_prefix(self, key: List[int], **kwargs) -> Tuple[torch.Tensor, TreeNode]:
        """Find the matching prefix from the radix tree with GUARD tracking."""
        if self.disable or len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        # Update current request for GUARD algorithm
        self.current_request_key = key.copy()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        
        self.timestamp += 1
        return value, last_node

    def _match_prefix_helper(self, node: TreeNode, key: List):
        """Match prefix helper with GUARD access tracking."""
        #self._predictor_access(node)
        node.freq += 1
        
        if self.degrade_to_lru == False:
            # GUARD: If node is in U (unrequested), remove it and mark as guarded
            if node in self.U:
                self.U.remove(node)

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            #self._predictor_access(child)
            child.freq += 1
            
            if self.degrade_to_lru == False:
                # GUARD: Track access to child node
                if child in self.U:
                    self.U.remove(child)
            
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                original_key = child.key
                new_node = self._split_node(child.key, child, prefix_len)
                self._predictor_split(original_key, node, new_node)
                # copy ts from node when splitting node
                new_node.last_access_ts = node.last_access_ts

                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def insert(self, key: List, value=None, finished_req = False):
        if self.disable:
            return 0
        
        if finished_req == True:
            self.current_ts += 1

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, finished_req)
    
    def _evict_by_lru(self, num_tokens: int):
        if self.disable:
            return
        
        self.token_to_kv_pool_allocator.record_eviction(num_tokens)
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def _start_new_phase(self):
        """Start a new phase: unguard all cached pages and reset phase-specific flags."""
        self.current_phase += 1
        
        # Collect all cached nodes (excluding root)
        all_nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node != self.root_node and not node.evicted:
                all_nodes.append(node)
                # Reset phase-specific flags
                node.guarded = False
            
            for child in node.children.values():
                if not child.evicted:
                    stack.append(child)
        
        # All cached pages become unrequested old pages
        self.U = set(all_nodes)
        self.evicted_in_phase = set()
        self.rand_evict_budget = 0
        self.current_phase_err_num = 0
        logger.info(f"new phase starts, num of all_nodes: {len(self.U)}")

    def evict(self, num_tokens: int):
        """GUARD eviction algorithm implementation."""
        if self.disable:
            return
        
        if self.degrade_to_lru == True:
            self._evict_by_lru(num_tokens)
            return
        
        self.token_to_kv_pool_allocator.record_eviction(num_tokens)

        num_evicted = 0

        evictable_leaves = [node for node in self._collect_leaves() if node != self.root_node and node.lock_ref == 0]
        while num_evicted < num_tokens and len(evictable_leaves):
            # Step 1: If U is empty, start new phase
            if len(self.U) == 0:
                self._start_new_phase()

            # Step 3: Choose eviction strategy based on GUARD algorithm
            victim = None

            if self.rand_evict_budget > 0:
                u_candidates = [node for node in evictable_leaves if node in self.U]
                if u_candidates:
                    victim = random.choice(u_candidates)
                    self.rand_evict_budget -= 1

            if victim is None:
                # Strategy 2: Belady algorithm on unguarded nodes
                unguarded_candidates = [node for node in evictable_leaves if not node.guarded]
                if len(unguarded_candidates) == 0:
                    self._start_new_phase()
                    unguarded_candidates = evictable_leaves

                self._predict(unguarded_candidates)
                # Choose node with maximum reuse distance (farthest reuse)
                victim = max(unguarded_candidates, key=lambda node: node.pred)
                logger.info(f"victim pred = {victim.pred}, num of unguarded nodes: {len(unguarded_candidates)}")

            # Step 4: Perform eviction
            evictable_leaves.remove(victim)
            self.token_to_kv_pool_allocator.free(victim.value)
            num_evicted += len(victim.value)
            
            # Mark as evicted in current phase
            address = hash(tuple(victim.key))
            self.evicted_in_phase.add(address)
            
            # Remove from U if present
            self.U.discard(victim)

            logger.info(f"victim parent: {str(victim.parent)}, root node: {str(self.root_node)}")
            # Delete the leaf node
            self._delete_leaf(victim)

            if len(victim.parent.children) == 0 and victim.parent != self.root_node and victim.parent.lock_ref == 0:
                evictable_leaves.append(victim.parent)

    def cache_finished_req(self, req):
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            token_ids[:page_aligned_len], page_aligned_kv_indices, True
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def set_algo_type(self, algo_type):
        if self.algo_type != algo_type:
            self.algo_type = algo_type
            if algo_type == "lru":
                self.degrade_to_lru = True
                logger.info(f"Caching algorithm switches from guard to lru")
            elif algo_type == "guard":
                self.degrade_to_lru = False
                logger.info(f"Caching algorithm switches from lru to guard")
        else:
            logger.info(f"Caching algorithm is already {self.algo_type}")

    def set_online_training(self, obj):
        if self.degrade_to_lru == True or self.predictor.enable_online_training is None:
            logger.info(f"No training needed")
            return

        logger.info(f"Configure online training as {obj}")
        self.predictor.enable_online_training = obj.enable_online_training
        if self.predictor.enable_online_training == "on":
            if obj.training_interval is not None:
                self.predictor.training_interval = obj.training_interval
            if obj.training_window is not None:
                self.predictor.training_window = obj.training_window
            logger.info(f"Current online training config: enable({self.predictor.enable_online_training}), interval({self.predictor.training_interval}), window({self.predictor.training_window})")

    def cache_unfinished_req(self, req):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(page_aligned_token_ids)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    # Inherit other methods from RadixCache
    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node
    
    def _predictor_access(self, node: TreeNode, current_ts):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return
        #logger.info(f"pred access key = {hash(tuple(node.key))}")
        self.predictor.access(hash(tuple(node.key)), current_ts)
        #if node.pred_valid == 1:
         #   logger.info(f"node pred = {node.pred}, truth = {self.current_ts}, interval = {self.current_ts - node.last_access_ts}, node key = {hash(tuple(node.key))}")
        node.pred_valid = 0

        #logger.info(f"current ts: {self.current_ts}")
        #if self.current_ts % 100 == 0:
        #    captured = self._capture_print()
        #    logger.info(f"---------------------------------------------------- tree structure: {captured}")

    def _predictor_split(self, original_key, node: TreeNode, new_node: TreeNode):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return
        self._predictor_feature_copy(original_key, node.key)
        self._predictor_feature_copy(original_key, new_node.key)
        # copy pred from original node
        new_node.pred_valid = node.pred_valid
        new_node.pred = node.pred

    def _predictor_feature_copy(self, key, new_key):
        self.predictor.feature_copy(hash(tuple(key)), hash(tuple(new_key)))

    def _predictor_spawn(self, node: TreeNode, new_node: TreeNode):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return
        self._predictor_feature_copy(node.key, new_node.key)
        # copy pred from parent node
        new_node.pred_valid = node.pred_valid
        new_node.pred = node.pred
       # pass

    def _judge_evicted_in_phase(self, node: TreeNode):
        address = hash(tuple(node.key))
        if address in self.evicted_in_phase:
            return True
        return False

    def _insert_helper(self, node: TreeNode, key: List, value, finished_req):
        if len(key) == 0:
            return 0
        # update ts and features only when the request is finished
        if finished_req == True:
            self._predictor_access(node, self.current_ts)
            node.last_access_ts = self.current_ts

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            if finished_req == True:
                self._predictor_access(node, self.current_ts)
                node.last_access_ts = self.current_ts

            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                original_key = node.key
                new_node = self._split_node(node.key, node, prefix_len)
                self._predictor_split(original_key, node, new_node)
                # copy ts from node when splitting node
                new_node.last_access_ts = node.last_access_ts

                if self._judge_evicted_in_phase(new_node):
                    new_node.guarded = True
                    self.rand_evict_budget += 1
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            #self._predictor_spawn(node, new_node)
            # copy ts from parent node when spawning node
            new_node.last_access_ts = node.last_access_ts

            if self._judge_evicted_in_phase(new_node):
                new_node.guarded = True
                self.rand_evict_budget += 1

            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

        self.token_to_kv_pool_allocator.evictable_size = self.evictable_size_
        return total_prefix_length

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")
        print(f"Current phase: {self.current_phase}")
        print(f"Unrequested pages in U: {len(self.U)}")

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            guard_info = f"G={current_node.guarded}, E={current_node.evicted_in_phase}"
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
                guard_info,
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

    def total_size(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                if not child.evicted:
                    values.append(child.value)
                    _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if values else torch.empty((0,), dtype=torch.int64, device=self.device)
    
if __name__ == "__main__":
    tree = GuardRadixCache(None, None, page_size=1, disable=False)

    tree.insert([1,2,3,4,5])
    tree.insert([1,2,3,4,5])
    tree.insert([1,2,3,4,5,6,7,8,9,10])
    tree.insert([1,2,3,4,5,11,12,13,14,15])
    tree.insert([4,5,6])
    tree.pretty_print()

    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(5)
    tree.evict(10)
    tree.pretty_print()

from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import numpy as np
import argparse
import json
from types import SimpleNamespace
import torch
import heapq
import random
import pickle
import time
import sys
import io
import os
import logging
import math
import json
from collections import defaultdict, deque
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple, Set

from sglang.bench_serving import get_tokenizer

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.guard_radix_cache import GuardRadixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.predictor.pleco import PLECOPredictor
from sglang.srt.predictor.popu import POPUPredictor
from sglang.srt.predictor.lrb import LRBReuseDistancePredictor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)
NRT = {}
NRT_truth = {}

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
        self.access_times = 0
        self.match_tag = 0
        self.key_path_len = 0

        self.prefix_key = None
        self.hash_value = None

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

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
    
    #def __lt__(self, other: "TreeNode"):
     #   return self.pred > other.pred

def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

def _key_match_page_size1_fast(key0: List, key1: List) -> int:
    a = np.asarray(key0)
    b = np.asarray(key1)
    n = min(len(a), len(b))
    diff = a[:n] != b[:n]
    return np.argmax(diff) if diff.any() else n

def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i

class PhaseLRURadixCache(BasePrefixCache):
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
        self.cache_size_k = 0
        self.page_size = page_size
        self.disable = disable
        self.evictable_size_ = 0

        # phase-level
        self.distinct_hash = set()
        self.distinct_item_count = 0
        self.phase_err_param = 1
        self.pred_evicted = set()
        self.lru_evicted = set()
        self.inv_count = 0
        self.U: Set[TreeNode] = set()
        self.lru_budget = 0
        self.lru_evict_count = 0
        self.pred_evict_count = 0
        self.dump_file_count = 0
        self.dump_req_keys = []
        self.rand = random.randint(0, 10000)

        # global-level
        self.deleted_node_count = 0
        self.current_ts = 0
        self.rank_sum = 0
        self.pred_rank_sum = 0
        self.lru_rank_sum = 0

        self.waiting_queue_cache = waiting_queue_cache

        self.algo_type = "lru"
        self.degrade_to_lru = True

        #self.predictor = POPUPredictor()
        #self.predictor = PLECOPredictor()
        self.predictor = LRBReuseDistancePredictor()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
            self.cache_size_k = self.token_to_kv_pool_allocator.size
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.key_path_len = 0
        self.root_node.prefix_key = None
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.U.clear()

        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return
        self._start_new_phase()

    def match_prefix(self, key: List[int], init_req=False, **kwargs) -> Tuple[torch.Tensor, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
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

        value, last_node = self._match_prefix_helper(self.root_node, key, init_req)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return value, last_node
    
    def _match_prefix_helper(self, node: TreeNode, key: List, init_req):
        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]

            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                # originial_key is splitted into node.key and new_node.key
                new_node = self._split_node(node.hash_value, node, prefix_len)
                if init_req == True:
                    new_node.match_tag += 1
                    #print(f"increase tag 2: {new_node.hash_value}, value: {new_node.match_tag}")

                #self._record_access(new_node, node.last_access_ts)
                #self._judge_evicted_in_phase(node)
                #self._judge_evicted_in_phase(new_node)

                value.append(new_node.value)
                node = new_node
                break
            else:
                if init_req == True:
                    node.match_tag += 1
                    #print(f"increase tag 1: {node.hash_value}, value: {node.match_tag}")

                value.append(node.value)
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
    
    def _insert_helper(self, node: TreeNode, key: List, value, finished_req):
        if len(key) == 0:
            return 0
        
        #if finished_req == True:
        #    self._predictor_access(node, self.current_ts)
        #    self._record_access(node, self.current_ts)

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]

            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                # originial_key is splitted into node.key and new_node.key
                new_node = self._split_node(node.hash_value, node, prefix_len)
                if finished_req == True:
                    self._predictor_access(new_node, self.current_ts)
                    self._record_access(new_node, self.current_ts)
                    new_node.match_tag -= 1
                    #print(f"decrease tag 1: {new_node.hash_value}, value: {new_node.match_tag}")
                # update ts for new_node
                #self._judge_evicted_in_phase(node)
                #self._judge_evicted_in_phase(new_node)

                node = new_node
            else:
                if finished_req == True:
                    self._predictor_access(node, self.current_ts)
                    self._record_access(node, self.current_ts)
                    node.match_tag -= 1
                    #print(f"decrease tag 2: {node.hash_value}, value: {node.match_tag}")

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            if node.prefix_key:
                new_node.prefix_key = node.prefix_key + node.key
            else:
                new_node.prefix_key = node.key
            new_node.key_path_len = len(new_node.prefix_key) + len(new_node.key)
            self._generate_node_hash_value(new_node)
            print(f"insert new_node hash v = {new_node.hash_value}")
            new_node.value = value

            if node != self.root_node:
                self._predictor_spawn(node, new_node)
                self._record_access(new_node, self.current_ts)
            elif finished_req == True:
                self._predictor_access(new_node, self.current_ts)
                self._record_access(new_node, self.current_ts)

            self._judge_evicted_in_phase(new_node)

            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

        if self.token_to_kv_pool_allocator:
            self.token_to_kv_pool_allocator.evictable_size = self.evictable_size_
        return total_prefix_length
    
    def _generate_node_hash_value(self, node):
        node.hash_value = hash(tuple(node.prefix_key + [node.key[0]]))
    
    def _start_new_phase(self):
        logger.info(f"start a new phase, cache_size_k = {self.cache_size_k}, current_node_count: {TreeNode.counter - self.deleted_node_count}")
        print(f"start a new phase, current_node_count: {TreeNode.counter - self.deleted_node_count}")

        self.distinct_hash.clear()
        self.distinct_item_count = 0
        self.pred_evicted.clear()
        self.lru_evicted.clear()
        self.inv_count = 0

        # Collect all cached nodes (excluding root)
        all_nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node != self.root_node and not node.evicted:
                all_nodes.append(node)
                #node.match_tag = 0

            for child in node.children.values():
                if not child.evicted:
                    stack.append(child)

        self.U = set(all_nodes)

        #self.phase_err_param = int(math.sqrt(self.phase_err_param))
        #if self.lru_evict_count > 0 and self.pred_evict_count > 0 \
        #     and 2 * self.lru_rank_sum / self.lru_evict_count > self.pred_rank_sum / self.pred_evict_count:
        #    self.phase_err_param = self.phase_err_param / 2
        self.phase_err_param = max(self.phase_err_param / 2, 1)
        self.lru_budget = math.floor(self.lru_budget / 2)
        #self.lru_budget = self.lru_budget
    
    def _judge_evicted_in_phase(self, node: TreeNode):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return
        
        #if node.hash_value is not None:
        #    print(f"judge node hash_value = {node.hash_value}, current_ts = {self.current_ts}, NRT truth = {NRT_truth[node.hash_value]}")
        #    if self.current_ts != NRT_truth[node.hash_value]:
        #        print(f"!!!!!!!!!!!!!!!!!!! error !!!!!!!!!!!!!!!! of NRT truth")

        if node.hash_value in self.pred_evicted:
            #print(f"size of self.U = {len(self.U)}")
            #if len(self.U) > 0:
            rank = len(self.U)
            self.pred_evict_count += 1
            self.pred_rank_sum += rank
            logger.info(f"rank: {rank}, sum of inversions: {self.pred_rank_sum}, pred avg inv = {self.pred_rank_sum / self.pred_evict_count}")
            if self.algo_type == "blindoracle":
                self.lru_budget = 0
            else:
                self.phase_err_param = min(self.phase_err_param * 2, 100000000)
                self.lru_budget = min(self.lru_budget + self.phase_err_param, 100000000)
            print(f"current_ts = {self.current_ts}, hash_value = {node.hash_value}, reset lru_budget = {self.lru_budget}, phase_err_param = {self.phase_err_param}")
            
            logger.info(f"reset lru_budget = {self.lru_budget}, phase_err_param = {self.phase_err_param}")

        if node.hash_value in self.lru_evicted:
            #if len(self.U) > 0:
            rank = len(self.U)
            self.lru_evict_count += 1
            self.lru_rank_sum += rank
            logger.info(f"lru_evicted, rank: {rank}, sum of inversions: {self.lru_rank_sum}, lru avg inv = {self.lru_rank_sum / self.lru_evict_count}")

    def set_algo_type(self, algo_type):
        if self.algo_type != algo_type:
            self.algo_type = algo_type
            if algo_type == "lru":
                self.degrade_to_lru = True
                logger.info(f"Caching algorithm switches to lru")
            elif algo_type == "phaselru":
                self.degrade_to_lru = False
                logger.info(f"Caching algorithm switches to phaselru")
            elif algo_type == "belady":
                self.degrade_to_lru = False
                logger.info(f"Caching algorithm switches to belady")
            elif algo_type == "blindoracle":
                self.degrade_to_lru = False
                logger.info(f"Caching algorithm switches to blindoracle")
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

    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        if self.current_ts % 1000 == 0:
            logger.info(f"current ts: {self.current_ts}")
            captured = self._capture_print()
            logger.info(f"---------------------------------------------------- tree structure: {captured}")

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

    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        #self.dump_req_keys.append(token_ids)
        #if len(self.dump_req_keys) >= 20:
        #    file_name = str(self.rand) + '_dump_req_keys_' + str(self.dump_file_count) + '.pkl'
        #    self.dump_file_count += 1
        #    with open(file_name, 'wb') as f:
        #        pickle.dump(self.dump_req_keys, f)
        #    self.dump_req_keys = []

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

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}, number of nodes: {TreeNode.counter - self.deleted_node_count}")

    def _capture_print(self):
        buffer = io.StringIO()
        sys.stdout = buffer
        self.pretty_print()
        sys.stdout = sys.__stdout__
        return buffer.getvalue()

    def total_size(self):
        return self._total_size_helper()
 
    def belady_predict(self, hash):
        #print(f"NRT[{hash}] = {NRT[hash]}")
        if len(NRT[hash]) == 0:
            return 100000000
        return NRT[hash][0]

    def _predict(self, nodes: List[TreeNode]):
        if self.algo_type == "belady":
            for node in nodes:
                node.pred = self.belady_predict(node.hash_value)
                node.pred_valid = 1

            #for node in nodes:
            #    print(f"node hash_value = {node.hash_value}, pred = {node.pred}, truth = {str(NRT[node.hash_value])}")

            return

        node_to_pred = []
        addresses = []
        key_path_lens = []
        for node in nodes:
            #if node.pred_valid == 0:
            node_to_pred.append(node)
            addresses.append(node.hash_value)
            key_path_lens.append(node.key_path_len)

        if len(node_to_pred) > 0:
            preds = self.predictor.predict(addresses, key_path_lens, self.current_ts)
            for i in range(len(node_to_pred)):
                node = node_to_pred[i]
                if preds[i] == 2**62:
                    node.pred = -node.last_access_ts + preds[i]
                else:
                    node.pred = node.last_access_ts + preds[i]
                node.pred_valid = 1

        #for i in range(len(node_to_pred)):
         #   node = node_to_pred[i]
          #  print(f"node, pred = {node.pred}, truth = {NRT[node.hash_value][0] if len(NRT[node.hash_value]) > 0 else 10000000}, features = {self.predictor.get_features(node.hash_value, self.current_ts)}")

    def _evict_by_lru(self, num_tokens: int, based_on_budget: bool):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            if based_on_budget == True and self.lru_budget < 1:
                break

            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            if self.token_to_kv_pool_allocator:
                self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if based_on_budget == True:
                self.lru_evicted.add(x.hash_value)
                self.lru_budget -= 1

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

        return num_evicted
    
    def _evict_by_pred(self, num_tokens: int):
        leaves = self._collect_leaves()
        self._predict(leaves)

        #print(f"start pred")
        #for node in leaves:
        #    print(f"current ts = {self.current_ts}, leaf node pred = {node.pred}")
        #self.pretty_print()

        heap_by_pred = []
        for node in leaves:
            logger.info(f"leave match_tag = {node.match_tag}, hash_value = {node.hash_value}")
            #print(f"leave match_tag = {node.match_tag}, hash_value = {node.hash_value}")
            heapq.heappush(heap_by_pred, (node.match_tag, -node.pred, node))
            #heapq.heappush(heap_by_pred, (0, -node.pred, node))

        num_evicted = 0
        while num_evicted < num_tokens and len(heap_by_pred):
            _, _, x = heapq.heappop(heap_by_pred)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            print(f"eviction, current ts = {self.current_ts}, len of key = {len(x.key)}, hash = {x.hash_value}, pred = {x.pred}, truth = {NRT[x.hash_value][0] if len(NRT[x.hash_value]) > 0 else 10000000}, features = {self.predictor.get_features(x.hash_value, self.current_ts)}")

            if self.token_to_kv_pool_allocator:
                self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            #self.pred_evicted.add(hash(tuple(x.key)))
            self.pred_evicted.add(x.hash_value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent != self.root_node and x.parent.lock_ref == 0:
                self._predict([x.parent])
                heapq.heappush(heap_by_pred, (x.parent.match_tag, -x.parent.pred, x.parent))
                #heapq.heappush(heap_by_pred, (0, -x.parent.pred, x.parent))
        
        return num_evicted

    def evict(self, num_tokens: int):
        if self.disable:
            return

        #self.current_ts += 1
        if self.token_to_kv_pool_allocator:
            self.token_to_kv_pool_allocator.record_eviction(num_tokens)

        actual_evicted_num = 0
        if self.degrade_to_lru == True:
            actual_evicted_num = self._evict_by_lru(num_tokens, False)
            return actual_evicted_num
        
        if self.token_to_kv_pool_allocator:
            self.token_to_kv_pool_allocator.free_group_begin()

        #print(f"lru_budget: {self.lru_budget}")
        if self.algo_type == "belady":
            print("set lru_budget = 0")
            self.lru_budget = 0

        if  self.lru_budget >= 1:
            lru_evicted_num = self._evict_by_lru(num_tokens, True)
            actual_evicted_num += lru_evicted_num
            num_tokens -= lru_evicted_num

        if num_tokens > 0:
            pred_evicted_num = self._evict_by_pred(num_tokens)
            actual_evicted_num += pred_evicted_num
            num_tokens -= pred_evicted_num

        if self.token_to_kv_pool_allocator:
            self.token_to_kv_pool_allocator.free_group_end()

        return actual_evicted_num

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
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _split_node(self, hash_value, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(child.key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.prefix_key = child.prefix_key # added
        new_node.key_path_len = len(new_node.prefix_key) + len(new_node.key)
        new_node.match_tag = child.match_tag
        self._generate_node_hash_value(new_node)
        print(f"splot new_node hash v = {new_node.hash_value}")
        new_node.value = child.value[:split_len]
        new_node.parent.children[self.get_child_key_fn(new_node.key)] = new_node

        child.parent = new_node
        child.key = child.key[split_len:]
        child.prefix_key = child.prefix_key + new_node.key
        child.key_path_len = len(child.prefix_key) + len(child.key)
        self._generate_node_hash_value(child)
        child.value = child.value[split_len:]
        
        # update info for predictor after splitting
        self._predictor_split(hash_value, child, new_node)

        return new_node
    
    def _record_access(self, node: TreeNode, new_ts):
        node.access_times += 1
        node.last_access_ts = new_ts

        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return

        if node.hash_value not in self.distinct_hash:
            self.distinct_hash.add(node.hash_value)
            self.distinct_item_count += len(node.key)

        if self.distinct_item_count >= self.cache_size_k:
            self._start_new_phase()

        self.U.discard(node)
    
    def _predictor_access(self, node: TreeNode, current_ts):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return

        self.predictor.access(node.hash_value, node.key_path_len, current_ts)
        print(f"access, node hash = {node.hash_value}, key_path_len = {node.key_path_len}, key len = {len(node.key)}")
        node.pred_valid = 0

    def _predictor_split(self, original_hash, node: TreeNode, new_node: TreeNode):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return
        self._predictor_feature_copy(original_hash, node.hash_value)
        self._predictor_feature_copy(original_hash, new_node.hash_value)
        #self.predictor.feature_delete(original_hash)
        # copy pred from original node
        new_node.pred_valid = node.pred_valid
        new_node.pred = node.pred

    def _predictor_feature_copy(self, hash_value, new_hash_value):
        self.predictor.feature_copy(hash_value, new_hash_value)

    def _predictor_spawn(self, node: TreeNode, new_node: TreeNode):
        if self.degrade_to_lru == True or self.waiting_queue_cache == True:
            return

        if new_node.hash_value == 1399355274948978823:
            print(f"here: {str(node.key)}")
        self._predictor_feature_copy(node.hash_value, new_node.hash_value)
        # copy pred from parent node
        new_node.pred_valid = node.pred_valid
        new_node.pred = node.pred

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                "--" * current_indent,
                #f"node_id ({current_node.id}), depth ({current_indent / 2}), #keys {len(current_node.key)}",
                f"node_id ({current_node.id}), times {current_node.access_times}, pred {current_node.pred}, #keys {len(current_node.key)}",
                #current_node.key[:10],
                #f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)
        self.deleted_node_count += 1

    def _total_size_helper(self):
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
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to benchmark concurrent requests to a server."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="model path compatible with Hugging Face Transformers",
    )
    return parser.parse_args()

def load_multiturn_data(tokenizer):
    ret = []
    if os.path.exists("synthetic_multiturn_510_requests.pkl"):
        with open('synthetic_multiturn_510_requests.pkl', 'rb') as f:
            prompt_ids_list = pickle.load(f)
            for prompt_ids in prompt_ids_list:
                ret.append(tokenizer.encode(prompt_ids[1]['text']))
        print(f"total number of multiturn reqs: {len(ret)}")
    else:
        print(f"file not found")
    return ret

def load_poisson_multiturn_data(tokenizer):
    ret = []
    if os.path.exists("synthetic_poisson_multiturn_510_requests.pkl"):
        with open('synthetic_poisson_multiturn_510_requests.pkl', 'rb') as f:
            prompt_ids_list = pickle.load(f)
            for prompt_ids in prompt_ids_list:
                ret.append(tokenizer.encode(prompt_ids[1]['text']))
        print(f"total number of possion multiturn reqs: {len(ret)}")
    else:
        print(f"file not found")
    return ret

def load_qwen_bailian_data():
    ret = []
    if os.path.exists("qwen_traceA_blksz_16.jsonl"):
        with open('qwen_traceA_blksz_16.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # skip empty line
                    item = json.loads(line)
                    ret.append(item['hash_ids'])
    else:
        print(f"file not found")
    return ret

def load_stress_test_data(tokenizer):
    ret = []
    if os.path.exists("stress_test_token_id.pkl"):
        with open('stress_test_token_id.pkl', 'rb') as f:
            prompt_ids_list = pickle.load(f)
            for prompt_ids in prompt_ids_list:
                ret.append(prompt_ids)
                #print(f"prompt: {tokenizer.decode(prompt_ids)}")
        print(f"total number of stress reqs: {len(ret)}")
    else:
        print(f"file not found")
    return ret

if __name__ == "__main__":
    args = parse_args()
    tokenizer = get_tokenizer(args.model_path)

    tree = PhaseLRURadixCache(None, None, page_size=1, disable=False)
    #tree = GuardRadixCache(None, None, page_size=1, disable=False)

    #tree.set_algo_type("phaselru") #popu
    tree.set_algo_type("lru")
    #tree.set_algo_type("phaselru")
    #tree.set_algo_type("belady")
    #tree.set_algo_type("blindoracle")

    json_str = '{"enable_online_training": "on", "training_interval": 10000, "training_window": 100000}'
    obj = json.loads(json_str, object_hook=lambda d: SimpleNamespace(**d))
    #tree.set_online_training(obj)

    #sync_send_req_set = load_multiturn_data(tokenizer)
    #data_type = "multiturn"
    
    #sync_send_req_set = load_stress_test_data(tokenizer)
    #data_type = "stress"
    
    #sync_send_req_set = load_poisson_multiturn_data(tokenizer)
    #data_type = "poisson_multiturn"

    sync_send_req_set = load_qwen_bailian_data()
    data_type = "qwen_bailian"

    #if tree.algo_type == "belady":
    current_ts = 0
    for req in sync_send_req_set:
        current_ts += 1
        for j in range(len(req)):
            prefix_hash = hash(tuple(req[:j + 1]))
            if prefix_hash not in NRT:
                NRT[prefix_hash] = deque()
            NRT[prefix_hash].append(current_ts)

    #hash = -4232634994979945749
    #print(f"debug: {NRT[hash]}")

    #total_size = 153600
    #total_size = 76800 # 5%
    total_size = 30700 # 2%
    #total_size = 15360 # 1%
    
    tree.cache_size_k = total_size
    req_count = 0
    current_size = 0
    total_hit_id_count = 0
    total_req_id_count = 0
    total_req_len = 0
    for req in sync_send_req_set:
        if current_size + len(req) > total_size:
            #evicted_num = tree.evict(len(req))
            evicted_num = tree.evict(current_size + len(req) - total_size)
            current_size -= evicted_num
            print(f"evicted {len(req)}")

        # start access
        #if tree.algo_type == "belady":
        for j in range(len(req)):
            prefix_hash = hash(tuple(req[:j + 1]))
            #NRT_truth[prefix_hash] = NRT[prefix_hash].popleft()
            NRT[prefix_hash].popleft()

        prefix, _ = tree.match_prefix(req, init_req=True)
        total_hit_id_count += len(prefix)
        total_req_id_count += len(req)

        value = torch.zeros(len(req))
        tree.insert(req, value, True)
        current_size += len(req) - len(prefix)

        print(f"req_count = {req_count}, #req = {len(req)}, #prefix = {len(prefix)}, current_size = {current_size}")

        req_count += 1
        total_req_len += len(req)
        #captured = tree._capture_print()
        #logger.info(f"---------------------------------------------------- tree structure: {captured}")
        if req_count >= 1000000:
            break

    tree.pretty_print()
    print(f"total_size = {total_size}, req_count = {req_count}, total_req_len = {total_req_len}, algo = {tree.algo_type}, data type = {data_type}")
    print(f"Cache hit ratio: {total_hit_id_count / total_req_id_count}, avg hit id count = {total_hit_id_count / req_count}, avg req id count = {total_req_id_count / req_count}")

    #tree.insert("Hello")
    #tree.insert("Hello")
    #tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    #tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()

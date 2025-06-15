from sglang.srt.predictor.base_predictor import ReuseDistancePredictor
import time
import copy

class POPUPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.counts = {}
        self.base_time = time.monotonic()

    def split_copy(self, original_address, child_addr, parent_addr):
        # copy features from node with key = original_address
        self.counts[child_addr] = copy.deepcopy(self.counts[original_address])
        self.counts[parent_addr] = copy.deepcopy(self.counts[original_address])

    def feature_delete(self, address):
        del self.counts[address]

    def feature_copy(self, address, new_address):
        self.counts[new_address] = self.counts[address]

    def spawn_access(self, address, new_address):
        # copy features from parent node
        self.counts[new_address] = copy.deepcopy(self.counts[address])
        #self.access(new_address)

    def access(self, address, current_ts = None):
        if address not in self.counts:
            self.counts[address] = 0
        self.counts[address] += 1
    
    def predict(self, address):
        pred = (time.monotonic() - self.base_time) / self.counts[address]
        return pred
    
class BeladyPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()

    def split_copy(self, original_address, child_addr, parent_addr):
        pass

    def spawn_access(self, address, new_address):
        pass

    def access(self, address):
        pass
    
    def predict(self, address):
       return 0

class LRUPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.access_ts = {}

    def split_copy(self, original_address, child_addr, parent_addr):
        # copy features from node with key = original_address
        self.access_ts[child_addr] = self.access_ts[original_address]
        self.access_ts[parent_addr] = self.access_ts[original_address]

    def spawn_access(self, address, new_address):
        # copy features from parent node
        self.access_ts[new_address] = self.access_ts[address]

    def access(self, address, current_ts = None):
        self.access_ts[address] = time.monotonic()
    
    def predict(self, address):
        pred = -2 * self.access_ts[address]
        return pred
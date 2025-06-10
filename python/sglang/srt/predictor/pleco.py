from sglang.srt.predictor.base_predictor import ReuseDistancePredictor
import numpy as np

class PLECOPredictor(ReuseDistancePredictor):
    def __init__(self):
        super().__init__()
        self.timestamp = 1
        self.weights = []
        self.sum_weights = 0
        self.prev_occs = {}
        self.p = False

    def predict(self, address):
        this_weight = (self.timestamp + 10) ** (-1.8) * np.exp(-self.timestamp / 670)
        self.weights.append(this_weight)
        self.sum_weights += this_weight
        if address not in self.prev_occs:
            self.prev_occs[address] = []
        self.prev_occs[address].append(self.timestamp)
        prob = sum(self.weights[self.timestamp - i] for i in self.prev_occs[address]) / self.sum_weights
        pred = 1 / prob + self.timestamp - 1

        return pred
    
    def access(self, address):
        self.timestamp += 1
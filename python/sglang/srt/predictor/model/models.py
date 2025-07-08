import lightgbm as lgb
import numpy as np

class LightGBMModel:
    @classmethod
    def from_config(cls, deltanums, edcnums, model_file):
        return cls(deltanums, edcnums, model_file)

    def __init__(self, deltanums, edcnums, model_file):        
        self.model_ = lgb.Booster(model_file=model_file)
        self.deltanums = deltanums
        self.edcnums = edcnums
    
    def __call__(self, features):
        return self.forward(features)

    def forward(self, features):
        ypred = self.model_.predict(np.array(features, dtype=np.float64))
        return ypred
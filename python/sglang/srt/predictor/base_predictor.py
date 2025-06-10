from abc import ABC, abstractmethod
from typing import Union, List, Tuple

class Predictor(ABC):
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        '''
        Before evict, use predictor to refresh all slots' scores

        Scores can be reuse-distance, binary preds and cache's state(keys)

        When evicting, the scores must be the latest.
        '''
        raise NotImplementedError('Predictor: refresh_scores not implemented')
    
    def predict_score(self, ts, pc, address, cache_state) -> Union[int, float, str, None]:
        '''
        Predict this address's score, based on pc, address and cache_state.

        The score is only related to address, with the assistance of other variables

        '''
        raise NotImplementedError('Predictor: predict_score not implemented')

class ReuseDistancePredictor(Predictor):
    '''
    ReuseDistancePredictor only focus on address's score
    '''
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        return None 
    
class BinaryPredictor(Predictor):
    '''
    BinaryPredictor only focus on address's score 
    '''
    def refresh_scores(self, ts, pc, address, cache_state: Tuple[List, List]) -> List[Union[int, float, str]]:
        return None 
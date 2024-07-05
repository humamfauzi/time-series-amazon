from data_part import DataPart
from sklearn.metrics import mean_absolute_error
from typing import List
from sklearn.base import BaseEstimator  
from split_data import SplitData

class Model:
    def __init__(self, split_data: SplitData, predictor: BaseEstimator):
        self.split_data = split_data
        self.predictor = predictor

    def train(self):
        x, y = self.split_data.get_pair(DataPart.train)
        self.predictor.fit(x, y)
        return self
    
    def get_best_params(self):
        return self.predictor.best_params_

    def predict_and_score(self, data_part: DataPart) -> float:
        x, y = self.split_data.get_pair(data_part)
        y_pred = self.predictor.predict(x)
        return self.scoring(y, y_pred)

    def scoring(self, y_actual: List[float], y_pred: List[float]) -> float:
        return mean_absolute_error(y_actual, y_pred)

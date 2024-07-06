from data_part import DataPart
from sklearn.metrics import mean_absolute_error
from typing import List
from sklearn.base import BaseEstimator  
from split_data import SplitData

class ModelTimeSeries:
    def __init__(self, id: str, predictor: BaseEstimator):
        self.id = id
        self.predictor = predictor

    # TODO: Use time series CV
    def train(self, split_data: SplitData):
        x, y = split_data.get_pair(DataPart.train)
        self.predictor.fit(x, y)
        return self

    def predict_and_score(self, split_data: SplitData, data_part: DataPart) -> float:
        x, y = split_data.get_pair(data_part)
        y_pred = self.predictor.predict(x)
        return self.scoring(y, y_pred)

    def predict(self, x):
        return self.predictor.predict(x)

    def scoring(self, y_actual: List[float], y_pred: List[float]) -> float:
        return mean_absolute_error(y_actual, y_pred)
import time
from model import ModelTimeSeries
from data_part import DataPart
from split_data import SplitData
import pandas as pd
from typing import Optional, Dict, List

RAND_STATE = 123

class GridTimeSeries:
    def __init__(self, splt_data: SplitData):
        self.predictor: List[ModelTimeSeries] = []
        self.split_data = splt_data
        self.result_train = pd.DataFrame([], columns=['model', 'mae'])
        self.result_valid = pd.DataFrame([], columns=['model', 'mae'])
        self.train_time = pd.DataFrame([], columns=['model', 'time'])
        self.comparison = pd.DataFrame([], columns=['Train', 'Valid', 'Test'])
        self.trained_map: Dict[str, ModelTimeSeries] = {}

        self.best_model: Optional[ModelTimeSeries] = None
        
    def assign_model(self, pred: ModelTimeSeries):
        self.predictor.append(pred)
        return self

    def add_result_train(self, name: str, mae: float):
        self.result_train.loc[len(self.result_train)] = [name, mae]
        return self

    def add_result_valid(self, name: str, mae: float):
        self.result_valid.loc[len(self.result_valid)] = [name, mae]
        return self

    def add_train_time(self, name: str, time: float):
        self.train_time.loc[len(self.train_time)] = [name, time]
        return self

    def add_comparison(self, name: str, mae: List[float]):
        self.comparison.loc[name] = mae
        return self

    def run(self):
        for pred in self.predictor:
            start_time = time.time()
            trained = pred.train(self.split_data)
            train_mae = trained.predict_and_score(self.split_data, DataPart.train)
            valid_mae = trained.predict_and_score(self.split_data, DataPart.valid)
            elapsed = time.time() - start_time

            (self
                .add_result_train(pred.id, train_mae)
                .add_result_valid(pred.id, valid_mae)
                .add_train_time(pred.id, elapsed))
            self.trained_map[pred.id] = trained
        return self
    
    def show_result(self):
        print("Train Time")
        print(self.train_time.sort_values('time'))
        print("Train Result")
        print(self.result_train.sort_values('mae'))
        print("Valid Result")
        print(self.result_valid.sort_values('mae'))
        return self

    def pick_best_result(self):
        best_model = self.result_valid.sort_values('mae').iloc[0]
        best_model = self.trained_map[best_model['model']]
        print("Best Model", best_model.id)
        return self

    def run_test_data(self):
        best_model = self.result_valid.sort_values('mae').iloc[0]
        best_model = self.trained_map[best_model['model']]
        train_result = best_model.predict_and_score(self.split_data, DataPart.train)
        valid_result = best_model.predict_and_score(self.split_data, DataPart.valid)
        test_result = best_model.predict_and_score(self.split_data, DataPart.test)
        self.add_comparison(best_model.id, [train_result, valid_result, test_result])
        self.best_model = best_model

        baseline_model = self.trained_map['baseline']
        train_result = baseline_model.predict_and_score(self.split_data, DataPart.train)
        valid_result = baseline_model.predict_and_score(self.split_data, DataPart.valid)
        test_result = baseline_model.predict_and_score(self.split_data, DataPart.test)
        self.add_comparison(baseline_model.id, [train_result, valid_result, test_result])
        return self

    def show_final_result(self):
        print(self.comparison)
        return self

    # TODO: Using the prediction in as a previous quantity 
    def measure_continuous_prediction(self, test_date):
        pass
        

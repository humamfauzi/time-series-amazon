from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
from model import Model, ModelTimeSeries
from data_part import DataPart
from split_data import SplitData
import pandas as pd
from typing import Optional, Dict, List

RAND_STATE = 123

class Grid:
    def __init__(self, split_data: SplitData):
        self.split_data = split_data
        self.result_valid = pd.DataFrame([], columns=['model', 'mae' ])
        self.result_train = pd.DataFrame([], columns=['model', 'mae'])
        self.train_time = pd.DataFrame([], columns=['model', 'time'])
        self.train_best_params = pd.DataFrame([], columns=['model', 'best_params'])

        self.best_model: Optional[Model] = None
        self.best_model_identifier: Optional[str] = None
        self.trained: Dict[str, Model] = {}

    def run(self):
        start_time = time.time()
        self.prediction_run(DummyRegressor(), {'strategy': ['median']}, "Baseline")
        self.prediction_run(LinearRegression(), {}, "Linear Regression")
        dt_grid = {
            'max_depth': [None, 10, ],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': [RAND_STATE],
        }
        self.prediction_run(DecisionTreeRegressor(), dt_grid, "Decision Tree")
        self.prediction_run(KNeighborsRegressor(), {}, "K-Nearest Neighbour")

        # Ensemble Classification
        self.prediction_run(RandomForestRegressor(), {}, "Random Forest")
        self.prediction_run(GradientBoostingRegressor(), {}, "Gradient Boosting")

        end_time = time.time()
        print(f"fitting done in {end_time - start_time:.4f} seconds")
        return self

    def prediction_run(self, pred: BaseEstimator, grid: Dict[str, any], name: str):
        start = time.time()
        # TODO: Use time series CV
        pred = GridSearchCV(pred, grid, scoring='neg_mean_absolute_error', cv=10)
        trained = Model(self.split_data, pred).train()
        train_result = trained.predict_and_score(DataPart.train)
        valid_result = trained.predict_and_score(DataPart.valid)
        self.result_train.loc[len(self.result_train)] = [name, train_result]
        self.result_valid.loc[len(self.result_valid)] = [name, valid_result]
        self.train_best_params.loc[len(self.train_best_params)] = [name, trained.get_best_params()]
        self.trained[name] = trained
        elapsed_time = time.time() - start
        self.train_time.loc[len(self.result_train)] = [name, elapsed_time]
        return True

    def pick_best_result(self):
        best_model = self.result_valid.sort_values('mae').iloc[0]
        self.best_model_identifier = best_model['model']
        self.best_model = self.trained[best_model['model']]
        return self

    def run_test_data(self):
        # rerun all part 
        train_result = self.best_model.predict_and_score(DataPart.train)
        valid_result = self.best_model.predict_and_score(DataPart.valid)
        test_result = self.best_model.predict_and_score(DataPart.test)
        comparison = pd.DataFrame([train_result, valid_result, test_result], columns=['mae'], index=['Train', 'Valid', 'Test'])
        print("Best Model", self.best_model_identifier)
        print(comparison)
    
    def show_result(self):
        print("Train Time")
        print(self.train_time.sort_values('time'))
        print("Train Result")
        print(self.result_train.sort_values('mae'))
        print("Valid Result")
        print(self.result_valid.sort_values('mae'))
        print("Best Params")
        print(self.train_best_params)
        return self

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
        

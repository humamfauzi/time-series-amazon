from data_part import DataPart
import copy

class SplitData:
    def __init__(self, X_train, X_valid, X_test, y_train, y_valid, y_test):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test

        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

    def total_data(self, data):
        if data == DataPart.train.name:
            return self.X_train.shape[0]
        if data == DataPart.valid.name:
            return self.X_valid.shape[0]
        if data == DataPart.test.name:
            return self.X_test.shape[0]

    def get_pair(self, data_pair):
        if data_pair == DataPart.train:
            if self.X_train is None or self.y_train is None:
                raise ValueError("X train or y train is not exist")
            return copy.deepcopy(self.X_train), copy.deepcopy(self.y_train)
        if data_pair == DataPart.valid:
            if self.X_valid is None or self.y_valid is None:
                raise ValueError("X valid or y valid is not exist")
            return copy.deepcopy(self.X_valid), copy.deepcopy(self.y_valid)
        if data_pair == DataPart.test:
            if self.X_test is None or self.y_test is None:
                raise ValueError("X test or y test is not exist")
            return copy.deepcopy(self.X_test), copy.deepcopy(self.y_test)
        raise ValueError(f"Unknown value {data_pair}")
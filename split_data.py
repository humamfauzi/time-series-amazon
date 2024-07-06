from data_part import DataPart
import copy

class SplitData:
    def __init__(self, x_train, x_valid, x_test, y_train, y_valid, y_test):
        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test

        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

    def total_data(self, data):
        if data == DataPart.train.name:
            return self.x_train.shape[0]
        if data == DataPart.valid.name:
            return self.x_valid.shape[0]
        if data == DataPart.test.name:
            return self.x_test.shape[0]

    def get_pair(self, data_pair):
        if data_pair == DataPart.train:
            if self.x_train is None or self.y_train is None:
                raise ValueError("X train or y train is not exist")
            return copy.deepcopy(self.x_train), copy.deepcopy(self.y_train)
        if data_pair == DataPart.valid:
            if self.x_valid is None or self.y_valid is None:
                raise ValueError("X valid or y valid is not exist")
            return copy.deepcopy(self.x_valid), copy.deepcopy(self.y_valid)
        if data_pair == DataPart.test:
            if self.x_test is None or self.y_test is None:
                raise ValueError("X test or y test is not exist")
            return copy.deepcopy(self.x_test), copy.deepcopy(self.y_test)
        raise ValueError(f"Unknown value {data_pair}")
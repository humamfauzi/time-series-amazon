import pandas as pd
from column import Column
from preprocessing import DataPreprocessing
from split_data import SplitData
from sklearn.model_selection import train_test_split
from data_part import DataPart

RAND_STATE = 123

class SplitPreprocessData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        self.preprocessing_properties = None

        self.train = None
        self.valid = None
        self.test = None

        self.X_train = None
        self.X_valid = None
        self.X_test = None

        self.y_train = None
        self.y_valid = None
        self.y_test = None

    def split_only(self):
        return self.split_data()

    def input_output_split(self):
        source = [col for col in self.train.columns if col != Column.target()]
        self.X_train = self.train[source]
        self.X_valid = self.valid[source]
        self.X_test = self.test[source]

        self.y_train = self.train[Column.target()]
        self.y_valid = self.valid[Column.target()]
        self.y_test = self.test[Column.target()]
        return self

    def split_data(self):
        if self.train is not None:
            return self
        data_train, data_non_train = train_test_split(self.df, test_size=.2, random_state=RAND_STATE)
        data_test, data_valid = train_test_split(data_non_train, test_size=.5, random_state=RAND_STATE)
        self.train = data_train
        self.valid = data_valid
        self.test = data_test
        return self

    def get_train_data(self):
        return self.train

    def split_preprocess(self):
        return (self.split_data()
            .preprocess_train()
            .apply_preprocess_valid()
            .apply_preprocess_test()
            .input_output_split()
            .get_split_data()
        )

    def preprocess_train(self):
        train = DataPreprocessing(self.train, None).preprocessing_reguler()
        self.train = train.df
        self.preprocessing_properties = train.export_preprocessing_properties()
        return self

    def apply_preprocess_valid(self):
        if self.preprocessing_properties is None:
            raise ValueError("preprocessing properties is not exist")
        self.valid = (DataPreprocessing(self.valid, self.preprocessing_properties)
            .preprocessing_reguler()
            .df)
        return self

    def apply_preprocess_test(self):
        if self.preprocessing_properties is None:
            raise ValueError("preprocessing properties is not exist")
        self.test = (DataPreprocessing(self.test, self.preprocessing_properties)
            .preprocessing_reguler()
            .df)
        return self

    def get_split_data(self):
        return SplitData(self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test)
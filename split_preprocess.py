import pandas as pd
from column import Column, ColumnTimeSeries
from preprocessing import DataPreprocessing, PreprocessTimeSeries
from split_data import SplitData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_part import DataPart
from typing import List, Optional

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



class SplitPreprocessTimeSeriesData:
    def __init__(self, df: pd.DataFrame, desired_sku: List[str]):
        self.df = df
        self.desired_sku = desired_sku
        self.post_df_columns = [
            ColumnTimeSeries.date.name,
            ColumnTimeSeries.simplified_sku.name,
            ColumnTimeSeries.current_quantity.name,
        ]
        # only for processing
        self.post_df = pd.DataFrame([], columns=self.post_df_columns)

        self.ts_df = pd.DataFrame([], columns=ColumnTimeSeries.all())

        self.standard_scaler: Optional[StandardScaler] = None

    def get_specific_sku(self, sku: str):
        return self.ts_df[self.ts_df[Column.simplified_sku.name] == sku]
    
    def convert_date_to_datetime(self):
        self.df[Column.date.name] = pd.to_datetime(self.df[Column.date.name])
        return self
    
    def create_simplified_sku(self):
        self.df[Column.simplified_sku.name] = pd.Series([ i.split("-")[0] for i in self.df[Column.stock_keeping_unit.name]], index=self.df.index)
        return self
    
    def date_sku_mask(self, date, sku):
        return (self.df[Column.date.name] == date) & (self.df[Column.simplified_sku.name] == sku)

    def date_sku_mask_post(self, date, sku):
        return (self.post_df[ColumnTimeSeries.date.name] == date) & (self.post_df[ColumnTimeSeries.simplified_sku.name] == sku)

    def group_by_sku(self):
        ordered_unique_date = list(self.df[Column.date.name].sort_values().unique())
        for date in ordered_unique_date:
            for sku in self.desired_sku:
                mask = self.date_sku_mask(date, sku)
                total_quantity_sku_in_day = self.df[mask][Column.target()].sum()
                current_count = len(self.post_df)
                self.post_df.loc[current_count] = [date, sku, total_quantity_sku_in_day]
        return self

    def reorder_data(self):
        self.post_df = self.post_df.sort_values(by=ColumnTimeSeries.date.name)
        return self
    
    def rearrange_data(self):
        def get_all_quantity_five_days_prior(date, sku):
            return [
                self.post_df[self.date_sku_mask_post(date - pd.Timedelta(days=1), sku)][ColumnTimeSeries.quantity_scaled.name].values[0],
                self.post_df[self.date_sku_mask_post(date - pd.Timedelta(days=2), sku)][ColumnTimeSeries.quantity_scaled.name].values[0],
                self.post_df[self.date_sku_mask_post(date - pd.Timedelta(days=3), sku)][ColumnTimeSeries.quantity_scaled.name].values[0],
                self.post_df[self.date_sku_mask_post(date - pd.Timedelta(days=4), sku)][ColumnTimeSeries.quantity_scaled.name].values[0],
                self.post_df[self.date_sku_mask_post(date - pd.Timedelta(days=5), sku)][ColumnTimeSeries.quantity_scaled.name].values[0],
            ]
        all_date = self.post_df[ColumnTimeSeries.date.name].sort_values().unique()
        all_sku = self.post_df[ColumnTimeSeries.simplified_sku.name].unique()
        highlighted_date = all_date[5:]
        for date in highlighted_date:
            for sku in all_sku:
                prev_quantity = get_all_quantity_five_days_prior(date, sku)
                masked = self.post_df[self.date_sku_mask_post(date, sku)]
                quantity_scaled = masked[ColumnTimeSeries.quantity_scaled.name].values[0]
                quantity_actual = masked[ColumnTimeSeries.current_quantity.name].values[0]
                current_count = len(self.ts_df)
                assign = [date, sku, quantity_actual, quantity_scaled] + prev_quantity 
                self.ts_df.loc[current_count] = assign
        print(self.ts_df.head()) 
        return self

    def split_dates(self):
        all_date = self.post_df[ColumnTimeSeries.date.name].sort_values().unique()
        self.test_dates = all_date[-3:] # try to predict three days
        self.valid_dates = all_date[-6:-3] # try to validate three days
        self.train_dates = all_date[:-6] # the rest is training data
        return self

    def split_train_test(self):
        self.train = self.ts_df[self.ts_df[ColumnTimeSeries.date.name].isin(self.train_dates)]
        self.valid = self.ts_df[self.ts_df[ColumnTimeSeries.date.name].isin(self.valid_dates)]
        self.test = self.ts_df[self.ts_df[ColumnTimeSeries.date.name].isin(self.test_dates)]
        return self
    
    def get_dates(self):
        return self.train_dates, self.valid_dates, self.test_dates

    def get_train_scaling(self):
        picked = self.post_df[self.post_df[ColumnTimeSeries.date.name].isin(self.train_dates)]
        dp = (PreprocessTimeSeries(picked[ColumnTimeSeries.current_quantity.name], None)
            .rescale_fit())
        self.standard_scaler = dp.get_scaler()
        return self

    def apply_scaling_to_all(self):
        if self.standard_scaler is None:
            raise ValueError("StandardScaler is not fitted")
        self.post_df[ColumnTimeSeries.quantity_scaled.name] = (PreprocessTimeSeries(self.post_df[ColumnTimeSeries.current_quantity.name], self.standard_scaler)
            .transform()
            .get_series())
        return self

    def get_data(self):
        x_train = self.train[ColumnTimeSeries.train()]
        y_train = self.train[ColumnTimeSeries.target()]

        x_valid = self.valid[ColumnTimeSeries.train()]
        y_valid = self.valid[ColumnTimeSeries.target()]

        x_test = self.test[ColumnTimeSeries.train()]
        y_test = self.test[ColumnTimeSeries.target()]
        return SplitData(x_train, x_valid, x_test, y_train, y_valid, y_test)
    
    def get_dates(self):
        return self.train_dates, self.valid_dates, self.test_dates

    
import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from column import Column, ColumnTimeSeries
from typing import Optional

class DataPreprocessing:
    def __init__(self, df, previous):
        self.seed = 123
        self.ohe = None

        if previous is not None:
            self.ohe = previous['ohe']

        self.df = copy.deepcopy(df)
    
    # sequence of reprocessing.
    def preprocessing_reguler(self):
        print("original shape", self.df.shape)

        (self.remove_no_location_rows()
        .remove_no_target()
        .set_order_id_as_index()
        .create_is_promotion_column()
        .convert_date_to_datetime()
        .create_week_column()
        .create_month_column()
        .drop_date_column()
        .clean_ship_state()
        .remove_cancelled_shipment()
        .create_ohe_categorical()
        .drop_unused_column()
        )

        print("final shape", self.df.shape)
        return self

    def set_order_id_as_index(self):
        self.df = self.df.set_index(Column.order_id.name)
        return self   

    def remove_no_location_rows(self):
        self.df = self.df[self.df[Column.ship_city.name].notna()]
        return self   

    def remove_no_target(self):
        self.df = self.df[self.df[Column.target()].notna()]
        return self   

    def drop_unused_column(self):
        self.df = self.df.drop(Column.dropped(), axis=1)
        return self   

    def create_is_promotion_column(self):
        promo = pd.Series([1 if x else 0  for x in self.df[Column.promotion_id.name].isna()], index=self.df.index)
        self.df[Column.is_promotion.name] = promo
        return self   

    def convert_date_to_datetime(self):
        self.df[Column.date.name] = pd.to_datetime(self.df[Column.date.name], format="%m-%d-%y", dayfirst=True)
        return self   

    def create_week_column(self):
        dates = self.df[Column.date.name]
        self.df = self.df.assign(week=lambda x: dates.dt.day // 7)
        return self   

    def create_month_column(self):
        dates = self.df[Column.date.name]
        self.df = self.df.assign(month=lambda x: dates.dt.month)
        return self   

    def drop_date_column(self):
        self.df = self.df.drop([Column.date.name], axis=1)
        return self   

    def clean_ship_state(self):
        def transform_state(state_name):
            mapped = {
                # manual cleaning, Vehicle Registration Code
                "RJ": "RAJASTHAN",
                "AR": "ARUNACHAL PRADESH",
                "PB": "PUNJAB",
                "NL": "NAGALAND",
                "APO": "ANDHRA PRADESH",
                # Area naming
                "PUNJAB/MOHALI/ZIRAKPUR": "PUNJAB",
                "NEW DELHI": "DELHI",
                # Colonial naming
                "PONDICHERRY": "PUDUCHERRY",
                "ORISSA": "ODISHA",
                # Mispelling
                "RAJSHTHAN": "RAJASTHAN",
                "RAJSTHAN": "RAJASTHAN"
            }
            upper = state_name.upper()
            if upper in mapped:
                return mapped[upper]
            return upper
        modified = pd.Series([transform_state(i) for i in self.df[Column.ship_state.name]], index=self.df.index)
        self.df[Column.ship_state.name] = modified
        return self   

    def remove_cancelled_shipment(self):
        self.df = self.df[self.df[Column.courier_status.name] != "Cancelled"]
        return self   

    def create_ohe_categorical(self):
        col_ohe = Column.require_ohe()
        if self.ohe is None:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.ohe.fit(self.df[col_ohe])
        transform = self.ohe.transform(self.df[col_ohe])
        new_ohe_df = pd.DataFrame(transform, columns=self.ohe.get_feature_names_out(Column.require_ohe()), index=self.df.index)
        self.df = pd.concat([self.df, new_ohe_df], axis=1).drop(col_ohe, axis=1)
        return self   

    def export_preprocessing_properties(self):
        return {
            'ohe': self.ohe
        }



class PreprocessTimeSeries:
    def __init__(self, df: pd.DataFrame, prev: StandardScaler):
        self.series = copy.deepcopy(df)
        self.array = np.array(self.series).reshape(-1, 1)
        self.standard_scaler: Optional[StandardScaler] = prev

    def rescale_fit(self):
        ss = StandardScaler().fit(self.array)
        self.standard_scaler = ss
        return self

    def transform(self):
        self.series = pd.Series(
            self.standard_scaler.transform(self.array).reshape(-1),
                index = self.series.index)
        return self

    def get_scaler(self):
        return self.standard_scaler

    def get_series(self):
        return self.series


from typing import List
from column import Column
import pandas as pd
import pprint
class ExplanatoryDataAnalysis:
    def __init__(self, df: pd.DataFrame):
        # does not need deepcopy because no modification in EDA
        self.df = df
        self.sku: List[str] = []

    def check_na(self):
        # check each column for NaN value
        pprint.pprint({
            "categorical":dict(self.df[Column.categorical()].isna().sum(0)),
            "item_identifier":dict(self.df[Column.item_identifier()].isna().sum(0)),
            "date":self.df[Column.date.name].isna().sum(0),
            "target":self.df[Column.target()].isna().sum(0)
        })
        return self

    # categorical value have possible value
    # this methods would print all for each column with its counts
    def categorical_possible_value(self):
        for column in Column.categorical():
            print(self.df[column].value_counts())
            print("")
        return self
    
    def check_sku_date(self):
        self.df[Column.date.name] = pd.to_datetime(self.df[Column.date.name])
        unique_date = list(self.df[Column.date.name].sort_values().unique())
        self.df[Column.simplified_sku.name] = pd.Series([ i.split("-")[0] for i in self.df[Column.stock_keeping_unit.name]], index=self.df.index)
        all_sku = set(self.df[self.df[Column.date.name] == unique_date[0]][Column.simplified_sku.name].unique())
        for date in unique_date[1:]:
            unique_sku = set(self.df[self.df[Column.date.name] == date][Column.simplified_sku.name].unique())
            all_sku = all_sku.intersection(unique_sku)
        self.sku = list(all_sku)
        return self
    
    def get_all_desired_sku(self):
        return self.sku
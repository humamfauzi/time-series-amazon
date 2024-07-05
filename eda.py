from column import Column
import pprint
class ExplanatoryDataAnalysis:
    def __init__(self, df):
        # does not need deepcopy because no modification in EDA
        self.df = df
        self.loan_interest_map = {}

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
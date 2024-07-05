import pandas as pd 
from column import Column
import copy

class Loader():
    def __init__(self):
        self.gdrive_main_dir = '/content/drive/MyDrive/dataset/retail_transaction/'
        self.file_name = 'amazon_sales_report.csv'

        self.df = None

    def swap_column(self):
        if self.df is None:
            return self
        mapped_enum = [column.name for column in Column]
        column_map = {column: mapped_enum[index] for index, column in enumerate(list(self.df.columns))}
        self.df = self.df.rename(columns=column_map)

    def load_via_disk(self):
        if self.df is None:
            self.df = pd.read_csv(self.file_name)
            self.swap_column()
        return self

    def load_via_gdrive(self):
        if self.df is None:
            self.df = pd.read_csv(self.gdrive_main_dir + self.file_name)
            self.swap_column()
        return self

    def profile(self):
        print("available column", list(self.df.columns))
        print("head", self.df.head())
        print("shapes", self.df.shape)
        return self

    def get_df(self):
        return copy.deepcopy(self.df)

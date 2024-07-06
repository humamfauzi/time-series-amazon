import copy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional

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


from loader import Loader
from grid import GridTimeSeries
from model import ModelTimeSeries
from split_preprocess import SplitPreprocessTimeSeriesData
from eda import ExplanatoryDataAnalysis
from column import ColumnTimeSeries
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


def alternative_main():
    ld = (Loader()
        # loading data via disk and store it inside class container
        .load_via_disk()

        # display basic properties of the data
        .profile()

        # load the data as a variable
        .get_df())
    desired_sku = (ExplanatoryDataAnalysis(ld)
        # check for NaN value
        .check_na()

        # checking sku items that ordered for each day
        # this would be chosen item for time series forecasting
        .check_sku_date()
        
        # after checking, we want to know the sku that we want to forecast
        .get_all_desired_sku())

    # Splitting data for time series has different method and sequence
    # therefore we need to use different class
    sptsd = (SplitPreprocessTimeSeriesData(ld, desired_sku)
        # convert date time to datetime object for better handling
        .convert_date_to_datetime()

        # all sku contain extra information, we want to simplify it
        # so we can group it
        .create_simplified_sku()

        # grouping all sku that we want to forecast grouped quantity would get summed
        .group_by_sku()

        # reorder the data data so it sequential based on its date
        .reorder_data()

        # split the dates
        .split_dates()

        # fit rescaling only in train data
        .get_train_scaling()
        
        # apply rescaling to all data
        .apply_scaling_to_all()

        # rearrange data so the column would be the previous quantity of the sku
        .rearrange_data()

        # split the data into train, valid and test
        .split_train_test()
        )

    # get the data for modeling
    split_data = sptsd.get_data()

    # Parameter and model search is also different, so it also need
    # different class
    (GridTimeSeries(split_data)
        # assign model to be compared
        .assign_model(ModelTimeSeries("baseline", DummyRegressor()))
        .assign_model(ModelTimeSeries("linear_regression", LinearRegression()))
        .assign_model(ModelTimeSeries("decision_tree", DecisionTreeRegressor()))
        .assign_model(ModelTimeSeries("knn", KNeighborsRegressor()))
        .assign_model(ModelTimeSeries("gradient_boosting", GradientBoostingRegressor()))

        # run assigned model
        .run()

        # show the result of the training
        .show_result()

        # based on the latest run, get the best model and
        # try to run it on test data
        .run_test_data()

        # show the result of the test data
        .show_final_result())
    
if __name__ == "__main__":
    alternative_main()
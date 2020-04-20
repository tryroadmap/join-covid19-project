import train
import logistic_regression
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
# ML libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
le = preprocessing.LabelEncoder()

def calculate_trend(df, lag_list, column):
    for lag in lag_list:
        trend_column_lag = "Trend_" + column + "_" + str(lag)
        df[trend_column_lag] = (df[column]-df[column].shift(lag, fill_value=-999))/df[column].shift(lag, fill_value=0)
    return df


def calculate_lag(df, lag_list, column):
    for lag in lag_list:
        column_lag = column + "_" + str(lag)
        df[column_lag] = df[column].shift(lag, fill_value=0)
    return df


# Run the model for Spain
def main():
	TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA")
	TEST_DATA = os.environ.get("TEST_DATA")
	
	train_data = pd.read_csv(TRAINING_DATA_DIR)
	test = pd.read_csv(TEST_DATA)

	add_columns = train.addingColumns(train_data,test)




	data,country_dict,all_data = train.addingWolrd(add_columns)

	dates_list = ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', 
                 '2020-03-10', '2020-03-11','2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',
                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24']


	country_name = os.environ.get("COUNTRY")
	# country_name = 'Spain'
	day_start = 35 
	lag_size = 30

	data = logistic_regression.lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

	logistic_regression.plot_real_vs_prediction_country(data, train_data, country_name, 39, dates_list)

	logistic_regression.plot_real_vs_prediction_country_fatalities(data, train_data, country_name, 39, dates_list)

	# ts = time.time()

	# Inputs
	# country_name = "Italy"
	# day_start = 35 
	# lag_size = 30

	# data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)
	# plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)
	# plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)



if __name__ == "__main__":
	main()


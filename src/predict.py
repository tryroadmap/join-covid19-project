import train

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


def main():

	# test = os.environ.get("TEST_DATA")
	# train_data = os.environ.get("TRAINING_DATA")

	TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA")
	TEST_DATA = os.environ.get("TEST_DATA")
	
	train_data = pd.read_csv(TRAINING_DATA_DIR)
	test = pd.read_csv(TEST_DATA)

	add_columns = train.addingColumns(train_data,test)
	data,country_dict,all_data = train.addingWolrd(add_columns)

	# le = preprocessing.LabelEncoder()

	# Select train (real) data from March 1 to March 22nd

	dates_list = ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', 
	                 '2020-03-10', '2020-03-11','2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',
	                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24']

	# Filter Spain, run the Linear Regression workflow
	# country_name = "Spain"	country_name = "Spain"
# 
	country_name = os.environ.get("COUNTRY")


	day_start = 39
	data_country = data[data['Country/Region']==country_dict[country_name]]
	data_country = data_country.loc[data_country['Day_num']>=day_start]
	X_train, Y_train_1, Y_train_2, X_test = train.split_data(data_country)
	model, pred = train.lin_reg(X_train, Y_train_1, X_test)

	# Create a df with both real cases and predictions (predictions starting on March 12th)
	X_train_check = X_train.copy()
	X_train_check['Target'] = Y_train_1

	X_test_check = X_test.copy()
	X_test_check['Target'] = pred

	X_final_check = pd.concat([X_train_check, X_test_check])

	# Select predictions from March 1st to March 24th
	predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target
	real_data = train_data.loc[(train_data['Country/Region']==country_name) & (train_data['Date'].isin(dates_list))]['ConfirmedCases']
	dates_list_num = list(range(0,len(dates_list)))


	# Plot results
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

	ax1.plot(dates_list_num, np.exp(predicted_data))
	ax1.plot(dates_list_num, real_data)
	ax1.axvline(10, linewidth=2, ls = ':', color='grey', alpha=0.5)
	ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
	ax1.set_xlabel("Day count (from March 1st to March 22nd)")
	ax1.set_ylabel("Confirmed Cases")

	ax2.plot(dates_list_num, predicted_data)
	ax2.plot(dates_list_num, np.log(real_data))
	ax2.axvline(10, linewidth=2, ls = ':', color='grey', alpha=0.5)
	ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
	ax2.set_xlabel("Day count (from March 1st to March 22nd)")
	ax2.set_ylabel("Log Confirmed Cases")

	plt.suptitle(("ConfirmedCases predictions based on Linear Regression for "+country_name))

	plt.show()


if __name__ == "__main__":

	main()

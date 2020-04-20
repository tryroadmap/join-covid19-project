import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
# ML libraries
import lightgbm as lgb
import xgboost as xgb

import librosa.display





TRAINING_DATA_DIR = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")

train_data = pd.read_csv(TRAINING_DATA_DIR)
test = pd.read_csv(TEST_DATA)

print(train_data.head(5))
print(train_data.describe())
print("Number of Country/Region: ", train_data['Country/Region'].nunique())
print("Dates go from day", min(train_data['Date']), "to day", max(train_data['Date']), ", a total of", train_data['Date'].nunique(), "days")
print("Countries with Province/State informed: ", train_data[train_data['Province/State'].isna()==False]['Country/Region'].unique())







# data manipulation
import pandas as pd

# mathematical functions
import numpy as np
from scipy.stats import randint, uniform
import random

# data visualization
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.subplots as sp
from plotly.subplots import make_subplots


# data splitting
from sklearn.model_selection import train_test_split

# data preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from category_encoders import BinaryEncoder
from imblearn.over_sampling import SMOTE

# algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# model training requirements
import warnings
from sklearn.model_selection import RandomizedSearchCV, KFold

# model evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# import logging
import logging

# save model
import joblib


def load_data():
    try:
        
        # Load application_record.csv
        df1 = pd.read_csv("C:/Users/aswinram/Aswin's Data Science Portfolio/Credit Card Approval Prediction/data/application_record.csv")
        # Remove spaces in columns name
        df1.columns = df1.columns.str.replace(' ','_')

        # Load credit_record.csv
        df2 = pd.read_csv("C:/Users/aswinram/Aswin's Data Science Portfolio/Credit Card Approval Prediction/data/credit_record.csv")
        # Remove spaces in columns name
        df2.columns = df2.columns.str.replace(' ','_')
        
        print("Loading data successful")
        
        return df1, df2
    
    except Exception as e:
        print("Error loading data:", e)

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



def split_data(df, target_feature):
    """
    Split the given dataframe into training, validation, and test sets based on the specified target feature.

    Parameters:
    -----------
    df: pandas dataframe
        The dataframe containing the data to split.
    target_feature: str
        The name of the target feature to use for the split.

    Returns:
    --------
    The training, validation, and test sets in the following order:
    X_train, y_train, X_val, y_val, X_test, y_test
    
    """
    
    try:
        # Assign input features (also for feature selection)
        X = df[['CODE_GENDER', 'FLAG_EMAIL', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
        'FLAG_PHONE', 'FLAG_WORK_PHONE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
        'NAME_INCOME_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED']]

        # Assign target feature
        y = df[target_feature]

        # Perform stratified train_val-test split for input features
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

        # Further split the training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=0)
        
        
        print("Data splitting successful")

        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        raise ValueError("Failed to split data:", str(e))

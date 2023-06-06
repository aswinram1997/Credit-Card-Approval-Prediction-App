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


def train_best_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test):
    """
    Train the best model on the entire dataset.

    Parameters:
    -----------
    X_train_scaled: pandas dataframe
        The scaled training features.
    y_train: pandas series
        The training labels.
    X_val_scaled: pandas dataframe
        The scaled validation features.
    y_val: pandas series
        The validation labels.
    X_test_scaled: pandas dataframe
        The scaled test features.
    y_test: pandas series
        The test labels.

    Returns:
    --------
    The trained best model.
    """

    # create Best model - Random Forest
    best_model = RandomForestClassifier(max_depth=35, max_features='log2', min_samples_leaf=2,
                       min_samples_split=4, n_estimators=25, random_state=0)

    # Train on entire dataset

    # Vertically concatenate the dataframes
    X_all = pd.concat([X_train_scaled, X_val_scaled, X_test_scaled], axis=0)

    # Vertically concatenate the pandas series
    y_all = pd.concat([y_train, y_val, y_test], axis=0)

    try:
        # fit the best_model on entire dataset
        best_model = best_model.fit(X_all, y_all)
        return best_model

    except Exception as e:
        print(f"Error training best model: {e}")

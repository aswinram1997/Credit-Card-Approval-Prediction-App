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


def perform_merge(df1, df2):
    """
    Perform inner join on ID column between df1 and df2.
    
    Args:
        df1 (pandas.DataFrame): the first dataframe to be merged
        df2 (pandas.DataFrame): the second dataframe to be merged
    
    Returns:
        pandas.DataFrame: the merged dataframe
    """
    try:    
        df = pd.merge(df1, df2, on='ID', how='outer')
        
        print("Merging dataframes successful")
        
    except Exception as e:
        print("Error during merge operation:", str(e))
        return None
    
    return df




def annotated_data(df):
    """
    Clean the input dataframe by:
    - mapping the "STATUS" column to the 2 categories
    
    Args:
        df (pandas.DataFrame): the input dataframe
    
    Returns:
        pandas.DataFrame or None: the cleaned dataframe, or None if an error occurs
    """
    try:
        # Define a function to map the status values to the 3 categories
        def map_status(STATUS):
            if STATUS in ['0', '1', '2', '3', '4', '5']:
                return 0 # Deny
            elif STATUS in ['C', 'X']:
                return 1 # Approve
            else:
                return np.nan

        # Replace the status column with the mapped values
        df["STATUS"] = df["STATUS"].apply(map_status)

        print("Annotating dataframe successful")
        
    except Exception as e:
        print("Error during annotating data:", str(e))
        return None
    
    return df




def recognize_features(df):
    """
    Select features for the input dataframe by:
    - identifying the target feature
    - identifying the numeric, categorical, continuous numeric, binary, ordinal, nominal, and high cardinality features
    
    Args:
        df (pandas.DataFrame): the input dataframe
    
    Returns:
        - target_feature (str): the target feature
        - all_features (list of str): all features
        - numeric_features (list of str): numeric features
        - categorical_features (list of str): categorical features
        - continuous_numeric_features (list of str): continuous numeric features
        - binary_features (list of str): binary features
        - ordinal_features (list of str): ordinal features
        - nominal_features (list of str): nominal features
        - high_cardinality_features (list of str): high cardinality features
    """
    try:
        # -----TARGET SELECTION-----
        
        # Output Feature
        target_feature = 'STATUS'


        # -----INPUT FEATURE RECOGNITION-----
        
        # -----all features-----
        all_features = df.columns.to_list()


        # -----numeric features-----
        numeric_features = [feature for feature in df.columns if df[feature].dtype != 'object' and df[feature].dtype !='datetime64[ns]']


        # -----categorical features-----
        categorical_features = [feature for feature in df.columns if df[feature].dtype == 'object']


        # -----contionus numieric features-----
        continuous_numeric_features = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']


        # -----binary features-----
        binary_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 
                           'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
        df[binary_features] = df[binary_features].astype('object')


        # -----ordinal features-----
        ordinal_features = []


        # -----nominal features-----
        nominal_features = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']


        # -----high cardinality features-----
        
        
        # Set the threshold for high cardinality
        threshold = 7

        # Calculate the number of unique values in each column
        cardinality = df[categorical_features].nunique()

        # Select the columns where the number of unique values is greater than the threshold
        high_cardinality_features = cardinality[cardinality > threshold].index.tolist()
        
        
        print("Feature recognition successful")


    except Exception as e:
        print("Error during feature recognition:", str(e))
        return None
    
    
    return (target_feature, all_features, numeric_features, categorical_features,
            continuous_numeric_features, binary_features, ordinal_features, nominal_features,
            high_cardinality_features)



def clean_data(df):
    """
    Function to clean data for credit card approval prediction model.

    Parameters:
    -----------
    df : pandas DataFrame
        The input data to clean.

    Returns:
    --------
    pandas DataFrame
        The cleaned data.
    """
    try:
        # Remove rows with missing values for the specified columns
        df = df.dropna(subset=['STATUS', 'CODE_GENDER'])

        # Remove ID as it is not useful for credit card approval prediction for new customers
        df = df.drop(columns=['ID'])

        # Remove Occupation Type column
        df = df.drop(columns=['OCCUPATION_TYPE'])
        
        print("Data cleaning successful")
    
    except KeyError as e:
        print(f"Error: {e} column not found.")
        return None
    
    except Exception as e:
        print("Error:", str(e))
        return None
    
    return df

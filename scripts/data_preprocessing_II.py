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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from category_encoders import BinaryEncoder

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


def count_encoder(X_train, X_val, X_test, high_cardinality_features):
    """
    Encodes high cardinality features using CountEncoder.

    Args:
    X_train (pd.DataFrame): Training dataset.
    X_val (pd.DataFrame): Validation dataset.
    X_test (pd.DataFrame): Test dataset.
    high_cardinality_features (list): List of high cardinality features.

    Returns:
    Tuple of encoded training, validation, and test datasets.
    """

    try:
        # Create an instance of CountEncoder 
        c_encoder = ce.CountEncoder()

        # get updated high cardinality feature names
        high_cardinality_features = np.intersect1d(high_cardinality_features, X_train.columns)

        # check if high_cardinality_features are available then apply count encoder to X_train, X_val, and X_test
        if high_cardinality_features.size == 0:
            print('Note: High cardinality features Unavailable!')
            X_train_c_encoded, X_val_c_encoded, X_test_c_encoded = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            # fit_transform object on train set and transform on test set
            X_train_c_encoded, X_val_c_encoded, X_test_c_encoded = c_encoder.fit_transform(X_train[high_cardinality_features]), c_encoder.transform(X_val[high_cardinality_features]), c_encoder.transform(X_test[high_cardinality_features])
        
        print("Count encoding successful")
        return X_train_c_encoded, X_val_c_encoded, X_test_c_encoded, c_encoder, high_cardinality_features

    except Exception as e:
        print(f"Error in encoding high cardinality features: {e}")
        return None
    

    

def binary_encoder(X_train, X_val, X_test, binary_features):
    """
    Encode binary features using BinaryEncoder.

    Args:
        X_train (pd.DataFrame): Training dataset.
        X_val (pd.DataFrame): Validation dataset.
        X_test (pd.DataFrame): Test dataset.
        binary_features (list): List of binary features to encode.

    Returns:
        Tuple of encoded training, validation, and test datasets.
    """
    try:
        # Create an instance of BinaryEncoder
        b_encoder = BinaryEncoder()

        # get updated binary feature names
        binary_features = np.intersect1d(binary_features, X_train.columns)

        # check if binary_features are available then apply binary encoder to X_train, X_val, and X_test
        if binary_features.size == 0:
            print('Note: Binary features Unavailable!')
            X_train_b_encoded, X_val_b_encoded, X_test_b_encoded = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            # fit_transform object on train set and transform on test set
            X_train_b_encoded, X_val_b_encoded, X_test_b_encoded = b_encoder.fit_transform(X_train[binary_features]), b_encoder.transform(X_val[binary_features]), b_encoder.transform(X_test[binary_features])
        
        print("Binary encoding successful")
        return X_train_b_encoded, X_val_b_encoded, X_test_b_encoded, b_encoder, binary_features

    except Exception as e:
        print(f"Error in encoding binary features: {e}")
        return None, None, None
    
    
    
    
def one_hot_encoder(X_train, X_val, X_test, nominal_features):
    """
    One-hot encode nominal features using a specified encoder object, and transform the train, validation, and test sets.
    
    Args:
    encoder: A OneHotEncoder object to use for encoding.
    nominal_features: A list of nominal feature names to encode.
    X_train: Training dataset.
    X_val: Test dataset.
    X_test: A pandas DataFrame containing the test set.
    
    Returns:
    A tuple of pandas DataFrames containing the one-hot encoded versions of the train, validation, and test sets.
    """
    try:
        # Create an instance of OneHotEncoder
        encoder = OneHotEncoder(handle_unknown = 'ignore')

        # get updated nominal feature names
        nominal_features = np.intersect1d(nominal_features, X_train.columns)

        # check if nominal_features are available then apply one hot encoder to X_train, X_val, and X_test
        if nominal_features.size == 0:
            print('Note: Nominal features Unavailable!')
            X_train_o_encoded, X_val_o_encoded, X_test_o_encoded = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            # fit_transform object on train set and transform on test set
            X_train_o_encoded, X_val_o_encoded, X_test_o_encoded = encoder.fit_transform(X_train[nominal_features]), encoder.transform(X_val[nominal_features]), encoder.transform(X_test[nominal_features])

            # Get the names of the nominal columns
            column_names = encoder.get_feature_names_out(nominal_features)

            # convert X_train_encoded, X_val_encoded, X_test_encoded to dense numpy array
            X_train_o_encoded, X_val_o_encoded, X_test_o_encoded = X_train_o_encoded.toarray(), X_val_o_encoded.toarray(), X_test_o_encoded.toarray()

            # convert X_train_encoded, X_val_encoded, X_test_encoded dense numpy array to DataFrame
            X_train_o_encoded, X_val_o_encoded, X_test_o_encoded = pd.DataFrame(X_train_o_encoded, columns=column_names), pd.DataFrame(X_val_o_encoded, columns=column_names), pd.DataFrame(X_test_o_encoded, columns=column_names)
        
        print("One-hot encoding successfull")
        return X_train_o_encoded, X_val_o_encoded, X_test_o_encoded, encoder, nominal_features
    
    except Exception as e:
        print(f"Error in encoding nominal features: {e}")
        
        
             
        
def combine_encoded_dataframes(X_train_c_encoded, X_train_b_encoded, X_train_o_encoded, X_val_c_encoded, X_val_b_encoded, X_val_o_encoded, X_test_c_encoded, X_test_b_encoded, X_test_o_encoded):
    """
    Combine the encoded DataFrames using pd.concat and return them.

    Parameters:
    -----------
    X_train_c_encoded: pandas.DataFrame
        Encoded DataFrame for high cardinality features of train set
    X_train_b_encoded: pandas.DataFrame
        Encoded DataFrame for binary features of train set
    X_train_o_encoded: pandas.DataFrame
        Encoded DataFrame for nominal features of train set
    X_val_c_encoded: pandas.DataFrame
        Encoded DataFrame for high cardinality features of validation set
    X_val_b_encoded: pandas.DataFrame
        Encoded DataFrame for binary features of validation set
    X_val_o_encoded: pandas.DataFrame
        Encoded DataFrame for nominal features of validation set
    X_test_c_encoded: pandas.DataFrame
        Encoded DataFrame for high cardinality features of test set
    X_test_b_encoded: pandas.DataFrame
        Encoded DataFrame for binary features of test set
    X_test_o_encoded: pandas.DataFrame
        Encoded DataFrame for nominal features of test set

    Returns:
    --------
    tuple of pandas.DataFrame
        A tuple containing combined encoded DataFrames for train, validation and test sets
    """
    
    try:
        # Reset the index of each DataFrame
        X_train_c_encoded, X_train_b_encoded, X_train_o_encoded = X_train_c_encoded.reset_index(drop=True), X_train_b_encoded.reset_index(drop=True), X_train_o_encoded.reset_index(drop=True)
        X_val_c_encoded, X_val_b_encoded, X_val_o_encoded = X_val_c_encoded.reset_index(drop=True), X_val_b_encoded.reset_index(drop=True), X_val_o_encoded.reset_index(drop=True)
        X_test_c_encoded, X_test_b_encoded, X_test_o_encoded = X_test_c_encoded.reset_index(drop=True), X_test_b_encoded.reset_index(drop=True), X_test_o_encoded.reset_index(drop=True)

        # Combine the encoded DataFrames using pd.concat
        X_train_encoded = pd.concat([X_train_c_encoded, X_train_b_encoded, X_train_o_encoded], axis=1)
        X_val_encoded = pd.concat([X_val_c_encoded, X_val_b_encoded, X_val_o_encoded], axis=1)
        X_test_encoded = pd.concat([X_test_c_encoded, X_test_b_encoded, X_test_o_encoded], axis=1)
        
        print("Combined encoded dataframes successfully")
        return X_train_encoded, X_val_encoded, X_test_encoded
    
    except Exception as e:
        print(f"Error combining encoded dataframes: {e}")
        
        
        
    
def scale_data(X_train, X_val, X_test, X_train_encoded, X_val_encoded, X_test_encoded, high_cardinality_features, binary_features, nominal_features):
    """
    Scales the data using StandardScaler.

    Parameters:
    X_train (pandas.DataFrame): Training data.
    X_val (pandas.DataFrame): Validation data.
    X_test (pandas.DataFrame): Testing data.

    Returns:
    X_train_scaled (pandas.DataFrame): Scaled training data.
    X_val_scaled (pandas.DataFrame): Scaled validation data.
    X_test_scaled (pandas.DataFrame): Scaled testing data.
    """

    try:
       # Create an instance of StandardScaler
        scaler = StandardScaler(with_mean=False)

        # Convert column names to set
        all_features = set(X_train.columns)

        # Create sets of high cardinality, binary, and nominal features
        hc_features = set(high_cardinality_features)
        binary_features = set(binary_features)
        nominal_features = set(nominal_features)

        # Find the set of features that are not in any of the three sets
        numeric_features = all_features - hc_features - binary_features - nominal_features

        # convert numeric_features to list
        numeric_features = np.array(list(numeric_features))

        # check if nominal_features are available then apply scaler to X_train, X_val, and X_test
        if numeric_features.size == 0:
            print('Numeric features Unavailable!')
            X_train_scaled, X_val_scaled, X_test_scaled = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            # fit_transform object on train set and transform on test set
            X_train_scaled, X_val_scaled, X_test_scaled = scaler.fit_transform(X_train[numeric_features]), scaler.transform(X_val[numeric_features]), scaler.transform(X_test[numeric_features])

            # convert X_train_scaled, X_val_scaled, X_test_scaled numpy array to DataFrame
            X_train_scaled, X_val_scaled, X_test_scaled = pd.DataFrame(X_train_scaled, columns=numeric_features), pd.DataFrame(X_val_scaled, columns=numeric_features), pd.DataFrame(X_test_scaled, columns=numeric_features)
    
        print("Data has been scaled successfully")
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler, numeric_features

    except Exception as e:
        print(f"An error occurred while scaling the data: {e}")
        return None, None, None
    
    

def update_scaled_data(X_train_encoded, X_train_scaled, X_val_encoded, X_val_scaled, X_test_encoded, X_test_scaled):
    """
    Updates the scaled dataframes with the encoded dataframes' features by concatenating matching columns.

    Parameters:
    -----------
    X_train_scaled (pandas.DataFrame): Scaled training data updated.
    X_val_scaled (pandas.DataFrame): Scaled validation data updated.
    X_test_scaled (pandas.DataFrame): Scaled testing data updated.

    Returns:
    --------
    Tuple of pandas DataFrames (X_train_scaled, X_val_scaled, X_test_scaled)
    """
    try:
        # update the scaled dataframe by concatenating with encoded dfs
        X_train_scaled = pd.concat([X_train_encoded, X_train_scaled], axis=1)
        X_val_scaled = pd.concat([X_val_encoded, X_val_scaled], axis=1)
        X_test_scaled = pd.concat([X_test_encoded, X_test_scaled], axis=1)

        print("Updated scaled dataframe successfully")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    except Exception as e:
        print("Error updating scaled dataframe:", e)
        return None, None, None

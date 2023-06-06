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

# import functions from scripts
from data_collection import load_data
from data_preprocessing_I import perform_merge, annotated_data, recognize_features, clean_data
from data_splitting import split_data
from data_preprocessing_II import count_encoder, binary_encoder, one_hot_encoder 
from data_preprocessing_II import combine_encoded_dataframes, scale_data, update_scaled_data
from model_training import train_best_model
from model_save import save_model


# CALL FUNCTIONS SEQUENTIALLY

# load data
df1, df2 = load_data()

# merge datasets
df = perform_merge(df1, df2)

# annotate dataset
df = annotated_data(df)

# feature recognition
target_feature, all_features, numeric_features, categorical_features, continuous_numeric_features, binary_features, ordinal_features, nominal_features, high_cardinality_features = recognize_features(df)

# data cleaning
df = clean_data(df)

# train, val, test split & feature selection
X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target_feature)

# count encoding
X_train_c_encoded, X_val_c_encoded, X_test_c_encoded, c_encoder, high_cardinality_features = count_encoder(X_train, X_val, X_test, high_cardinality_features)

# binary encoding
X_train_b_encoded, X_val_b_encoded, X_test_b_encoded, b_encoder, binary_features = binary_encoder(X_train, X_val, X_test, binary_features)

# one hot encoding
X_train_o_encoded, X_val_o_encoded, X_test_o_encoded, encoder, nominal_features = one_hot_encoder(X_train, X_val, X_test, nominal_features)

# combining encoded df's
X_train_encoded, X_val_encoded, X_test_encoded = combine_encoded_dataframes(X_train_c_encoded, X_train_b_encoded, X_train_o_encoded, X_val_c_encoded, X_val_b_encoded, X_val_o_encoded, X_test_c_encoded, X_test_b_encoded, X_test_o_encoded)

# feature scaling
X_train_scaled, X_val_scaled, X_test_scaled, scaler, numeric_features = scale_data(X_train, X_val, X_test, X_train_encoded, X_val_encoded, X_test_encoded, high_cardinality_features, binary_features, nominal_features)

# concatenate scaled data with encoded data
X_train_scaled, X_val_scaled, X_test_scaled = update_scaled_data(X_train_encoded, X_train_scaled, X_val_encoded, X_val_scaled, X_test_encoded, X_test_scaled)

# train best_model
best_model = train_best_model(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)

# save model
save_model(c_encoder, high_cardinality_features, b_encoder, binary_features, encoder, nominal_features, scaler, numeric_features, best_model)






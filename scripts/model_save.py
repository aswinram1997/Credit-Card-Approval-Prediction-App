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

# save model
import joblib



def save_model(c_encoder, high_cardinality_features, b_encoder, binary_features, encoder, nominal_features, scaler, numeric_features, best_model):
    """
    Save the preprocessing objects and model to a file
    """
    # Save the preprocessing objects
    try:
        joblib.dump({
            'count_encoder': c_encoder,
            'c_encoder_cols': high_cardinality_features,
            'binary_encoder': b_encoder,
            'b_encoder_cols': binary_features, 
            'one_hot_encoder': encoder,
            'encoder_cols': nominal_features,
            'scaler': scaler,
            'numeric_features': numeric_features,
        }, "C:/Users/aswinram/Aswin's Data Science Portfolio/Credit Card Approval Prediction/models/preprocessing_steps.joblib")
        
        print("Preprocessing objects saved successfully!")
    except Exception as e:
        print("Error occurred while saving preprocessing objects:", str(e))

        
    # Save the model
    try:
        joblib.dump(best_model, "C:/Users/aswinram/Aswin's Data Science Portfolio/Credit Card Approval Prediction/models/cca_model.joblib")
        
        print("Model saved successfully!")
    except Exception as e:
        print("Error occurred while saving the model:", str(e))

        
        

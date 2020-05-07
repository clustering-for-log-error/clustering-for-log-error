import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

def train_test(df):
    '''
    Returns X and y for train, validate and test datasets
    '''
    # don't blow away our original data
    df = df.copy()

    # validate data split
    train, test = sklearn.model_selection.train_test_split(df, train_size=.80, random_state=123)

    return train, test

# standard_scaler()
def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled, test_scaled = transform_scaler(train, test, scaler)

    return train_scaled, test_scaled, scaler

# scale_inverse()
def scale_inverse(train_scaled, test_scaled, scaler):
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train_scaled.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test_scaled.index.values])
 
    return train_unscaled, test_unscaled

def min_max_scaler(train, test):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled
import pandas as pd
import numpy as np
import acquire
from sklearn.preprocessing import MinMaxScaler


def handle_nulls(df):
    '''
    Transforms data brought in from SQL to handle nulls 
    Add calculated fields
    '''
    # Set nulls to be the median

    median = df.calculatedfinishedsquarefeet.median()
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(median)

    median = df.lotsizesquarefeet.median()
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(median)

    median = df.taxvaluedollarcnt.median()
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(median)

    median = df.taxamount.median()
    df.taxamount = df.taxamount.fillna(median)

    median = df.structuretaxvaluedollarcnt.median()
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(median)

    # Use the mode for nulls

    mode = df.yearbuilt.mode()
    df.yearbuilt = df.yearbuilt.fillna(mode)

    mode = df.landtaxvaluedollarcnt.mode()
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(mode)

    # drop all remaining rows with null values
    df = df.dropna()
    return df


def prepare_zillow(df):
    '''
    Takes in a df and returns the same df with: 
    Boolean columns indicating which county each observation is in,
    Column indicating age of home
    '''

    # create df with counties as booleans
    county_df = pd.get_dummies(df.fips)
    county_df.columns = ['Los_Angeles', 'Orange', 'Ventura']
    df = pd.concat([df, county_df], axis = 1)


    # calculate age of home
    df['age'] = 2017 - df.yearbuilt
    df.drop(columns=['yearbuilt'], inplace=True)
    return df

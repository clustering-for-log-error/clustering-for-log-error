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

    median = df.finishedsquarefeet12.median()
    df.finishedsquarefeet12 = df.finishedsquarefeet12.fillna(median)

    median = df.calculatedfinishedsquarefeet.median()
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(median)

    median = df.fullbathcnt.median()
    df.fullbathcnt = df.fullbathcnt.fillna(median)

    median = df.lotsizesquarefeet.median()
    df.lotsizesquarefeet = df.lotsizesquarefeet.fillna(median)

    median = df.taxvaluedollarcnt.median()
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(median)

    median = df.taxamount.median()
    df.taxamount = df.taxamount.fillna(median)

    median = df.taxamount.median()
    df.taxamount = df.taxamount.fillna(median)

    median = df.structuretaxvaluedollarcnt.median()
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(median)

    median = df.structuretaxvaluedollarcnt.median()
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(median)

    df.buildingqualitytypeid = df.buildingqualitytypeid.fillna(df.buildingqualitytypeid.median())


    # Use the mode for nulls
    mode = df.regionidcity.mode()
    df.regionidcity = df.regionidcity.fillna(mode)

    mode = df.regionidzip.mode()
    df.regionidzip = df.regionidzip.fillna(mode)

    mode = df.yearbuilt.mode()
    df.yearbuilt = df.yearbuilt.fillna(mode)

    mode = df.landtaxvaluedollarcnt.mode()
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(mode)

    # Imputing with most common observation
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna("None")
    df.unitcnt = df.unitcnt.fillna(1.0)
    df.unitcnt = df.unitcnt.fillna(1.0)


    # drop regionidneighborhood and transactiondate
    df = df.drop(columns=['transactiondate'])

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
    df = df.drop(columns = ['fips'])

    # calculate age of home
    df['age'] = 2017 - df.yearbuilt
    return df

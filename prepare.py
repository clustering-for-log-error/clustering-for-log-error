import pandas as pd 
import numpy as numpy

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

def better_names(df):
    """
    Rename columns to easier to understand names. Then reorder columns based on relevancy to others.
    """
    # rename
    df = df.rename(columns={"calculatedfinishedsquarefeet":"square_footage", "structuretaxvaluedollarcnt":"house_value", "landtaxvaluedollarcnt":"land_value", "taxvaluedollarcnt":"full_value", "lotsizesquarefeet":"lot_size"})
     
    # reorder
    df = df[["longitude", "latitude", "age", "bedroomcnt", "bathroomcnt", "square_footage", "lot_size", "house_value", "land_value", "full_value", "tax_rate", "Los_Angeles", "Orange", "Ventura", "logerror", "bed_bath_ratio"]]
    return df

def tax_rate(df):
    """
    Create a new taxrate column and drop the taxamount column, as it no longer adds value.
    """
    df["tax_rate"] = df.taxamount / df.taxvaluedollarcnt
    df.drop(columns=["taxamount"], inplace=True)
    return df

def transaction_month(df):
    """
    Replace the transactiondate with the month of the transaction.
    """
    df["month"] = pd.DatetimeIndex(df.transactiondate).month
    df.drop("transactiondate",axis=1, inplace=True)
    return df

def bed_bath_ratio(df):
    df['bed_bath_ratio'] = df.bedroomcnt/df.bathroomcnt
    return df


# remove outlier
col_out = ["bathroomcnt", "bedroomcnt", "tax_rate", "calculatedfinishedsquarefeet", "lotsizesquarefeet", "structuretaxvaluedollarcnt", "taxvaluedollarcnt", "landtaxvaluedollarcnt"]

def remove_outliers_iqr(df, col_out):
    for col in enumerate(col_out):
        col = str(col[1])
        
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        ub = q3 + 3 * iqr
        lb = q1 - 3 * iqr

        df = df[df[col] <= ub]
        df = df[df[col] >= lb]
    return df
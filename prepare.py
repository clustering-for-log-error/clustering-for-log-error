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
    df = df[["longitude", "latitude", "age", "month", "bedroomcnt", "bathroomcnt", "square_footage", "lot_size", "house_value", "land_value", "full_value", "tax_rate", "roomcnt", "Los_Angeles", "Orange", "Ventura", "logerror"]]
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
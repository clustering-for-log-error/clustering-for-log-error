import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def kmeans_model(df,n):
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    return kmeans.labels_
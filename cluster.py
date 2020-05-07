import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from env import host, user, password
import acquire
import wrangle
import prepare

import warnings
warnings.filterwarnings("ignore")



def elbow_method(list_of_3_variables):
    cluster_vars = list_of_3_variables

    ks = range(2,20)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(cluster_vars)

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.show()


def get_clusters_and_counts(k, var_list, cluster_col_name, train_scaled, test_scaled):
    """
    be sure your scaled X dataframes are named: train_scaled and test_scaled
    takes in k, list of vars to cluster on, and the new cluster id column name
    and returns the kmeans fitted object, dataframe of the train clusters with their observations, 
    test clusters and their observations, and a df of the number of observations per cluster on train. 
    """
    
    # find clusters
    kmeans = KMeans(n_clusters=k, random_state = 447)
    train_cluster_array = kmeans.fit_predict(train_scaled[var_list])
    test_cluster_array = kmeans.predict(test_scaled[var_list])
    
    # create df of cluster id with each observation
    train_clusters = pd.DataFrame(train_cluster_array, columns = [cluster_col_name], index = train_scaled.index)
    test_clusters = pd.DataFrame(test_cluster_array, columns = [cluster_col_name], index = test_scaled.index)
    
    # output number of observations in each cluster
    cluster_counts = train_clusters[cluster_col_name].value_counts()
    
    return kmeans, train_clusters, test_clusters, cluster_counts


def append_clusters_and_centroids(X_train, train_scaled, train_clusters, 
                                X_test, test_scaled, test_clusters, 
                                cluster_col_name, centroid_col_names_list, kmeans):

    """
    be sure your dataframes are named: X_train, X_test, train_scaled, test_scaled (dataframes of X scaled)
    takes in list of vars to cluster on, 
    and the new cluster id column name
    and returns the kmeans fitted object, dataframe of the train clusters with their observations, 
    test clusters and their observations, and a df of the number of observations per cluster on train. 
    """
    
    # join the cluster ID's with the X dataframes (the scaled and unscaled, train and test
    
    X_train = pd.concat([X_train, train_clusters], axis = 1)
    train_scaled = pd.concat([train_scaled, train_clusters], axis = 1)

    X_test = pd.concat([X_test, test_clusters], axis = 1)
    test_scaled = pd.concat([test_scaled, test_clusters], axis = 1)
      
    # get the centroids for  distinct cluster...
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=centroid_col_names_list).reset_index()
    centroids.rename(columns = {'index': cluster_col_name}, inplace = True)
    
    # merge the centroids with the X dataframes (both the scaled and unscaled)
    X_train = X_train.merge(centroids, how='left', on=cluster_col_name).set_index(X_train.index)
    train_scaled = train_scaled.merge(centroids, how = 'left', on = cluster_col_name).set_index(train_scaled.index)
    
    X_test = X_test.merge(centroids, how = 'left', on = cluster_col_name).set_index(X_test.index)
    test_scaled = test_scaled.merge(centroids, how = 'left', on = cluster_col_name).set_index(test_scaled.index)
    
    return X_train, train_scaled, X_test, test_scaled, centroids


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
from sklearn.neighbors import KDTree

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

def elbow(df, points=10):
    ks = range(1,points+1)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=123)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    print(pd.DataFrame(dict(k=ks, sse=sse)))
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k')
    plt.show()

def k_cluster_all(df, x, n):
    """
    Takes a dataframe and a single feature, and performs a 2d kmeans clustering on that feature against all other features in the dataframe. Also, specify the number of clusters to explore.
    """  
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    df["cluster"] = kmeans.predict(df)
    df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

    for col in df.columns:
        if col != x and col != "cluster":
            sns.relplot(data=df, x=x, y=col, hue='cluster', alpha=.3)
            plt.show()
    df.drop(columns="cluster", inplace=True)

def k_cluster_2d(df, x, y, n_max, n_min=2):
    """
    Plots a 2D cluster map of an inputted x and y, starting at 2 clusters, up to inputted max cluster amount
    Import whole dataframe, select the x and y values to cluster.
    """
    for n in range(n_min,n_max+1):
        kmeans = KMeans(n_clusters=n, random_state=123)
        kmeans.fit(df)
        df["cluster"] = kmeans.predict(df)
        df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

        sns.relplot(data=df, x=x, y=y, hue='cluster', alpha=.5)
        plt.title(f'{n} Clusters')
        df.drop(columns="cluster", inplace=True)

def k_cluster_3d(df, x, y, z, n):
    """
    Displays 3d plot of clusters.
    """
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    cluster_label = kmeans.labels_

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
  
    scatter = ax.scatter(df[x], df[y], df[z], c=cluster_label,alpha=.5)
    legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Clusters")
    
    ax.add_artist(legend)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    ax.set(xlabel=x, ylabel=y, zlabel=z)
    ax.xaxis.labelpad=-5
    ax.yaxis.labelpad=-5
    ax.zaxis.labelpad=-5
    plt.show()

def get_pde(df,bw):
    """
    Assits in plotting a parcel density estimation 2d scatter plot. Use the longitude and latitude as x, y coordinates and color these points by their density.
    """
    x = df['longitude'].values
    y = df['latitude'].values
    xy = np.vstack([x,y])
    X = np.transpose(xy)
    tree = KDTree(X,leaf_size = 20 )     
    parcelDensity = tree.kernel_density(X, h=bw,kernel='gaussian',rtol=0.00001)
    return parcelDensity

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

def test_sig(cluster_column,df):
    """
    Takes a column of clusters and performs a t-test with the logerrors of cluster (subset) against the population logerror.
    """  
    ttest_list = []
    pval_list = []
    stat_sig = []

    for cluster in cluster_column.unique():
        ttest, pval = stats.ttest_1samp(df["logerror"][cluster_column == cluster],df["logerror"].mean(),axis=0,nan_policy="propagate")
        ttest_list.append(ttest)
        pval_list.append(pval)
        sig = pval < 0.05
        stat_sig.append(sig)
        
    stats_cluster_column = pd.DataFrame({"ttest":ttest_list,"pval":pval_list,"stat_sig":stat_sig})
    return stats_cluster_column


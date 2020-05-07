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
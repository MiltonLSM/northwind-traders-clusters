"""
Utility functions for clustering analysis
"""

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

def tune_dbscan(data, max_eps):
    """
    Evaluate multiple DBSCAN hyperparameter combinations and return the results.

    Tests a range of `eps` and `min_samples` values, fits a DBSCAN model
    for each combination, and records the resulting number of clusters,
    number of noise points, and silhouette score.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The dataset to cluster. It should contain only numerical features,
        and ideally be scaled before applying DBSCAN.

    max_eps : float
        The maximum value of `eps` to test. Values from 0.1 up to
        (but not including) `max_eps` are evaluated in increments of 0.1.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing one row for each combination of `eps`
        and `min_samples`, with the following columns:

        - ``Eps`` : float
            The neighborhood radius used by DBSCAN.
        - ``Min Samples`` : int
            The minimum number of samples required to form a dense region.
        - ``Number of Clusters`` : int
            The number of clusters found, excluding noise points.
        - ``Number of Noise Points`` : int
            The number of points labeled as noise (-1).
        - ``Silhouette Score`` : float or None
            The silhouette score of the clustering. Returns ``None`` if
            fewer than two clusters are found.
    """

    results = []

    # define a range of eps and min_samples values to loop through
    eps_values = np.arange(.1, max_eps, .1)
    min_samples_values = np.arange(2, 10, 1)

    # loop through the combinations of eps and min_samples
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(data)
            labels = dbscan.labels_

            # count the number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # count the number of noise points (labeled as -1)
            n_noise = list(labels).count(-1)

            # calculate the silhouette score (excluding noise points)
            if n_clusters > 1:  # silhouette score requires at least 2 clusters
                silhouette = silhouette_score(data, labels, metric='euclidean', sample_size=None)
            else:
                silhouette = None

            results.append([eps, min_samples, n_clusters, n_noise, silhouette])

    # put the results in a dataframe
    dbscan_results = pd.DataFrame(results, columns=["Eps", "Min Samples", "Number of Clusters",
                                                    "Number of Noise Points", "Silhouette Score"])
    return dbscan_results



def display_results(model, data):
    """
    Display basic clustering evaluation results for a fitted model.

    Prints:
    1. The model object and its parameters.
    2. The number of samples assigned to each cluster.
    3. The silhouette score of the clustering.

    Parameters
    ----------
    model : object
        A fitted clustering model that has a `labels_` attribute,
        such as a model from scikit-learn (e.g., KMeans, DBSCAN,
        AgglomerativeClustering).

    data : array-like of shape (n_samples, n_features)
        The dataset used to fit the clustering model. This is passed
        to `silhouette_score` along with the cluster labels.

    Returns
    -------
    None
        This function only prints the clustering results and does not
        return any value.
    """
    print(model)
    print(Counter(model.labels_))
    print(silhouette_score(data, model.labels_))
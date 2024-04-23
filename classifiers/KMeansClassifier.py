"""
Using K-means as the backend, predict using the cluster centroids
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances
from utils.inference import find_majority


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.kmeans_ = KMeans(**self.kwargs).fit(self.X_)

        # compute the label for each resulting cluster
        self.cluster_indices_to_label_ = {}
        for i in range(self.kmeans_.n_clusters):
            mask = self.kmeans_.labels_ == i
            labels = self.y_[mask]
            p = find_majority(labels)
            self.cluster_indices_to_label_[i] = p
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        preds = self.kmeans_.predict(X)  # cluster ids
        return np.array(
            list(map(lambda p: self.cluster_indices_to_label_[p], preds))
        )

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **parameters):
        self.kwargs = parameters
        return self
        # for parameter, value in parameters.items():
        #     setattr(self, parameter, value)
        # return self

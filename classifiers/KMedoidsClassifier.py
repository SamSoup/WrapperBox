"""
Using K-Medoids as the backend, predict using the cluster centroids
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn_extra.cluster import KMedoids

class KMedoidsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.kmedoids_ = KMedoids(**self.kwargs).fit(self.X_)
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        preds = self.kmedoids_.predict(X) # cluster ids

        # now we need to figure out the actual labels from cluster ids
        find_row = lambda row: np.where(np.all(row==self.X_,axis=1))[0][0]
        centroid_indices = np.apply_along_axis(
            find_row, 1, self.kmedoids_.cluster_centers_
        )
        centroid_labels = self.y_[centroid_indices]
        cluster_to_label = {
            i: centroid_labels[i] for i in range(self.kmedoids_.n_clusters)
        }

        return np.array(list(map(lambda p: cluster_to_label[p], preds)))

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **parameters):
        self.kwargs = parameters
        return self
        # for parameter, value in parameters.items():
        #     setattr(self, parameter, value)
        # return self

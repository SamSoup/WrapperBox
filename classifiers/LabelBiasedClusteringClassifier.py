"""
A different way to construct the decision tree that is example-based might 
be based on supervised hierarchical clustering as follows. At the root, 
compute the centroid of positive vs. negative examples and bifurcate data 
based on proximity to these centroids.  

I could imagine either splitting the data evenly (equal group sizes, 
but some examples might be pushed into the group of the farther centroid) 
or unequally (each example clusters with whichever centroid is closer, 
leading to imbalanced groups). At each successive node, similarly, find 
centroids of positive vs. negative groups and bifurcate further. Because 
the decision tree is constructed in this way, explaining it in this way 
would be 100% faithful, but building the decision tree this way may yield 
a less accurate prediction model than the standard way of building the 
decision tree.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from utils import find_majority, randargmax

class LabelBiasedClusteringClassifier(BaseEstimator, ClassifierMixin):
    class Node:
        def __init__(self, indices, counts):
            self.indices = indices
            self.counts = counts
            self.children = []

    def __init__(self, max_depth=3, min_samples_split=50, min_samples_leaf=10):
        """
        Record hyperparameters
        
        max_depth: how deep we want to go before stopping

        min_samples_split: The minimum number of samples required to split an 
        internal node

        min_samples_leaf: The minimum number of samples required to be at a 
        leaf node. A split point at any depth will only be considered if it 
        leaves at least min_samples_leaf training samples in each of the left 
        and right branches.
        
        This considers a split for however many labels there are in the tree 
        """
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _split_based_on_centroids(self):
        """
        Compute the centroid of each class of examples, then assign data
        points based on proximity to each centroid
        """
        def preorder_traversal(indices, depth):
            labels = self.y_[indices]
            n = self.Node(indices, Counter(labels))
            if len(indices) < self.min_samples_split or depth > self.max_depth:
                return n
            subset = self.X_[indices]
            centroids = []
            # compute the centroids
            for l in self.classes_:
                group = subset[labels == l]
                if not group.size:
                    # pure group: no further splits necessary
                    return n
                centroids.append(group.mean(axis=0))
            # compute distance to centroids
            dists = euclidean_distances(subset, np.vstack(centroids))
            # assign each sample to the closest centroid
            clusters = randargmax(dists)
            for i in range(len(self.classes_)):
                next_node_indices = indices[clusters == i]
                if len(next_node_indices) < self.min_samples_leaf:
                    return n
                n.children.append(preorder_traversal(next_node_indices, depth+1))
            return n
        return preorder_traversal(np.arange(self.X_.shape[0]), 1)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        
        self.head_ = self._split_based_on_centroids()
        # Return the classifier
        return self

    def _split_predict(self, ex):
        """
        Using the bifurcated based on groups clusters,
        predict using centroids
        """
        node = self.head_
        def preorder_traversal(node, depth):
            if len(node.children) and depth <= self.max_depth:
                centroids = []
                for child in node.children:
                    indices = child.indices
                    centroid = self.X_[indices].mean(axis=0)
                    centroids.append(centroid)
                dists = euclidean_distances(ex.reshape(1, -1), np.vstack(centroids))
                where_to = np.argmin(dists.reshape(1, -1))
                return preorder_traversal(node.children[where_to], depth+1)
            else:
                # vote using the labels
                labels = self.y_[node.indices]
                p = find_majority(labels)
                return p
        return preorder_traversal(node, 1)

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        return np.apply_along_axis(
            func1d=self._split_predict, axis=1, arr=X
        )

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

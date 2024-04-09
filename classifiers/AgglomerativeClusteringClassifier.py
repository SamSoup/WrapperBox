"""
Regarding decision tree, I had suggested possible hierarchical clustering such 
that each node split is between centroids of positive vs. negative examples, 
however I wonder if good old divisive or agglomerative hierarchical clustering 
would be fine -- at each node, is it more like this centroid or that centroid 
(regardless of label), then when you get to the leaf you find out the correct 
label.  This might be more intuitive to the user since the splits at each node 
would be entirely based on example similarity, rather than being biased by 
latent labels they can't see.
"""

import numpy as np
import itertools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from utils import find_majority

class AgglomerativeClusteringClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=1,
        n_clusters=2,
        *,
        affinity="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
    ):
        self.max_depth = max_depth
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.affinity = affinity
        self.compute_distances = compute_distances

    def _find_samples_per_node(self):
        """
        Associate actual samples with the cluster splits
        """
        node = self.head_
        def preorder_traversal(node):
            if node in self.node_to_children_:
                l = self.node_to_children_[node]['left']
                l_c = preorder_traversal(l)
                r = self.node_to_children_[node]['right']
                r_c = preorder_traversal(r)
                indices = l_c + r_c
                self.node_to_children_[node]['indices'] = indices
                self.node_to_children_[node]['counts'] = Counter(self.y_[indices])
                return self.node_to_children_[node]['indices']
            else:
                # leaf node
                return [node]
        preorder_traversal(node)
        return self.node_to_children_

    def fit(self, X, y=None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        
        self.m_ = AgglomerativeClustering(
            n_clusters = self.n_clusters,
            distance_threshold = self.distance_threshold,
            memory = self.memory,
            connectivity = self.connectivity,
            compute_full_tree = self.compute_full_tree,
            linkage = self.linkage,
            affinity = self.affinity,
            compute_distances = self.compute_distances
        ).fit(self.X_)

        ii = itertools.count(self.X_.shape[0])
        self.node_to_children_ = {
            next(ii): {'left': x[0], 'right':x[1]} for x in self.m_.children_
        }
        self.head_ = max(self.node_to_children_)
        self.node_to_children_ = self._find_samples_per_node()
        return self

    def _hierarchical_predict(self, ex):
        """
        Using the HCs based on example similarity, predict using centroids
        """
        node = self.head_
        def preorder_traversal(node, depth):
            if node in self.node_to_children_ and depth < self.max_depth:
                centroids = []
                l = self.node_to_children_[node]['left']
                r = self.node_to_children_[node]['right']
                for c in [l, r]:
                    indices = self.node_to_children_[c]['indices']
                    centroid = self.X_[indices].mean(axis=0)
                    centroids.append(centroid)
                dists = euclidean_distances(ex.reshape(1, -1), np.vstack(centroids))
                where_to = np.argmin(dists.reshape(1, -1)) # go to min distance
                if not where_to:
                    return preorder_traversal(l, depth+1)
                else:
                    return preorder_traversal(r, depth+1)
            else:
                # treat this as a leaf node
                indices = self.node_to_children_[node]['indices']
                labels = self.y_[indices]
                p = find_majority(labels)
                return p
        return preorder_traversal(node, 0)

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        return np.apply_along_axis(
            func1d=self._hierarchical_predict, axis=1, arr=X
        )

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def plot_dendrogram(self, **kwargs):
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(self.m_.children_.shape[0])
        n_samples = len(self.m_.labels_)
        for i, merge in enumerate(self.m_.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [self.m_.children_, self.m_.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        return dendrogram(linkage_matrix, **kwargs)

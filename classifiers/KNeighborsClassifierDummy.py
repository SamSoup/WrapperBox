"""
KNN is very memory intensive. This class thus serves as a dummy class with
pre-computed results, but conforms to the standard interface for 
sklearn.neighbors.KNeighborsClassifier
"""

import pickle


class KNeighborsClassifierDummy:
    def __init__(
        self, predictions_path, neigh_dists_path, neigh_inds_path, n_neighbors=5
    ):
        self._K = n_neighbors

        with open(predictions_path, "rb") as f:
            self._predictions = pickle.load(f)

        with open(neigh_dists_path, "rb") as f:
            self._neigh_dists = pickle.load(f)

        with open(neigh_inds_path, "rb") as f:
            self._neigh_inds = pickle.load(f)

    def predict(self, X):
        # Return cached predict results
        return self._predictions

    def kneighbors(self, X, n_neighbors=5, return_distance=True):
        K = self._K if n_neighbors is None else n_neighbors
        neigh_dists = self._neigh_dists[:, :K]
        neigh_inds = self._neigh_inds[:, K]

        if return_distance:
            return neigh_dists, neigh_inds
        else:
            return neigh_inds

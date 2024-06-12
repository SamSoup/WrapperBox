"""
KNN is very memory intensive. This class thus serves as a dummy class with
pre-computed results, but conforms to the standard interface for 
sklearn.neighbors.KNeighborsClassifier
"""

import pickle


class KNeighborsClassifierDummy:
    def __init__(
        self,
        predictions_path: str,
        neigh_inds_path: str,
        neigh_dists_path: str = None,
        n_neighbors: int = 5,
    ):
        self.n_neighbors = n_neighbors

        with open(predictions_path, "rb") as f:
            self._predictions = pickle.load(f)

        with open(neigh_inds_path, "rb") as f:
            self._neigh_inds = pickle.load(f)

        if neigh_dists_path is not None:
            with open(neigh_dists_path, "rb") as f:
                self._neigh_dists = pickle.load(f)
        else:
            self._neigh_dists = None

    def predict(self, X):
        # Return cached predict results
        return self._predictions

    def kneighbors(self, X, n_neighbors=5, return_distance=True):
        K = self.n_neighbors if n_neighbors is None else n_neighbors
        neigh_inds = self._neigh_inds[:, :K]

        if return_distance:
            neigh_dists = self._neigh_dists[:, :K]
            return neigh_dists, neigh_inds
        else:
            return neigh_inds

# This script finds the minimal set of neighbors for a svm
from typing import List
from .interface import FindMinimalSubset
from sklearn.neighbors import KNeighborsClassifier
from utils.inference import find_majority_batched
from tqdm import tqdm
import numpy as np


class FindMinimalSubsetKNN(FindMinimalSubset):
    def _compute_movement(
        self, labels: np.ndarray, predictions: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """
        Exclusive helper to find the minimal subset for KNN greedily by
        iteratively "removing" the closest neighbor. In effect, this shifts
        a window of `window_size` downwards, until the majority label changes

        Args:
            labels (np.ndarray): (num_test_examples, num_train_eval_examples),
            the labels of the neighbors of each test prediction, ranked in order
            from closest to farthest

            predictions (np.ndarray): (num_test_examples), prediction of the
            knn classifier for each test example

            window_size (int, optional): K. Defaults to 5.

        Returns:
            np.ndarray: (num_test_examples), the number of window shifts it
            takes for the prediction to flip; could be -1 if no shift can be
            identified greedily
        """
        # Initialize variables to keep track of window and movement
        movement = np.full(shape=labels.shape[0], fill_value=-1, dtype=int)

        # Slide the window and compute movement for each row
        for i in tqdm(
            range(1, labels.shape[1] - window_size),
            "Computing Expl Indices for KNN",
        ):
            # Compute majority of the current window for each row
            current_window = labels[:, i : i + window_size]
            majority_current = find_majority_batched(current_window)
            # Check if majority of the window has changed from the first 5 per row
            changed_majority = np.logical_not(majority_current == predictions)
            # print("Current window:", current_window)
            # print("Predictions:", predictions)
            # print("Current majorities:", majority_current)
            # print("Changed majorities:", changed_majority)
            # print("Current movement compilated:", movement)
            # print(np.logical_and(movement == -1, changed_majority))
            movement[np.logical_and(movement == -1, changed_majority)] = i

        return movement

    def find_minimal_subset(
        self,
        clf: KNeighborsClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        # knn has no learning: so use train and eval labels together
        predictions = clf.predict(test_embeddings)
        neigh_ind = clf.kneighbors(
            X=test_embeddings,
            n_neighbors=len(train_labels),
            return_distance=False,
        )
        # use the neigh_ind to retrieve the indices of the neighbors
        neigh_labels = train_labels[neigh_ind]
        # print(neigh_labels.shape)
        # print(test_embeddings.shape)
        # print(predictions.shape)
        # input()
        # the task of finding the minimal set for the nearest neighbor approach
        # is just using a sliding window to see when the majority label changes
        # from the predictions

        movement = self._compute_movement(
            labels=neigh_labels,
            predictions=predictions,
            window_size=clf.n_neighbors,
        )
        subset_indices = []
        # the train indices to remove is simply from 0:movement
        for indices, end in tqdm(zip(neigh_ind, movement)):
            subset_indices.append(indices[:end].tolist())

        return subset_indices

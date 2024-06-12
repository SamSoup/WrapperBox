# This script finds the minimal set of neighbors for a svm
from typing import Iterable, List

from sympy import substitution
from MinimalSubsetToFlipPredictions.models.interface import FindMinimalSubset
from sklearn.neighbors import KNeighborsClassifier
from utils.inference import find_majority, find_majority_batched
from collections import Counter, deque
from tqdm import tqdm
import numpy as np
import multiprocessing


class FindMinimalSubsetKNN(FindMinimalSubset):
    def _find_subset_single(
        self, prediction: int, neighbors: Iterable[int], K: int
    ) -> List[int]:
        """
        Finds the indices of neighbors such that their removal leads to a
            prediction flip for a k-nearest neighbors model.

        Args:
            prediction (int): The initial predicted label of the test input.
            neighbors (Iterable[int]): The labels of the neighbors of the
                test input, sorted by proximity.
            K (int): The number of nearest neighbors to consider in the
                sliding window.

        Returns:
            List[int]: A list of indices of neighbor labels that, when removed,
                cause the prediction to flip.
        """
        k_neighbors = deque(neighbors[:K])  # Initial K neighbors
        remaining_neighbors = deque(neighbors[K:])  # Remaining neighbors

        # Initialize vote counter for the K-window
        vote_counter = Counter(k_neighbors)

        # To keep track of the indices of neighbors to remove
        indices_to_remove = []

        # Iterate through the neighbors
        for idx, neighbor in tqdm(enumerate(neighbors)):
            if neighbor == prediction:
                # Temporarily remove the neighbor from the K-window and update the counter
                if neighbor in k_neighbors:
                    k_neighbors.remove(neighbor)
                    indices_to_remove.append(idx)
                    vote_counter[neighbor] -= 1
                    if vote_counter[neighbor] == 0:
                        del vote_counter[neighbor]

                    # Add the next neighbor from the remaining list to maintain the window size
                    if remaining_neighbors:
                        next_neighbor = remaining_neighbors.popleft()
                        k_neighbors.append(next_neighbor)
                        vote_counter[next_neighbor] += 1

                    # Check the majority label in the current K-window
                    majority_label = vote_counter.most_common(1)[0][0]

                    # Check if the majority label has changed
                    if majority_label != prediction:
                        return indices_to_remove  # Return the indices that cause the flip

        # If no flip occurred, return the indices list
        return indices_to_remove

    def _find_subset_parallel(
        self, predictions: Iterable[int], neighbors_matrix: np.ndarray, K: int
    ) -> List[List[int]]:
        """
        Finds subsets of neighbors in parallel for multiple test inputs,
        such that their removal leads to prediction flips.

        Args:
            predictions (Iterable[int]): The initial predicted labels of
                the test inputs.
            neighbors_matrix (np.ndarray): A 2D array where each row contains
                the labels of the neighbors for each test input, sorted by proximity.
            K (int): The number of nearest neighbors to consider in the sliding
                window.

        Returns:
            List[List[int]]: A list of lists, where each inner list contains
                the subset of neighbor indices that cause the prediction to
                flip for each test input.
        """
        # Use half of the available CPUs to prevent overload
        max_processes = multiprocessing.cpu_count() // 2
        chunk_size = max(1, len(predictions) // (max_processes * 2))

        with multiprocessing.Pool(
            processes=max_processes, maxtasksperchild=4
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        self._find_subset_single,
                        [
                            (prediction, neighbors, K)
                            for prediction, neighbors in zip(
                                predictions, neighbors_matrix
                            )
                        ],
                        chunksize=chunk_size,
                    ),
                    total=len(predictions),
                )
            )
        return results

    def _find_subset_brute_force(
        self, predictions: Iterable[int], neighbors_matrix: np.ndarray, K: int
    ) -> List[List[int]]:
        """
        Finds subsets of neighbors one-by-one for multiple test inputs,
        such that their removal leads to prediction flips.

        Args:
            predictions (Iterable[int]): The initial predicted labels of
                the test inputs.
            neighbors_matrix (np.ndarray): A 2D array where each row contains
                the labels of the neighbors for each test input, sorted by proximity.
            K (int): The number of nearest neighbors to consider in the sliding
                window.

        Returns:
            List[List[int]]: A list of lists, where each inner list contains
                the subset of neighbor indices that cause the prediction to
                flip for each test input.
        """
        subset_indices = []
        for prediction, neighbors in tqdm(
            zip(predictions, neighbors_matrix), total=len(predictions)
        ):
            subset_indices.append(
                self._find_subset_single(
                    prediction=prediction, neighbors=neighbors, K=K
                )
            )

        return subset_indices

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
        predictions = clf.predict(test_embeddings)
        neigh_ind = clf.kneighbors(
            X=test_embeddings,
            n_neighbors=len(train_labels),
            return_distance=False,
        )
        neigh_labels = train_labels[neigh_ind]

        # the task of finding the minimal set for the nearest neighbor approach
        # is just using a sliding window to see when the majority label changes
        # from the predictions
        # movement = self._compute_movement(
        #     labels=neigh_labels,
        #     predictions=predictions,
        #     window_size=clf.n_neighbors,
        # )
        # subset_indices = []
        # # the train indices to remove is simply from 0:movement
        # for indices, end in tqdm(zip(neigh_ind, movement)):
        #     subset_indices.append(indices[:end].tolist())

        # do parallel versus do iterative, due to memory constraints
        if predictions.size < 1000:
            print("Samll Sample Size, Finding St via Parallel Processes")
            label_indices_to_remove = self._find_subset_parallel(
                predictions=predictions,
                neighbors_matrix=neigh_labels,
                K=clf.n_neighbors,
            )
        else:
            print("Large Sample Size, Finding St via Brute Force")
            label_indices_to_remove = self._find_subset_brute_force(
                predictions=predictions,
                neighbors_matrix=neigh_labels,
                K=clf.n_neighbors,
            )

        # need to convert label indices to example indices
        subset_indices = []
        for i, l_indices in enumerate(label_indices_to_remove):
            subset_indices.append(neigh_ind[i][l_indices].tolist())

        return subset_indices

# This script finds the minimal set of examples required to flip a lmeans pred

from collections import defaultdict
import gc
import sys
from typing import Iterable, List
from sklearn.base import clone
from tqdm import tqdm
from utils.partition import partition_indices
from ExampleBasedExplanations.lmeans import LMeansExampleBasedExplanation
from MinimalSubsetToFlipPredictions.wrappers.interface import FindMinimalSubset
from classifiers.KMeansClassifier import KMeansClassifier
import numpy as np


class FindMinimalSubsetLMeans(FindMinimalSubset):
    def __init__(
        self, ITERATIVE_THRESHOLD: int = None, SPLITS: int = None
    ) -> None:
        super().__init__()
        self.SPLITS = SPLITS
        self.ITERATIVE_THRESHOLD = ITERATIVE_THRESHOLD
        ## for internal use
        self._last_seen_subset_size = None
        self._refining_last_split_chunk = False

    def find_minimal_subset_cluster_batched(
        self,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        # Find minimal subsets by iteratively removing examples from each
        # cluster; This is good when clusters themeselves are small but are
        # not easily parallelizable when clusters are few and big

        handler = LMeansExampleBasedExplanation()
        cluster_idx_to_explanation = handler._get_cluster_idx_to_explanation(
            M=None,  # want all examples
            clf=clf,
            train_embeddings=train_embeddings,
        )
        for cluster_idx in cluster_idx_to_explanation:
            print(
                f"Found {len(cluster_idx_to_explanation[cluster_idx])} "
                f" train examples for cluster {cluster_idx}"
            )
        # construct a dict mapping cluster_idx: test_example_idx
        cluster_ids_test = clf.kmeans_.predict(test_embeddings)
        print(f"First 5 Cluster IDs for test examples: {cluster_ids_test[:5]}")
        predictions = clf.predict(test_embeddings)
        print(f"First 5 Original predictions: {predictions[:5]}")
        cluster_idx_to_test_idx = defaultdict(list)
        for i, value in enumerate(cluster_ids_test):
            cluster_idx_to_test_idx[value].append(i)
        for cluster_idx, indices in cluster_idx_to_test_idx.items():
            print(
                f"Found {len(indices)} test examples for cluster {cluster_idx}"
            )
        # procedure: iteratively remove cluster examples and recluster,
        # then, check if predictions are flipped for the example
        # in that cluster only (because other examples would have a
        # different removal order)
        subset_indices_per_example = [
            [] for _ in range(test_embeddings.shape[0])
        ]
        num_classes = len(np.unique(train_labels))
        for cluster_idx, indices in cluster_idx_to_explanation.items():
            # convert numpy array to int for storage
            indices = indices.astype(int).tolist()
            print(f"Start example removal for cluster {cluster_idx}")
            removed_indices = []
            test_idx_to_check = cluster_idx_to_test_idx[cluster_idx]
            train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
            cluster_idx_pred = clf.cluster_indices_to_label_[cluster_idx]
            print(
                f"For cluster {cluster_idx}, "
                f"LMeans always predicted label {cluster_idx_pred}"
            )
            for train_idx in tqdm(indices):
                print(f"Removing training example {train_idx}")
                # accum removed examples + get remaining items to check
                removed_indices.append(train_idx)
                test_embeddings_subset = test_embeddings[test_idx_to_check]

                # Exclude selected examples from the training set
                train_mask[train_idx] = False
                X_train = train_embeddings[train_mask]
                y_train = train_labels[train_mask]
                print(f"Train subset shapes: {X_train.shape}, {y_train.shape}")
                if len(np.unique(y_train)) == num_classes:
                    new_clf = clone(clf)
                    new_clf.fit(X_train, y_train)
                    new_predictions = new_clf.predict(test_embeddings_subset)
                    new_old_eq = np.all(new_predictions == cluster_idx_pred)
                    print(
                        f"New cluster indices to label {new_clf.cluster_indices_to_label_}"
                    )
                    print(f"New centroids: {new_clf.kmeans_.cluster_centers_}")
                    print(
                        f"{new_predictions}, is all equal to {cluster_idx_pred}: {new_old_eq} "
                    )
                else:
                    print("Not enough data points for all unique classes")
                    new_predictions = [
                        -1 for _ in range(len(test_idx_to_check))
                    ]

                # check if a prediction flip has been reached, for
                # all of the examples
                remaining_test_idx_to_check = []
                for test_idx, new_pred in zip(
                    test_idx_to_check, new_predictions
                ):
                    if cluster_idx_pred != new_pred:
                        print(
                            f"\nIterative removal of train examples "
                            f"{removed_indices} lead to flipped pred for "
                            f"test example {test_idx}.\n"
                        )
                        subset_indices_per_example[test_idx].extend(
                            removed_indices
                        )
                    else:
                        remaining_test_idx_to_check.append(test_idx)
                # update remaining params as needed
                # when remaining_test_idx_to_check is empty: we have done our job
                if not remaining_test_idx_to_check:
                    break
                test_idx_to_check = remaining_test_idx_to_check
        return subset_indices_per_example

    def _batched_remove_and_refit(
        self,
        x: np.ndarray,
        prediction: int,
        indices_to_remove: np.ndarray,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        indices_to_always_remove: np.ndarray = None,
    ) -> Iterable[int]:
        """
        Splits the dataset into K batches for removal, iteratively removing
        batch 1, 2, ...K until a flip is detected.

        If removing all data does not lead to a flip: we assume that no such
        subset of examples to flip exists

        Once a subset is identified: we try to filter it down by checking
        if that subset removed is less than a threshold, if so, iteratively
        remove to filter down the subset. Note this (re)removes examples
        already collectively removed beforehand, since a collection of examples
        that is not a subset to flip does NOT imply after any subset of that
        subset cannot lead to flip.

        If subset is larger than the threshold, repeat the above steps, using
        the current subset of examples as the new "dataset" to try to see if a
        subset of that can still lead to removal (by again chunking into K
        splits, etc) recursively

        Note that this may not get to the "minimal" subset for removal,
        but would get a "subset" s.t their removal would lead to prediction flip

        Args:
            x (np.ndarray): (hidden_size), an test input
            prediction (int): the model's prediction for that test input
            indices_per_test_to_remove (np.ndarray): (num_examples_to_remove)
            clf (KMeansClassifier): the original model
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            train_labels (np.ndarray): (num_train_examples)
            indices_to_always_remove (np.ndarray, optional): the only difference
                from batched removal. Sometimes we want to iteratively remove
                examples having always removing some set (e.g., in bet. chunks)
                Defaults to None.

        Returns:
            Iterable[int]: empty, or the list of indices that compose a subset
            of training examples s.t. their removal results in a flipped pred
        """
        num_classes = len(np.unique(train_labels))
        sections_indices = partition_indices(
            N=indices_to_remove.shape[0], M=self.SPLITS
        )
        print(
            f"Initiating Batch Removal with {indices_to_remove.size} exs total "
            f"across {self.SPLITS} splits",
            file=sys.stderr,
        )
        # Iteratively remove sections and check for a prediction flip
        for section_idx in tqdm(sections_indices, "Batch Removel"):
            reduced_indices = indices_to_remove[:section_idx]
            if indices_to_always_remove is not None:
                reduced_indices = np.concatenate(
                    [indices_to_always_remove, reduced_indices]
                )
            print(
                f"\nBatching removing the first {section_idx} closest centroid exs\n"
                f"After having removed examples {indices_to_always_remove}\n"
            )
            # print("Reduced indices:", reduced_indices)
            train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
            train_mask[reduced_indices] = False

            # Create a reduced training set without the current section
            reduced_embeddings = train_embeddings[train_mask]
            reduced_labels = train_labels[train_mask]

            # print("Reduced Embedding:", reduced_embeddings.shape)
            # print("Reduced labels:", reduced_labels.shape)

            # if after removal there is only one/less class, then obv flipped
            if len(np.unique(reduced_labels)) == num_classes:
                # Retrain the classifier on the reduced dataset
                new_clf = clone(clf)
                new_clf.fit(reduced_embeddings, reduced_labels)
                new_prediction = new_clf.predict(x.reshape(1, -1))[0]
                # new_prediction = new_clf.predict(x.reshape(1, -1))[0]
                print(
                    f"\nRemoved {reduced_indices.size} examples\n"
                    f"New prediction: {new_prediction}\n"
                    f"Old Prediction: {prediction}\n"
                )
                # Do not need the new clf anymore: call gc
                del new_clf
                gc.collect()
            else:
                print("Not enough data points for all unique classes")
                new_prediction = -1  # assume auto-flip when missing a label cls

            # Check for a prediction flip
            if prediction != new_prediction:
                print(f"\nFound subset with size {reduced_indices.size}.\n")
                if (
                    self._last_seen_subset_size is not None
                    and self._last_seen_subset_size == reduced_indices.size
                    and self._refining_last_split_chunk
                ):
                    # break out early, recursive refining of last chunk
                    # has failed to identify a smaller subset
                    return reduced_indices
                self._last_seen_subset_size = reduced_indices.size
                # Found, but need to split again because very large
                if reduced_indices.size >= self.ITERATIVE_THRESHOLD:
                    # check if subset_indices is reduced: if not, then we
                    # have a problem: must reduce again, from the last
                    # chunk of len(reduced_indices) into `SPLITS` splits
                    if indices_to_remove.size == reduced_indices.size:
                        print(
                            "Recursive refinement failed to identify a smaller"
                            " subset. Initiating refinement of the last split chunk"
                        )
                        self._refining_last_split_chunk = True
                        second_last_chunk_end_idx = partition_indices(
                            N=reduced_indices.shape[0], M=self.SPLITS
                        )[-2]
                        prior_chunk = reduced_indices[
                            0:second_last_chunk_end_idx
                        ]
                        last_chunk_indices = reduced_indices[
                            second_last_chunk_end_idx : reduced_indices.shape[0]
                        ]
                        if last_chunk_indices.size >= self.ITERATIVE_THRESHOLD:
                            print(
                                f"Above thresold {self.ITERATIVE_THRESHOLD}.\n"
                                "Recursively refining last chunk"
                            )
                            subset_indices = self._batched_remove_and_refit(
                                x=x,
                                prediction=prediction,
                                indices_to_remove=last_chunk_indices,
                                clf=clf,
                                train_embeddings=train_embeddings,
                                train_labels=train_labels,
                                indices_to_always_remove=prior_chunk,
                            )
                        else:
                            print(
                                f"Below thresold {self.ITERATIVE_THRESHOLD}.\n"
                                "Iteratively refining last chunk"
                            )
                            subset_indices = self._iterative_remove_and_refit(
                                x=x,
                                prediction=prediction,
                                indices_to_remove=last_chunk_indices,
                                clf=clf,
                                train_embeddings=train_embeddings,
                                train_labels=train_labels,
                                indices_to_always_remove=prior_chunk,
                            )
                    else:
                        self._refining_last_split_chunk = False
                        print(
                            f"Found subset is above the threshold {self.ITERATIVE_THRESHOLD}.\n",
                            "Initiating recursive call to further refine...\n",
                        )
                        subset_indices = self._batched_remove_and_refit(
                            x=x,
                            prediction=prediction,
                            indices_to_remove=reduced_indices,
                            clf=clf,
                            train_embeddings=train_embeddings,
                            train_labels=train_labels,
                        )
                else:
                    self._refining_last_split_chunk = False
                    # iteratively refine, small enough
                    print(
                        f"Found subset is below the threshold {self.ITERATIVE_THRESHOLD}\n",
                        "initiating iterative call to further refine...\n",
                    )
                    subset_indices = self._iterative_remove_and_refit(
                        x=x,
                        prediction=prediction,
                        indices_to_remove=reduced_indices,
                        clf=clf,
                        train_embeddings=train_embeddings,
                        train_labels=train_labels,
                    )
                return subset_indices
        return []  # No subset found

    def _iterative_remove_and_refit(
        self,
        x: np.ndarray,
        prediction: int,
        indices_to_remove: np.ndarray,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        indices_to_always_remove: np.ndarray = None,
    ) -> Iterable[int]:
        """
        Iteratively remove examples as indexed by indices_to_remove to find a
        subset of examples s.t. their removal flips the prediction x

        Args:
            x (np.ndarray):
            prediction (int):
            indices_to_remove (np.ndarray):
            clf (KMeansClassifier):
            train_embeddings (np.ndarray):
            train_labels (np.ndarray):
            indices_to_always_remove (np.ndarray, optional): the only difference
                from batched removal. Sometimes we want to iteratively remove
                examples having always removing some set (e.g., in bet. chunks)
                Defaults to None.

        Returns:
            Iterable[int]: empty, or the list of indices that compose a subset
            of training examples s.t. their removal results in a flipped pred
        """

        num_classes = len(np.unique(train_labels))
        for i in tqdm(
            range(1, indices_to_remove.shape[0] + 1), "Iterative Removal"
        ):
            size = 0
            if indices_to_always_remove is not None:
                reduced_indices = np.concatenate(
                    [indices_to_always_remove, reduced_indices]
                )
                size = indices_to_always_remove.size
            print(
                f"\nIteratively removing the first {i} centroid examples.\n"
                f"After having removed {size} examples\n"
            )
            reduced_indices = indices_to_remove[:i]

            # mask to keep track of which training examples to keep
            train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
            # Exclude selected examples from the training set
            train_mask[reduced_indices] = False
            X_train = train_embeddings[train_mask]
            y_train = train_labels[train_mask]

            # Clone the original model and retrain, unless there is not enough
            # labels per unique class
            if len(np.unique(y_train)) == num_classes:
                new_clf = clone(clf)
                new_clf.fit(X_train, y_train)
                new_prediction = new_clf.predict(x.reshape(1, -1))[0]
                # new_prediction = new_clf.predict(x.reshape(1, -1))[0]
            else:
                print("Not enough data points for all unique classes")
                new_prediction = -1

            # Check if the prediction has flipped
            if new_prediction != prediction:
                print(
                    f"\nIterative removal of examples {reduced_indices}"
                    " lead to flipped prediction.\n"
                )
                return reduced_indices
        return []  # no subset can flip

    def find_minimal_subset_brute_force(
        self,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ):
        # Find subsets through a brute force, searching fashion
        # This is good because examples can be chunked to be embarassingly
        # parallel
        # preprocessing is equivalent to finding example-based explanations
        handler = LMeansExampleBasedExplanation()
        sorted_indices_per_test_example = handler.get_explanation_indices(
            M=None,  # want indices for all centroid examples
            clf=clf,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
        )

        subset_indices_per_example = []
        predictions = clf.predict(test_embeddings)
        for i, (x, prediction, indices_to_remove) in (
            pbar := tqdm(
                enumerate(
                    zip(
                        test_embeddings,
                        predictions,
                        sorted_indices_per_test_example,
                    )
                ),
                total=predictions.size,
            )
        ):
            pbar.set_description(f"Finding Minimal Set for Example {i}\n")
            # reset globals for each input
            self._last_seen_subset_size = None
            self._refining_last_split_chunk = False
            # if total training set less than threshold, just go to iterative
            if train_embeddings.shape[0] <= self.ITERATIVE_THRESHOLD:
                subset_indices = self._iterative_remove_and_refit(
                    x=x,
                    prediction=prediction,
                    indices_to_remove=indices_to_remove,
                    clf=clf,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                )
            else:
                subset_indices = self._batched_remove_and_refit(
                    x=x,
                    prediction=prediction,
                    indices_to_remove=indices_to_remove,
                    clf=clf,
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                )
            # convert from numpy to list for serialization later, if need be
            if isinstance(subset_indices, np.ndarray):
                subset_indices = subset_indices.tolist()
            subset_indices_per_example.append(subset_indices)

        return subset_indices_per_example

    def find_minimal_subset(
        self,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ):
        if self.SPLITS is not None and self.ITERATIVE_THRESHOLD is not None:
            return self.find_minimal_subset_brute_force(
                clf=clf,
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_labels=train_labels,
            )
        else:
            return self.find_minimal_subset_cluster_batched(
                clf=clf,
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_labels=train_labels,
            )

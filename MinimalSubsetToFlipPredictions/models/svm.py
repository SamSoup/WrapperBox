# This script finds the minimal set of examples required to flip a svm pred
from sympy import reduced
from MinimalSubsetToFlipPredictions.models.interface import FindMinimalSubset
from ExampleBasedExplanations.svm import SVMExampleBasedExplanation
from utils.models.svm import get_support_vectors, project_inputs
from utils.partition import partition_indices
from sklearn.neighbors import KNeighborsClassifier
from typing import Iterable, List
from sklearn.metrics.pairwise import euclidean_distances
from utils.constants.models import SVM_BASE
from sklearn.base import clone
from sklearn.svm import LinearSVC
from tqdm import tqdm
import numpy as np


class FindMinimalSubsetSVM(FindMinimalSubset):

    def __init__(self, ITERATIVE_THRESHOLD: int, SPLITS: int) -> None:
        super().__init__()
        # Parameters for batched removal
        self.SPLITS = SPLITS
        self.ITERATIVE_THRESHOLD = ITERATIVE_THRESHOLD

    def _batched_remove_and_refit(
        self,
        x: np.ndarray,
        prediction: int,
        indices_to_remove: np.ndarray,
        clf: LinearSVC,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
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
            clf (LinearSVC): the original model
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            train_labels (np.ndarray): (num_train_examples)

        Returns:
            Iterable[int]: empty, or the list of indices that compose a subset
            of training examples s.t. their removal results in a flipped pred
        """
        num_classes = len(np.unique(train_labels))
        sections_indices = partition_indices(
            N=indices_to_remove.shape[0], M=self.SPLITS
        )
        # print(indices_to_remove.shape)
        # print(sections_indices)
        # print(train_embeddings.shape)
        # Iteratively remove sections and check for a prediction flip
        for section_idx in tqdm(sections_indices, "Batch Removel"):
            print(f"Batching removing the first {section_idx} closest SVs")
            reduced_indices = indices_to_remove[:section_idx]
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
                svs, _ = get_support_vectors(new_clf, reduced_embeddings)
                print(f"\nNem SVM has {svs.shape[0]}" "support vectors\n")
                new_prediction = new_clf.predict(x.reshape(1, -1))[0]
                print(
                    f"\nRemoved {reduced_indices.size} examples\n"
                    f"New prediction: {new_prediction}\n"
                    f"Old Prediction: {prediction}\n"
                )
            else:
                print("Not enough data points for all unique classes")
                new_prediction = -1  # assume auto-flip when missing a label cls

            # Check for a prediction flip
            if prediction != new_prediction:
                # Found, but need to split again
                if reduced_indices.size >= self.ITERATIVE_THRESHOLD:
                    # check if subset_indices is reduced: if not, then we
                    # have a problem: must reduce iteratively, from the last
                    # chunk of len(reduced_indices) into `SPLITS` splits
                    if indices_to_remove.shape[0] == reduced_indices.shape[0]:
                        print(
                            "Recursive refinement failed to identify a smaller"
                            " subset, initiating iterative refinement of the "
                            "last split chunk"
                        )
                        second_last_chunk_end_idx = partition_indices(
                            N=reduced_indices.shape[0], M=self.SPLITS
                        )[-2]
                        prior_chunk = reduced_indices[
                            0:second_last_chunk_end_idx
                        ]
                        last_chunk_indices = reduced_indices[
                            second_last_chunk_end_idx : reduced_indices.shape[0]
                        ]
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
                        print(
                            f"\nFound subset with size {reduced_indices.size}.\n"
                            f"It is above the threshold {self.ITERATIVE_THRESHOLD}\n",
                            "initiating recursive call to further refine...\n",
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
                    # iteratively refine, small enough
                    print(
                        f"\nFound subset with size {reduced_indices.size}.\n"
                        f"It is below the threshold {self.ITERATIVE_THRESHOLD}\n",
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
        clf: LinearSVC,
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
            clf (LinearSVC):
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
            print(
                f"Iteratively removing the first {i} examples.\n"
                f"After having removed examples {indices_to_always_remove}\n"
            )
            reduced_indices = indices_to_remove[:i]
            if indices_to_always_remove is not None:
                reduced_indices = np.concatenate(
                    [indices_to_always_remove, reduced_indices]
                )
            # mask to keep track of which training examples to keep
            train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
            # Exclude selected examples from the training set
            train_mask[reduced_indices] = False
            X_train = train_embeddings[train_mask]
            y_train = train_labels[train_mask]

            # Clone the original model and retrain, unless there is not enough
            # labels per unique class
            if len(np.unique(y_train)) == num_classes:
                new_svm_clf = clone(clf)
                new_svm_clf.fit(X_train, y_train)
                new_prediction = new_svm_clf.predict(x.reshape(1, -1))[0]
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

    def find_minimal_subset(
        self,
        clf: LinearSVC,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        # the preprocessing is equivalent to finding the example-based
        # explanations for SVM
        handler = SVMExampleBasedExplanation()
        svs, _ = get_support_vectors(clf, train_embeddings)
        sorted_indices_per_test_example = handler.get_explanation_indices(
            M=svs.shape[0],  # want indices to all sv
            clf=clf,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
        )

        # print(sorted_indices_per_test_example)
        # print(train_embeddings.shape)
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
            )
        ):
            pbar.set_description(f"Finding Minimal Set for Example {i}\n")
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

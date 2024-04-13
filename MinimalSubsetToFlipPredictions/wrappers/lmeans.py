# This script finds the minimal set of examples required to flip a lmeans pred

from collections import defaultdict
from typing import List
from sklearn.base import clone
from tqdm import tqdm
from ExampleBasedExplanations.lmeans import LMeansExampleBasedExplanation
from MinimalSubsetToFlipPredictions.wrappers.interface import FindMinimalSubset
from classifiers import KMeansClassifier
import numpy as np


class FindMinimalSubsetLMeans(FindMinimalSubset):
    def __init__(self) -> None:
        super().__init__()
        # PARAMS here

    def find_minimal_subset(
        self,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        # the preprocessing is equivalent to finding the example-based
        # explanations for SVM
        handler = LMeansExampleBasedExplanation()
        indices_per_cluster_idx = handler._get_cluster_idx_to_explanation(
            M=None,  # want all examples
            clf=clf,
            train_embeddings=train_embeddings,
        )
        # construct a dict mapping cluster_idx: test_example_idx
        cluster_ids_test = clf.kmeans_.predict(test_embeddings)
        predictions = clf.predict(test_embeddings)
        print(f"Original predictions: {predictions}")
        cluster_idx_to_test_idx = defaultdict(list)
        for i, value in enumerate(cluster_ids_test):
            cluster_idx_to_test_idx[value].append(i)
        # procedure: iteratively remove cluster examples and recluster,
        # then, check if predictions are flipped for the example
        # in that cluster only (because other examples would have a
        # different removal order)
        subset_indices_per_example = [
            [] for _ in range(test_embeddings.shape[0])
        ]
        num_classes = len(np.unique(train_labels))
        for cluster_idx, indices in indices_per_cluster_idx.items():
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

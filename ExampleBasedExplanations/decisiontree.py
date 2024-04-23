from sklearn.tree import DecisionTreeClassifier
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from scipy.spatial.distance import cdist
from typing import List
import numpy as np


class DecisionTreeExampleBasedExplanation(ExampleBasedExplanation):
    def get_explanation_indices(
        self,
        M: int,
        clf: DecisionTreeClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        Return the

        Args:
            clf (DecisionTreeClassifier):
            M (int): number of examples to return for explanation, = K here
            dataset (DatasetDict): must have at least {
                "train": ['text', 'label'...]
                "test": ['text', 'label'...]
            }
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            test_embeddings (np.ndarray): (num_test_examples, hidden_size)

        Returns:
            np.ndarray: The M closest leaf example indices for each test example,
                (num_test_examples, M)
        """

        leaf_ids_train = clf.apply(train_embeddings)
        leaf_ids_test = clf.apply(test_embeddings)

        indices = []
        for i in range(test_embeddings.shape[0]):
            # [0, num_train_examples)
            train_indices = np.arange(leaf_ids_train.shape[0])
            leaf_idx = leaf_ids_test[i]
            # only obtain the training examples in the same leaf
            # as the test example
            train_leaf_neighbor_mask = leaf_ids_train == leaf_idx
            train_indices_subset = train_indices[train_leaf_neighbor_mask]
            # unforunately, the set of examples to compute the distance is
            # different per input examples, and thus is not easily paralleled
            # TODO: could simply group test examples by leaf, then compute
            # all in parallel to save some time?
            dist_mat = cdist(
                test_embeddings[i].reshape(1, -1),
                train_embeddings[train_leaf_neighbor_mask],
            )
            # one potential complications here is that there may not be M
            # examples in a leaf, if that leaf is particularly small
            top_k = np.argsort(dist_mat, axis=1)[:, :M].flatten()
            indices.append(train_indices_subset[top_k])

        return indices

# Class to find the ExampleBasedExplanations for KNN
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from sklearn.neighbors import KNeighborsClassifier
from datasets import DatasetDict
from typing import List
import numpy as np


class KNNExampleBasedExplanation(ExampleBasedExplanation):
    def get_explanation_indices(
        self,
        M: int,
        clf: KNeighborsClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        For KNN, the example-based explanations for
        `test` examples are simply its k nearest neighbor

        Args:
            M (int): number of examples to return for explanation, = K here
            clf (KNeighborsClassifier):
            dataset (DatasetDict): must have at least {
                "train": ['text', 'label'...]
                "test": ['text', 'label'...]
            }
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            test_embeddings (np.ndarray): (num_test_examples, hidden_size)

        Returns:
            List[List[int]]: The M neighbor indices for each test example
        """

        # obtain the neighbor indices
        neigh_indices = clf.kneighbors(
            X=test_embeddings, n_neighbors=M, return_distance=False
        )

        return neigh_indices.tolist()

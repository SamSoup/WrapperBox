# Class to find the ExampleBasedExplanations for SVM
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from classifiers.KMedoidsClassifier import KMedoidsClassifier
from utils.models.svm import project_inputs
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from typing import List, Dict
from datasets import DatasetDict
import numpy as np


class KMedoidsExampleBasedExplanation(ExampleBasedExplanation):
    def _get_all_medoids_distances(
        self,
        clf: KMedoidsClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distances from all medoids to all training examples.

        Args:
            clf (KMedoidsClassifier): The classifier with medoids information.
            train_embeddings (np.ndarray): Embeddings of the training examples.

        Returns:
            np.ndarray: Distance matrix from each medoid to each training example.
        """
        medoid_indices = clf.kmedoids_.medoid_indices_
        medoid_embeddings = train_embeddings[medoid_indices]
        return cdist(test_embeddings, medoid_embeddings)

    def get_explanation_indices(
        self,
        M: int,
        clf: KMedoidsClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        Returns the indices of the M closest medoids to each test example.

        Args:
            M (int): Number of medoids to return for each test example. When M
                is None or exceeds the number of medoids, return all medoids
                sorted by distance.
            clf (KMedoidsClassifier):
            train_embeddings (np.ndarray): Embeddings of the training examples.
            test_embeddings (np.ndarray): Embeddings of the test examples.

        Returns:
            List[List[int]]: The M closest medoid indices to each test example, sorted by distance.
        """
        # Get distance matrix from all medoids to all train examples
        dist_mat = self._get_all_medoids_distances(
            clf, train_embeddings, test_embeddings
        )

        indices = []
        for dist_row in dist_mat:
            if M is None or M >= len(clf.kmedoids_.medoid_indices_):
                sorted_indices = np.argsort(dist_row)
            else:
                sorted_indices = np.argsort(dist_row)[:M]
            indices.append(sorted_indices.tolist())

        return indices

    def get_explanation_indices_2(self):
        """
        Only for LMeans: another way to retrieve example-based explanations
        might be finding the M nearest examples to the input examples
        (within the same cluster: like clustered KNN)
        """
        pass

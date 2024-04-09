# Class to find the ExampleBasedExplanations for SVM
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from classifiers.KMeansClassifier import KMeansClassifier
from utils.models.svm import project_inputs
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from typing import List
from datasets import DatasetDict
import numpy as np


class KMeansExampleBasedExplanation(ExampleBasedExplanation):
    def get_explanation_indices(
        self,
        M: int,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        Returns the indices to the M closest example to the cluster centroid,
        where each centroid maps to a label class

        Args:
            M (int): number of examples to return for explanation
            clf (KMeansClassifier):
            dataset (DatasetDict): must have at least {
                "train": ['text', 'label'...]
                "test": ['text', 'label'...]
            }
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            test_embeddings (np.ndarray): (num_test_examples, hidden_size)

        Returns:
            List[List[int]]: The M closest example indices to the corr. cluster
                centroid for each test example
        """

        cluster_ids_train = clf.kmeans_.labels_
        cluster_ids_test = clf.kmeans_.predict(test_embeddings)

        centroids = clf.kmeans_.cluster_centers_
        cluster_indices_to_label_ = clf.cluster_indices_to_label_
        centroids, cluster_indices_to_label_

        # for each cluster, compute the indices the closest M training examples to the
        # centroids
        cluster_idx_to_explanation = {}
        for cluster_idx in cluster_indices_to_label_:
            # filter down the training set to only those in the cluster
            train_indices = np.arange(train_embeddings.shape[0])
            train_mask = cluster_ids_train == cluster_idx
            train_indices_subset = train_indices[train_mask]
            dist_mat = cdist(
                centroids[cluster_idx].reshape(1, -1),
                train_embeddings[train_mask],
            )
            top_k = np.argsort(dist_mat, axis=1)[:, :M].flatten()
            cluster_idx_to_explanation[cluster_idx] = train_indices_subset[
                top_k
            ]
        # now, for each test example, find its corresponding cluster, and then record
        # its explanation indices
        indices = []
        for i in range(test_embeddings.shape[0]):
            cluster_id = cluster_ids_test[i]
            indices.append(cluster_idx_to_explanation[cluster_id])

        return indices

    def get_explanation_indices_2(self):
        """
        Only for LMeans: another way to retrieve example-based explanations
        might be finding the M nearest examples to the input examples
        (within the same cluster: like clustered KNN)
        """
        pass

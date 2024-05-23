# Class to find the ExampleBasedExplanations for SVM
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from classifiers.KMeansClassifier import KMeansClassifier
from utils.models.svm import project_inputs
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from typing import List, Dict
from datasets import DatasetDict
import numpy as np


class LMeansExampleBasedExplanation(ExampleBasedExplanation):
    def _get_cluster_idx_to_explanation(
        self,
        M: int,
        clf: KMeansClassifier,
        train_embeddings: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Similar to get_explanation_indices, but instead of returning a list
        of training example indices per test examples, returns the list of
        indices per cluster centroid/idx instead, which will be the same for
        all examples from the same cluster/idx
        """
        cluster_ids_train = clf.kmeans_.labels_
        centroids = clf.kmeans_.cluster_centers_
        # for each cluster, compute the indices of the
        # closest M training examples to the cluster centroids
        cluster_idx_to_explanation = {}
        print(cluster_ids_train, centroids, clf.kmeans_.n_clusters)
        for cluster_idx in range(clf.kmeans_.n_clusters):
            # filter down the training set to only those in the cluster
            train_indices = np.arange(train_embeddings.shape[0])
            train_mask = cluster_ids_train == cluster_idx
            train_indices_subset = train_indices[train_mask]
            dist_mat = cdist(
                centroids[cluster_idx].reshape(1, -1),
                train_embeddings[train_mask],
            )
            if M is None:
                # assume we want all examples
                M = train_indices_subset.size
            top_k = np.argsort(dist_mat, axis=1)[:, :M].flatten()
            cluster_idx_to_explanation[cluster_idx] = train_indices_subset[
                top_k
            ]

        print(cluster_idx_to_explanation)
        return cluster_idx_to_explanation

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
            M (int): number of examples to return for explanation. When M is
                None, assume we want all examples in the cluster ranked
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
        cluster_idx_to_explanation = self._get_cluster_idx_to_explanation(
            M=M,
            clf=clf,
            train_embeddings=train_embeddings,
        )

        # now, for each test example, find its corresponding cluster,
        # then record its explanation indices
        indices = []
        cluster_ids_test = clf.kmeans_.predict(test_embeddings)
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

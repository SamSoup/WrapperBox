# Class to find the ExampleBasedExplanations for SVM
from ExampleBasedExplanations.interface import ExampleBasedExplanation
from sklearn.svm import LinearSVC
from utils.models.svm import project_inputs
from sklearn.metrics.pairwise import euclidean_distances
from typing import List
from datasets import DatasetDict
import numpy as np


class SVMExampleBasedExplanation(ExampleBasedExplanation):
    def get_explanation_indices(
        self,
        M: int,
        clf: LinearSVC,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        """
        Returns the indices to the M closest support vector to the projected
        point on the hyperplane (perpendicular to that of the test inputs)
        as the example_based explanations

        *Not that M here should NOT exceed the maximum number of support vectors,
        although python indexing simply returns the original list anyway

        Args:
            M (int): number of examples to return for explanation, = K here
            clf (SVC):
            dataset (DatasetDict): must have at least {
                "train": ['text', 'label'...]
                "test": ['text', 'label'...]
            }
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            test_embeddings (np.ndarray): (num_test_examples, hidden_size)


        Returns:
            List[List[int]]: The M closest sv indices for each test example
        """
        projected_vectors = project_inputs(inputs=test_embeddings, clf=clf)
        dist_to_support_vectors = euclidean_distances(
            projected_vectors, clf.support_vectors_
        )

        # print(projected_vectors)
        # print(dist_to_support_vectors)

        # Indices of support vectors, sorted ascendingly by distance
        # from 0 to num_of_support_vectors; this will be translated/mapped
        # to 0 to num_of_train_eval_examples
        sorted_indices_per_test_example = np.argsort(
            dist_to_support_vectors, axis=1
        )

        # only return the M closest sv, note these are
        # indices within [0, num_support_vectors)
        sorted_indices_per_test_example = sorted_indices_per_test_example[:, :M]

        # print(sorted_indices_per_test_example)
        # Translate indices of support vectors to indices of training example
        original_indices = clf.support_[sorted_indices_per_test_example]

        return original_indices

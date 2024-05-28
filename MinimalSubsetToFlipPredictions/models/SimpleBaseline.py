## A simple baseline for finding the minium subset to flip predictions
## for a classification task is to just remove all training instances that
## has the same label as the prediction


import sys
from typing import Any, List

from lightgbm import LGBMModel
import numpy as np
from sklearn.base import BaseEstimator
from MinimalSubsetToFlipPredictions.models.interface import FindMinimalSubset
from utils.models import get_predictions


class FindMinimalSubsetSimpleBaseline(FindMinimalSubset):
    def find_minimal_subset(
        self,
        clf: BaseEstimator,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        predictions = get_predictions(clf, test_embeddings)

        # for each prediction value, cache the example indices to remove
        # and then simply assign: this is beneficial because
        # num_samples >> num_classes
        unique_predictions = np.unique(predictions)

        # Cache the indices to remove per unique prediction
        removal_indices_cache = {}

        for pred in unique_predictions:
            # Find training examples with matching labels as prediction
            indices_to_remove = np.where(train_labels == pred)[0]
            removal_indices_cache[pred] = indices_to_remove.tolist()

        # For each test example, record the indices to remove based on its prediction
        indices_to_remove_per_test = []

        for pred in predictions:
            indices_to_remove_per_test.append(removal_indices_cache[pred])

        return indices_to_remove_per_test

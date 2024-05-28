## A simple baseline for finding the minium subset to flip predictions
## for a classification task is to just remove all training instances that
## has the same label as the prediction


import sys
from typing import Any, List

import numpy as np
from sklearn.base import BaseEstimator
from MinimalSubsetToFlipPredictions.interface import FindMinimalSubset


class FindMinimalSubsetSimpleBaseline(FindMinimalSubset):
    def find_minimal_subset(
        self,
        clf: BaseEstimator,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        predictions = clf.predict(test_embeddings)

        # for each prediction value, cache the example indices to remove
        # and then simply assign: this is beneficial because
        # num_samples >> num_classes
        unique_predictions = np.unique(predictions)

        # reduced_indices = indices_to_remove[:section_idx]
        # # print("Reduced indices:", reduced_indices)
        # train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
        # train_mask[reduced_indices] = False

        # # Create a reduced training set without the current section
        # reduced_embeddings = train_embeddings[train_mask]
        # reduced_labels = train_labels[train_mask]
        pass

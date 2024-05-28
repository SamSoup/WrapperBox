from typing import List
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from tqdm import tqdm

from utils.models import get_predictions


def retrain_and_evaluate_validity(
    clf: BaseEstimator,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    x_test: np.ndarray,
    indices_to_exclude: np.ndarray,
):
    train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
    train_mask[indices_to_exclude] = False
    reduced_embeddings = train_embeddings[train_mask]
    reduced_labels = train_labels[train_mask]
    old_pred = get_predictions(clf, x_test.reshape(1, -1))[0]
    new_clf = clone(clf)
    new_clf.fit(reduced_embeddings, reduced_labels)
    new_pred = get_predictions(new_clf, x_test.reshape(1, -1))[0]
    # this subset is valid only if new prediction does not equal old prediction
    return old_pred, new_pred, new_pred != old_pred


def evaluate_predictions(
    clf: BaseEstimator,
    flip_list: List[List[int]],
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    ex_indices_to_check: List[int],
):
    is_valid_subsets = []
    for test_ex_idx in tqdm(ex_indices_to_check):
        f_list = flip_list[test_ex_idx]
        # if flip list is empty: then it is obviously false
        if f_list is None or len(f_list) == 0:
            is_valid_subsets.append(False)
            continue
        _, _, is_valid_subset = retrain_and_evaluate_validity(
            clf=clf,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            x_test=test_embeddings[test_ex_idx],
            indices_to_exclude=f_list,
        )
        is_valid_subsets.append(is_valid_subset)

    return is_valid_subsets

import copy
from typing import Dict, Iterable, List, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm

from utils.models import get_predictions


def retrain_and_refit(
    clf: BaseEstimator,
    indices_to_exclude: List[int],
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    x_test: np.ndarray,
):
    train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
    train_mask[indices_to_exclude] = False
    reduced_embeddings = train_embeddings[train_mask]
    reduced_labels = train_labels[train_mask]
    old_pred = clf.predict_proba(x_test.reshape(1, -1))
    new_clf = clone(clf)
    new_clf.fit(reduced_embeddings, reduced_labels)
    new_pred = new_clf.predict_proba(x_test.reshape(1, -1))
    return old_pred, new_pred, new_pred - old_pred


def retrain_and_evaluate_validity_flip_variant(
    clf: BaseEstimator,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    x_test: np.ndarray,
    indices_to_flip: np.ndarray,
):
    """
    Retrains the classifier after flipping the label of certain indices
    from the training data, and evaluates the validity of the new predictions.

    Parameters:
    - clf: BaseEstimator
        The classifier to be retrained.
    - train_embeddings: np.ndarray
        The training data embeddings.
    - train_labels: np.ndarray
        The training data labels.
    - x_test: np.ndarray
        The test sample for which to evaluate predictions.
    - indices_to_flip: np.ndarray
        The indices of the training data to flip their labels.

    Returns:
    - old_pred: The prediction of the original classifier.
    - new_pred: The prediction of the new classifier.
    - is_valid: Boolean indicating if the new prediction is different from the old prediction.
    """

    train_mask = np.zeros(train_embeddings.shape[0], dtype=int)
    train_mask[indices_to_flip] = 1
    new_labels = train_labels.astype(int) ^ train_mask
    old_pred = get_predictions(clf, x_test.reshape(1, -1))[0]
    new_clf = clone(clf)
    new_clf.fit(train_embeddings, new_labels)
    new_pred = get_predictions(new_clf, x_test.reshape(1, -1))[0]
    # this subset is valid only if new prediction does not equal old prediction
    return old_pred, new_pred, new_pred != old_pred


def retrain_and_evaluate_validity(
    clf: BaseEstimator,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    x_test: np.ndarray,
    indices_to_exclude: np.ndarray,
):
    """
    Retrains the classifier after excluding certain indices from the training data,
    and evaluates the validity of the new predictions.

    Parameters:
    - clf: BaseEstimator
        The classifier to be retrained.
    - train_embeddings: np.ndarray
        The training data embeddings.
    - train_labels: np.ndarray
        The training data labels.
    - x_test: np.ndarray
        The test sample for which to evaluate predictions.
    - indices_to_exclude: np.ndarray
        The indices of the training data to be excluded.

    Returns:
    - old_pred: The prediction of the original classifier.
    - new_pred: The prediction of the new classifier.
    - is_valid: Boolean indicating if the new prediction is different from the old prediction.
    """

    train_mask = np.ones(train_embeddings.shape[0], dtype=bool)
    train_mask[indices_to_exclude] = False
    reduced_embeddings = train_embeddings[train_mask]
    reduced_labels = train_labels[train_mask]
    old_pred = get_predictions(clf, x_test.reshape(1, -1))[0]
    new_clf = clone(clf)
    new_clf.fit(reduced_embeddings, reduced_labels)
    new_pred = get_predictions(new_clf, x_test.reshape(1, -1))[0]
    # this subset is valid only if new prediction does not equal old prediction
    # print(old_pred, new_pred)
    return old_pred, new_pred, new_pred != old_pred


def evaluate_predictions_flip_variant(
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
        _, _, is_valid_subset = retrain_and_evaluate_validity_flip_variant(
            clf=clf,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            x_test=test_embeddings[test_ex_idx],
            indices_to_flip=f_list,
        )
        is_valid_subsets.append(is_valid_subset)

    return is_valid_subsets


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


def compute_coverage(flip_list: List[List[int]]):
    total = len(flip_list)
    identified_subset = 0
    for l in flip_list:
        if l is not None and len(l) > 0:
            identified_subset += 1

    coverage = round(identified_subset / total * 100, 2)
    frac_str = f"{identified_subset}/{total}"
    print(f"Identified {frac_str} subsets.")
    print(f"Coverage: {coverage}%")

    return [coverage, frac_str]


def compute_validity(flip_list: List[List[int]], is_valid: List[bool]):
    total = len(flip_list)
    valid = 0
    identified_subset = 0
    subset_sizes = []
    for i, l in enumerate(flip_list):
        if l is not None and len(l) > 0:
            identified_subset += 1
            if is_valid[i]:
                valid += 1
                subset_sizes.append(len(l))

    # NOTE: this computes valid / total; can also compute precision
    total_validity = round(valid / total * 100, 2)
    precision = round(valid / identified_subset * 100, 2)

    validity_str = f"{valid}/{total}"
    precision_str = f"{valid}/{identified_subset}"
    print(f"Overall validity is {validity_str}, or {total_validity}%")
    print(f"Precision validity is {precision_str}, or {precision}%")

    return [total_validity, validity_str, precision, precision_str]


def compute_valid_subset_sizes(
    flip_list: List[List[int]], is_valid: List[bool]
):
    subset_sizes = []
    for l, val in zip(flip_list, is_valid):
        if l is not None and len(l) > 0 and val:
            subset_sizes.append(len(l))

    return subset_sizes


def compute_median_sizes(flip_list: List[List[int]], is_valid: List[bool]):
    subset_sizes = compute_valid_subset_sizes(flip_list, is_valid)

    med_size = np.median(np.array(subset_sizes))

    print(
        f"Median Valid Subset Sizes is {med_size}, out of {len(subset_sizes)} valid subsets"
    )

    return med_size


# check of the non_empty sets, how many are actually valid
def compute_subset_metrics(flip_list: List[List[int]], is_valid: List[bool]):
    validity, v_str, precision, p_str = compute_validity(flip_list, is_valid)
    metrics = {
        "Coverage": compute_coverage(flip_list),
        "Validity": [validity, v_str],
        "Precision Validity": [precision, p_str],
        "Median Size": compute_median_sizes(flip_list, is_valid),
    }

    return metrics


# Custom evaluate function using Yang's prediction probability outputs
# Takes in new predictions and old predictions
def evaluate_by_prediction_probas(
    old_predictions: Iterable[float],
    new_predictions: Iterable[float],
    thresh: float = 0.5,
) -> Iterable[bool]:
    is_valid = []
    for po, pn in zip(old_predictions, new_predictions):
        if pn is None:
            is_valid.append(False)
            continue
        old_p = po >= thresh
        new_p = pn >= thresh
        if old_p != new_p:
            is_valid.append(True)
        else:
            is_valid.append(False)

    return is_valid


def evaluate_by_class(
    old_predictions: np.ndarray,
    flip_list: List[List[Union[int, None]]],
    is_valid: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    # Instead of aggregate validity, evaluate by class
    # 1. Positive -> Negative flips
    # 2. Negative -> Positive flips
    # 3. Aggregate
    # We do this by checking the old prediction and separating it into two class
    class_metrics = {}
    for c in np.unique(old_predictions):
        indices = np.where(old_predictions == c)[0]

        # Take only those that belong to class c
        old_subset = old_predictions[indices]
        is_valid_subset = is_valid[indices]
        flip_list_subset = [flip_list[i] for i in indices]

        metrics = compute_subset_metrics(
            flip_list=flip_list_subset, is_valid=is_valid_subset
        )
        class_metrics[c] = metrics
    return class_metrics

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


def compute_coverage(flip_list: List[List[int]]):
    total = len(flip_list)
    identified_subset = 0
    for l in flip_list:
        if l is not None and len(l) > 0:
            identified_subset += 1

    coverage = round(identified_subset / total * 100, 2)

    print(f"Identified {identified_subset}/{total} subsets.")
    print(f"Coverage: {coverage}%")

    return coverage


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

    print(f"{valid}/{identified_subset} identified subsets are valid")
    print(f"Overall validity is {valid}/{total}, or {total_validity}%")
    print(f"Precision validity is {valid}/{identified_subset}, or {precision}%")

    return total_validity, precision


def compute_median_sizes(flip_list: List[List[int]], is_valid: List[bool]):
    subset_sizes = []
    for l, val in zip(flip_list, is_valid):
        if l is not None and len(l) > 0 and val:
            subset_sizes.append(len(l))

    med_size = np.median(np.array(subset_sizes))

    print(
        f"Median Valid Subset Sizes is {med_size}, out of {len(subset_sizes)} valid subsets"
    )

    return med_size


# check of the non_empty sets, how many are actually valid
def compute_subset_metrics(flip_list: List[List[int]], is_valid: List[bool]):
    overall_validity, precision_validity = compute_validity(flip_list, is_valid)
    metrics = {
        "Coverage": compute_coverage(flip_list),
        "Overall Validity": overall_validity,
        "Precision Validity": precision_validity,
        "Median Size": compute_median_sizes(flip_list, is_valid),
    }

    return metrics

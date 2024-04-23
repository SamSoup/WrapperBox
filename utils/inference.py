from lib2to3.pytree import Base
from typing import Iterable, Union, Dict, List, Tuple
from scipy import stats
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, clone
from random import randint
from tqdm import tqdm
import numpy as np
import itertools


def find_majority_batched(votes: np.ndarray):
    """
    votes: (n_samples, n_votes), each vote is a label

    Caution: this will always prioritize the smallest mode as the return value
    use only when there is an odd number of votes

    When labels are 0 or 1, this could be done by
    # np.mean(current_window, axis=1) > 0.5
    but this would not work for multi-label scenarios
    """
    mode, count = stats.mode(votes, axis=1, keepdims=False)
    return mode


def find_majority(votes: Iterable[Union[str, int]]):
    """
    Given a set of votes, find the majority vote

    Args:
        votes (_type_): _description_

    Returns:
        _type_: _description_
    """
    vote_count = Counter(votes)
    tops = vote_count.most_common(1)
    if len(tops) > 1:
        # break ties randomly
        idx = randint(0, len(tops) - 1)
        return tops[idx][0]
    return tops[0][0]


def cross_validation_with_grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, List[Union[str, int]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    scoring=f1_score,
) -> Tuple[BaseEstimator, float, Dict[str, List[Union[str, int]]]]:
    """
    Perform grid search with manual cross-validation for hyperparameter tuning.
    Different from sklearn.GridSearchCV, this uses the same validation set

    Args:
        estimator: An uninitialized sklearn-compatible estimator.
        param_grid (dict): Dictionary with parameters names (`str`) as keys and
            lists of parameter settings to try as values.
        X_train, y_train (np.array): Training data and labels.
        X_eval, y_eval (np.array): Validation data and labels.
        scoring (callable): A callable to evaluate the predictions on the
            validation set.

    Returns:
        best_model: The model with the best hyperparameters, fitted on the
            combined training and validation dataset.
        best_score: The best score achieved on the validation set.
        best_params: The set of hyperparameters that achieved the best score.
    """
    best_score = float("-inf")
    best_params = None
    best_model = None

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    for params in tqdm(itertools.product(*values), f"Grid Searching..."):
        hyperparams = dict(zip(keys, params))

        # Initialize the model with the current set of hyperparameters
        model = clone(estimator)
        model.set_params(**hyperparams)

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_pred_eval = model.predict(X_eval)
        score = scoring(y_eval, y_pred_eval)

        # Update best_score, best_params, and best_model if the current model
        # is better
        if score > best_score:
            best_score = score
            best_params = hyperparams
            best_model = model

    # Optionally, refit the best model on the combined training and
    # validation dataset
    X_combined = np.vstack((X_train, X_eval))
    y_combined = np.hstack((y_train, y_eval))
    best_model.fit(X_combined, y_combined)

    return best_model, best_score, best_params


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, prefix: str, is_multiclass: bool = False):
    if is_multiclass:
        results = {}
        results[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)
        agg = ["micro", "macro", "weighted"]
        for avg in agg:
            results[f"{prefix}_{avg}_f1"] = f1_score(
                y_true, y_pred, average=avg
            )
            results[f"{prefix}_{avg}_precision"] = precision_score(
                y_true, y_pred, average=avg
            )
            results[f"{prefix}_{avg}_recall"] = recall_score(
                y_true, y_pred, average=avg
            )
        # simply set accuracy, precision, recall, f1 = macro-averaged
        results[f"{prefix}_f1"] = results[f"{prefix}_macro_f1"]
        results[f"{prefix}_precision"] = results[f"{prefix}_macro_precision"]
        results[f"{prefix}_recall"] = results[f"{prefix}_macro_recall"]
        # also get per class f1, precision, recall
        fl_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        for i, f1, precision, recall in zip(
            range(len(fl_per_class)),
            fl_per_class,
            precision_per_class,
            recall_per_class,
        ):
            results[f"{prefix}_{i}_f1"] = f1
            results[f"{prefix}_{i}_precision"] = precision
            results[f"{prefix}_{i}_recall"] = recall
        return results
    # compute f1, accraucy, precision, recall for binary case
    return {
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred),
        f"{prefix}_recall": recall_score(y_true, y_pred),
    }

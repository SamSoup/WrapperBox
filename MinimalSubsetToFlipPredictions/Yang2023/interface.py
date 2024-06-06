"""
Interface for calling code from https://arxiv.org/abs/2302.02169
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from .smallest_k import IP
from .recursive import IP_iterative

# the IP function expects the following parameters
# model: LogisticRegression: prefitted logistc regression model
# dataname: str: prefix string for output res
# X: {train: (n_samples, embedding_dim), dev: (n_samples, embedding_dim)}
# y: {train: (n_samples), dev: (n_samples)}
# threshold: typically 0.5, threshold to consider a "flip" occurs
# l2: regularization parameter for LogisticRegression in sklearn
# C (inverse of regularization strength) = 1/l2, so larger l2 = smaller C
# = stronger penality/regularization


def compute_minimal_subset_to_flip_predictions(
    model: LogisticRegression,
    dataset_name: str,
    train_embeddings: np.ndarray,
    eval_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_labels: np.ndarray,
    test_labels: np.ndarray,
    thresh: int = 0.5,
    l2: int = 1,
    output_dir: str = "./results",
    algorithm: str = "fast",  # fast or slow, corresponding to Algo 1 or 2
):
    X, y = {}, {}
    X["train"] = np.vstack([train_embeddings, eval_embeddings])
    X["dev"] = test_embeddings
    y["train"] = np.concatenate([train_labels, eval_labels])
    y["dev"] = test_labels

    # important! The IP function will save to output_dir (./results default)
    # a pickled list of indices (or None) that indicates the minimal set
    # for each prediction
    if algorithm == "fast":
        IP(
            model=model,
            dataname=dataset_name,
            X=X,
            y=y,
            l2=l2,
            thresh=thresh,
            output_dir=output_dir,
        )
    else:
        IP_iterative(
            model=model,
            dataname=dataset_name,
            X=X,
            y=y,
            l2=l2,
            thresh=thresh,
            output_dir=output_dir,
        )

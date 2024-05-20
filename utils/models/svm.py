# Utility functions for svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
from typing import Tuple, Union


def project_inputs(
    inputs: np.ndarray, clf: Union[LinearSVC, SVC]
) -> np.ndarray:
    """
    Project the input embeddings onto their orthongonal point on a
    linear SVM's hyperplane.

    Formula: X_p = X - (w*X+b)/(w*w)*w, where

    w*x+b = signed distance from the point x to the hyperplane
    w*w = ||w||^2, normalizes the distance along the direction of w

    Together, compute the projection vector by moviing from x directly towards
    the hyperplane (defined by the line of w*x+b)

    Args:
        input (np.ndarray): (num_examples, hidden_size)
        w (np.ndarray): weights of shape (hidden_size)
        b (float): constant bias term

    Returns:
        np.ndarray: The projected vectors of shape (num_examples, hidden_size)
    """
    # Assuming you have your trained linear SVM model: svm_model
    # And your input points: x_input (shape = [n_samples, 1024])

    w = clf.coef_[0]  # Weight vector
    b = clf.intercept_[0]  # Bias term

    # Compute the distances from x_input to the hyperplane for all examples
    # Note: np.dot(x_input, w) performs a row-wise dot product if x_input is a
    # matrix and w is a vector
    dist_to_hyperplane = (np.dot(inputs, w) + b) / np.linalg.norm(w)

    # Compute the projection vectors
    # Since distances is a 1D array of shape (n_samples,), we need to make it
    # compatible with the shape of x_input
    # for broadcasting. We reshape distances to (n_samples, 1) and then multiply
    # by w, which broadcasts the operation
    # across all dimensions except the last, aligning with the shape of w.
    x_proj_on_hyperplane = inputs - np.outer(dist_to_hyperplane, w)
    # projection_vectors = (
    #     dist_to_hyperplane[:, np.newaxis] * w / np.linalg.norm(w)
    # )

    # # Compute the points on the hyperplane
    # x_proj_on_hyperplane = inputs - projection_vectors

    return x_proj_on_hyperplane


def get_support_vectors(
    M: Union[LinearSVC, SVC], X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the support vector and its corresponding indices

    Args:
        M (Union[LinearSVC, SVC]): the svm model
        X (np.ndarray): the training features

    Returns:
        Tuple[np.ndarray, np.ndarray]: support vectors of shape
            (n_SV, n_features), and indices of shape (n_SV)
    """

    if isinstance(M, LinearSVC):
        return M.support_vectors_, M.support_

    # for linear svc, the code is a bit more complicated, because
    # sklearn's default linearsvc does not actually retrieve the svs
    # from # https://scikit-learn.org/stable/auto_examples/svm/plot_linearsvc_support_vectors.html
    # obtain the support vectors through the decision function
    decision_function = M.decision_function(X)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # The support vectors are the samples that lie within the margin
    # boundaries, whose size is conventionally constrained to 1
    support_vector_indices = np.unique(
        np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    )
    support_vectors = X[support_vector_indices]
    return support_vectors, support_vector_indices

# Utility functions for svm
from sklearn.svm import LinearSVC
import numpy as np


def project_inputs(inputs: np.ndarray, clf: LinearSVC) -> np.ndarray:
    """
    Project the input embeddings onto their orthongonal point on a
    linear SVM's hyperplane

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
    # Note: np.dot(x_input, w) performs a row-wise dot product if x_input is a matrix and w is a vector
    dist_to_hyperplane = np.abs(np.dot(inputs, w) + b) / np.linalg.norm(w)

    # Compute the projection vectors
    # Since distances is a 1D array of shape (n_samples,), we need to make it compatible with the shape of x_input
    # for broadcasting. We reshape distances to (n_samples, 1) and then multiply by w, which broadcasts the operation
    # across all dimensions except the last, aligning with the shape of w.
    projection_vectors = (
        dist_to_hyperplane[:, np.newaxis] * w / np.linalg.norm(w)
    )

    # Compute the points on the hyperplane
    x_proj_on_hyperplane = inputs - projection_vectors

    return x_proj_on_hyperplane

"""
The simplest Classiifer: simply returns a random label out of all the possible
ones
"""

import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.classes_ = None
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)

    def fit(self, X, y):
        """Fit the classifier on the training data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Store the unique classes from the target values
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict random labels for the given samples.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Samples for which to predict the labels.

        Returns:
        y_pred : array, shape (n_samples,)
            Randomly predicted labels.
        """
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        # Generate random predictions
        n_samples = X.shape[0]
        random_indices = self.rng.integers(
            low=0, high=len(self.classes_), size=n_samples
        )
        return self.classes_[random_indices]

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **parameters):
        self.kwargs = parameters
        return self

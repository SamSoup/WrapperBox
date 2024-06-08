# from lightgbm import LGBMModel
import numpy as np
from sklearn.base import BaseEstimator
from typing import Iterable, Union


# def get_predictions(
#     M: Union[LGBMModel, BaseEstimator], X: np.ndarray
# ) -> Iterable[int]:
#     if isinstance(M, LGBMModel):
#         prediction_probas = M.predict(X)
#         if prediction_probas.ndim < 2:
#             # threshold using 0.5
#             predictions = (prediction_probas >= 0.5).astype(int)
#         else:
#             predictions = np.argmax(prediction_probas, axis=1)
#         return predictions
#     elif isinstance(M, BaseEstimator):
#         return M.predict(X)
#     else:
#         raise ValueError("Unsupported Classifier for predictions")


def get_predictions(M: BaseEstimator, X: np.ndarray) -> Iterable[int]:
    return M.predict(X)

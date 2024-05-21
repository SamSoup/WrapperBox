from typing import Iterable, List, Union
from lightgbm import LGBMModel
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def get_leaf_ids(
    M: Union[LGBMModel, DecisionTreeClassifier],
    X: np.ndarray,
) -> List[int]:
    if isinstance(M, DecisionTreeClassifier):
        return M.apply(X)
    elif isinstance(M, LGBMModel):
        return M.predict(X, pred_leaf=True)
    else:
        raise ValueError(
            "Unsupported Decision Tree Classifier for leaf ID retrieval"
        )


def get_predictions(
    M: Union[LGBMModel, DecisionTreeClassifier], X: np.ndarray
) -> Iterable[int]:
    if isinstance(M, DecisionTreeClassifier):
        return M.predict(X)
    elif isinstance(M, LGBMModel):
        prediction_probas = M.predict(X)
        if prediction_probas.ndim < 2:
            # threshold using 0.5
            predictions = (prediction_probas >= 0.5).astype(int)
        else:
            predictions = np.argmax(prediction_probas, axis=1)
        return predictions
    else:
        raise ValueError("Unsupported Decision Tree Classifier for predictions")

from typing import List, Union
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

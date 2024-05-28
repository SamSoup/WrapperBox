from sklearn.base import BaseEstimator
from typing import Union
from utils.constants.directory import SAVED_MODELS_DIR
import pickle
import os


def load_saved_wrapperbox_model(
    dataset: str,
    model: str,
    seed: Union[str, int],
    pooler: str,
    wrapperbox: str,
) -> BaseEstimator:

    path_to_wrapperbox = os.path.join(
        SAVED_MODELS_DIR,
        dataset,
        f"{model}_seed_{seed}",
        pooler,
        f"{wrapperbox}.pkl",
    )
    with open(path_to_wrapperbox, "rb") as f:
        return pickle.load(f)

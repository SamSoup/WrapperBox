from typing import Union
from utils.constants.directory import EMBEDDINGS_DIR
import os
import numpy as np


def load_saved_embeddings(
    dataset: str,
    model: str,
    seed: Union[str, int],
    split: str,
    pooler: str,
    layer: Union[str, int],
) -> np.ndarray:

    path_to_wrapperbox = os.path.join(
        EMBEDDINGS_DIR,
        dataset,
        f"{model}_seed_{seed}",
        split,
        pooler,
        f"layer_{layer}.csv",
    )
    return np.loadtxt(path_to_wrapperbox, delimiter=",")

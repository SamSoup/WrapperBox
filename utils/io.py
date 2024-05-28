from typing import Union
from huggingface_hub import login
from sklearn.base import BaseEstimator
from datasets import load_dataset
from utils.constants.directory import (
    SAVED_MODELS_DIR,
    EMBEDDINGS_DIR,
    CACHE_DIR,
)
import numpy as np
import os
import json
import pickle
import time
import datasets


def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_json(filename: str):
    with open(filename, "r") as handle:
        data = json.load(handle)

    return data


def load_embeddings(
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


def load_wrapperbox(
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


def load_dataset_from_hf(
    dataset: str, retries: int = 3, delay: int = 5
) -> datasets.DatasetDict:
    # Retrieve the token from the environment variable
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    # Log in to Hugging Face using the token
    login(token=token)

    attempt = 0
    while attempt < retries:
        try:
            # Attempt to load the dataset
            return load_dataset(
                f"Samsoup/{dataset}", cache_dir=CACHE_DIR, use_auth_token=True
            )
        except Exception as e:
            print(
                f"Attempt {attempt + 1} failed with error: \n"
                f"{str(e)}. Retrying in {delay} seconds..."
            )
            attempt += 1
            time.sleep(delay)

    raise Exception(
        "Failed to load dataset after multiple attempts due to 401 error"
    )


def load_labels_at_split(dataset: Union[str, datasets.DatasetDict], split: str):
    if isinstance(dataset, str):
        dataset = load_dataset_from_hf(dataset)
    return np.array(dataset[split]["label"])

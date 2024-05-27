import os
import time
from datasets import load_dataset
from utils.constants.directory import CACHE_DIR
from typing import Union
from huggingface_hub import login
from huggingface_hub.utils._errors import RepositoryNotFoundError
import datasets
import numpy as np


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
            return load_dataset(f"Samsoup/{dataset}", cache_dir=CACHE_DIR)
        except RepositoryNotFoundError as e:
            if "401 Client Error" in str(e):
                print(
                    f"Attempt {attempt + 1} failed with 401 error.\n"
                    "Retrying in {delay} seconds..."
                )
                attempt += 1
                time.sleep(delay)
            else:
                raise e

    raise Exception(
        "Failed to load dataset after multiple attempts due to 401 error"
    )


def load_labels_at_split(dataset: Union[str, datasets.DatasetDict], split: str):
    if isinstance(dataset, str):
        dataset = load_dataset_from_hf(dataset)
    return np.array(dataset[split]["label"])

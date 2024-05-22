from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from datasets import DatasetDict, concatenate_datasets
from typing import List
from tqdm import tqdm
import os
import numpy as np
import json


# Define the generic interface
class FindMinimalSubset(ABC):
    @abstractmethod
    def find_minimal_subset(
        self,
        clf: BaseEstimator,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> List[List[int]]:
        """
        For each test example, find a (minimum) subset of training examples
        s.t. they removal would result in a prediction flip

        Args:
            clf (BaseEstimator): the preciction model
            train_embeddings (np.ndarray): (num_train_examples, hidden_size)
            test_embeddings (np.ndarray): (num_test_examples, hidden_size)
            train_labels (np.ndarray): (num_train_examples)

        Returns:
            List[List[int]]: the indices of training example subset s.t. their
            removal would flip a prediction for a test example (one list per
            test example)
        """
        pass

    def persist_to_disk(
        self,
        dataset: DatasetDict,
        dataset_name: str,
        model_name: str,
        wrapper_name: str,
        minimal_subset_indices: List[List[int]],
        offset: int,
        output_dir: str = "./results",
    ):
        compiled_data = []

        for i, indices in tqdm(
            enumerate(minimal_subset_indices), "persisting to disc"
        ):
            # NOTE: due to the possibly large size of minimal subset,
            # only store the indices and NOT the associated example/label
            # minimal_subset = [
            #     {
            #         "index": idx,
            #         "example": dataset["train"]["text"][idx],
            #         "label": dataset["train"]["label"][idx],
            #     }
            #     for idx in indices
            # ]
            compiled_data.append(
                {
                    "id": i + offset,
                    "text": dataset["test"][i]["text"],
                    "label": dataset["test"][i]["label"],
                    "minimum_subset": indices,
                }
            )

        # Serialize the list of JSON objects to a JSON string
        json_string = json.dumps(compiled_data, indent=2)
        prefix = f"{offset}to{offset + len(minimal_subset_indices)}"
        prefix = f"{prefix}_{dataset_name}_{model_name}_{wrapper_name}"

        # Write the JSON string to a file
        with open(
            os.path.join(
                output_dir,
                f"{prefix}_minimal_subsets.json",
            ),
            "w",
        ) as f:
            f.write(json_string)

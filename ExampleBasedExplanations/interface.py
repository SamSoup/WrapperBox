# interface for finding example-based explanations
# Define the generic interface
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from datasets import DatasetDict
from typing import List
import pandas as pd
import numpy as np
import json
import os


class ExampleBasedExplanation(ABC):
    @abstractmethod
    def get_explanation_indices(
        self,
        M: int,
        clf: BaseEstimator,
        train_embeddings: np.ndarray,
        test_embeddings: np.ndarray,
    ) -> List[List[int]]:
        pass

    def persist_to_disk(
        self,
        dataset: DatasetDict,
        dataset_name: str,
        model_name: str,
        wrapper_name: str,
        predictions: np.ndarray,
        explanation_indices: np.ndarray,
        output_dir: str = "./results",
    ):
        """
        Write the example-based explanations to disk in two ways:

        1. A list of jsons, where each element conforms to
        schema.json, see example.json for what it should look like

        2. A csv file, with columns:
        Test input, Ground Truth, Prediction, and Explanations
        (string), (integer), (integer), List(Tuple(str, int))
        where each explanation is a pair of example text and its label

        *Note it is the responsibility of the caller to ensure that
        the output directory already exists.

        Args:
            dataset (DatasetDict): {
                "train": ["text", "label", ...]
                "test": ["text", "label", ...]
            }
            dataset_name (str): _description_
            model_name (str): _description_
            wrapper_name (str): _description_
            predictions (np.ndarray): (num_test_examples)
                where each element is [0, num_label_classes)
            explanation_indices (np.ndarray): (num_test_examples, M),
                where each element is [0, len(dataset["train"]))
            output_dir (str, optional): _description_. Defaults to "./results".
        """
        compiled_data = []

        # Iterate through each example in the test set
        for i, (text, label) in enumerate(
            zip(dataset["test"]["text"], dataset["test"]["label"])
        ):
            prediction = predictions[i]

            # Fetch explanations using indices for the current test example
            explanations = [
                (
                    dataset["train"]["text"][index],
                    int(dataset["train"]["label"][index]),
                )
                for index in explanation_indices[i]
            ]

            # Compile information for the current test example
            example_data = {
                "Test input": text,
                "Ground Truth": int(label),
                "Prediction": int(prediction),
                "Explanations": explanations,
            }

            compiled_data.append(example_data)

        filename = os.path.join(
            output_dir,
            f"{dataset_name}_{model_name}_{wrapper_name}_explanations",
        )

        # 1. store as list of json
        json_string = json.dumps(compiled_data, indent=2)
        with open(f"{filename}.json", "w") as f:
            f.write(json_string)
        print(f"{filename}.json saved")

        # 2. store as csv
        df = pd.DataFrame(compiled_data)

        df.to_csv(f"{filename}.csv", index=False)
        print(f"{filename}.csv saved")

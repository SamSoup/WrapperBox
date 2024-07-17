"""
Loads a generative model and run generations on the test set.
If classification, further translate the text output into a n-way
decision.

Usage: see python3 main.py --help
"""

import argparse
import json
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from utils.constants.directory import DATA_DIR
from CustomDatasets import TextDataset
from utils.hf import get_model_and_tokenizer
from utils.inference import compute_metrics
from utils.io import mkdir_if_not_exists
from pprint import pprint
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to a prefix (or the raw prefix) to append to each input example.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for DataLoader."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for classification.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for resultant representations.",
    )
    args = parser.parse_args()

    # If config argument is provided, load configuration from JSON file
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        # Overwrite args with config
        for key, value in config.items():
            setattr(args, key, value)

    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args


def get_predictions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
):
    # NOTE: the current new simply code to use is `transformers.pipeline`, but
    # I like this current more manual version better for visibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Create the pipeline
    nlp_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    predictions = []

    # Iterate over the dataset; It is recommended that we iterate directly
    # over the dataset without needing to batch
    print("*** Running Sequence Classification ***")
    for pred in tqdm(nlp_pipeline(dataset)):
        predictions.append(pred)
        print(pred)

    return predictions


def main():
    args = get_args()
    random.seed(args.seed)
    mkdir_if_not_exists(args.output_dir)

    ## Load Dataset, from disk because
    ## datasets might require a GLIBC version higher
    ## than what is available
    if os.path.isfile(args.dataset_name_or_path):
        test_dataset = pd.read_csv(args.dataset_name_or_path)
    else:
        test_dataset = pd.read_csv(
            os.path.join(
                DATA_DIR, "datasets", args.dataset_name_or_path, "test.csv"
            )
        )
        test_dataset = TextDataset(texts=test_dataset["text"].tolist())

    ## Load Prompt pre-fix and update 'text' column to use this, if any
    if args.prompt is not None:
        if os.path.isfile(args.prompt):
            with open(args.prompt, "r") as file:
                prompt = file.read().strip()
        else:
            prompt = args.prompt
        test_dataset = test_dataset.map(
            lambda example: {"text": prompt.format(input=example["text"])}
        )

    ## Load Model
    model, tokenizer = get_model_and_tokenizer(
        args.model_name_or_path, causal_lm=False
    )

    ## Tokenize dataset + dataloader
    # test_dataset = TextDataset(texts=test_dataset["text"])
    # test_dataset = TokenizedDataset(
    #     texts=test_dataset["text"],
    #     labels=test_dataset["label"],
    #     tokenizer=tokenizer,
    #     max_length=1024,
    # )
    # dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    ## Obtain results
    results = get_predictions(
        model=model, tokenizer=tokenizer, dataset=test_dataset
    )

    if "labels" in test_dataset:
        metrics = compute_metrics(
            y_pred=results,
            y_true=test_dataset["labels"],
            is_multiclass=args.num_classes > 2,
        )
        pprint(metrics)

    output_file = os.path.join(args.output_dir, "output.json")
    with open(output_file, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()

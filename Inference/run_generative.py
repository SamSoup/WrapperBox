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
from typing import Callable
import pandas as pd
from tqdm import tqdm
from transformers import (
    pipeline,
)
from torch.utils.data import Dataset
from utils.constants.directory import PROMPTS_DIR, DATA_DIR
from utils.hf import get_model_and_tokenizer
from utils.inference import compute_metrics
from utils.io import mkdir_if_not_exists
from pprint import pprint
import numpy as np


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class SentenceWithPromptDataset(Dataset):
    def __init__(self, sentences, prompt):
        self.sentences = sentences
        self.prompt = prompt

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.prompt.format(input=self.sentences[idx])


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
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
        "--max_new_tokens",
        type=int,
        default=2,
        help="Maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for randomness.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Sample from the best k (number of) tokens. 0 means off (Default: 0, 0 ≤ top_k < 100000).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Sample from the set of tokens with highest probability such that sum of probabilies is higher than p. Lower values focus on the most probable tokens.Higher values sample more low-probability tokens (Default: 0.9, 0 < top_p ≤ 1)",
    )
    parser.add_argument(
        "--is_classification",
        action="store_true",
        help="If set, process the output as classification.",
    )
    parser.add_argument(
        "--load_half_precision",
        type=bool,
        default=False,
        help="If true, load the model in half precision",
    )
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


def generate_responses(
    pipeline: Callable, dataset: Dataset, args: argparse.Namespace
):
    results = []
    # Iterate over the dataset; It is recommended that we iterate directly
    # over the dataset without needing to batch
    print("*** Running Sequence Classification ***")
    for output in tqdm(
        pipeline(
            dataset,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            num_return_sequences=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_full_text=False,  # only return added text
        )
    ):
        generated_text = output[0]["generated_text"].strip()
        # extract a label if is classification
        if args.is_classification:
            pred = extract_classification_output(
                generated_text, args.num_classes
            )
            results.append(pred)
        else:
            results.append(generated_text)
    return results


def extract_classification_output(output, num_of_classes):
    try:
        decision = int(output.strip()[0])
        if 0 <= decision < num_of_classes:
            return decision
    except (ValueError, IndexError):
        print(f"An decision was not clear for:\n{output}")

    return random.randint(1, num_of_classes - 1)


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
    labels = test_dataset["label"] if "label" in test_dataset else None
    texts = test_dataset["text"].tolist()

    ## Load Prompt pre-fix and update 'text' column to use this
    if args.prompt is not None:
        if not os.path.isfile(args.prompt):
            args.prompt = os.path.join(PROMPTS_DIR, args.prompt)
        with open(args.prompt, "r") as file:
            prompt = file.read().strip()
        test_dataset = SentenceWithPromptDataset(texts, prompt)
    else:
        test_dataset = SentenceDataset(texts)

    ### Log the first input to check format
    print("First input:\n", texts[0])

    ## Load Model
    model, tokenizer = get_model_and_tokenizer(
        args.model_name_or_path,
        load_half_precison=args.load_half_precision,
        causal_lm=True,
    )
    model.eval()

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    results = generate_responses(text_generator, test_dataset, args)

    output_file = os.path.join(args.output_dir, "output.json")
    with open(output_file, "w") as file:
        json.dump(results, file)

    ## Optionally, if has test labels, compute some metrics
    if labels is not None:
        metrics = compute_metrics(
            y_pred=results,
            y_true=labels,
            is_multiclass=np.unique(labels).size > 2,
            prefix="test",
        )
        pprint(metrics)


if __name__ == "__main__":
    main()

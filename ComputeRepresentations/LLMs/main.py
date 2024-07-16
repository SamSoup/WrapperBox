"""
Command line runner file to compute representations.

Usage: python3 main.py --help
"""

import argparse
import json
import os
import numpy as np
from datasets import load_dataset
from ComputeRepresentations.LLMs.ModelForSentenceLevelRepresentation import (
    ModelForSentenceLevelRepresentation,
)
from utils.constants.directory import CACHE_DIR, PROMPTS_DIR
from utils.io import mkdir_if_not_exists
from datasets import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Specify a path or a hub location to a Dataset",
    )
    parser.add_argument(
        "--do_train",
        type=bool,
        default=True,
        help="Compute for train dataset?",
    )
    parser.add_argument(
        "--do_test",
        type=bool,
        default=True,
        help="Compute for test dataset?",
    )
    parser.add_argument(
        "--do_eval",
        type=bool,
        default=True,
        help="Compute for eval dataset?",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Specify a path or a hub location to a HF model.",
    )
    parser.add_argument(
        "--load_half_precision",
        type=bool,
        help="Should we load the model in half precision to save mem?",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        help="Name of pooler functions to sue. See `EmbeddingPooler` for options.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Max length of input tokens.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to a prefix (or the raw prefix) to append to each input example.",
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


def load_prompt_with_dataset(prompt_fname: str, dataset: Dataset):
    # Load Prompt pre-fix and update 'text' column to use this
    if os.path.isfile(prompt_fname):
        prompt_path = prompt_fname
    else:
        # just a file name, combine with known location
        prompt_path = os.path.join(PROMPTS_DIR, prompt_fname)
    with open(prompt_path, "r") as file:
        prompt = file.read().strip()
    # NOTE: Assume 'text' is the column of inputs to compute reps
    dataset = dataset.map(
        lambda example: {"text": prompt.format(input=example["text"])}
    )

    return dataset


if __name__ == "__main__":
    args = get_args()
    to_dos = {
        "train": args.do_train,
        "eval": args.do_eval,
        "test": args.do_test,
    }
    mkdir_if_not_exists(args.output_dir)
    datasets = load_dataset(
        args.dataset_name_or_path, use_auth_token=True, cache_dir=CACHE_DIR
    )
    model = ModelForSentenceLevelRepresentation(
        args.model_name_or_path, args.pooler, args.load_half_precision
    )
    for split, dataset in datasets.items():
        if to_dos[split]:
            print(f"***** Computing Representations for {split} dataset *****")

            if args.prompt_path is not None:
                dataset = load_prompt_with_dataset(
                    prompt_fname=args.prompt_path, dataset=dataset
                )

            # NOTE: Assume 'text' is the column of inputs to compute reps
            representations = model.extract_representations(
                texts=dataset["text"],
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            save_path = os.path.join(args.output_dir, f"{split}.npy")
            # Save representations as numpy file
            np.save(save_path, representations.numpy())

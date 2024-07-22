"""
Command line runner file to compute representations.

Usage: python3 main.py --help
"""

import argparse
import json
import os
from typing import Iterable
import pandas as pd
import numpy as np
from ComputeRepresentations.LLMs.ModelForSentenceLevelRepresentation import (
    ModelForSentenceLevelRepresentation,
)
from utils.constants.directory import DATA_DIR, PROMPTS_DIR
from utils.io import mkdir_if_not_exists


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
        "--is_causalLM",
        type=bool,
        default=True,
        help="Is the model loaded a CausalLM or SequenceClassification model?",
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
        "--chunk_size",
        type=int,
        default=100000,
        help="If dataset exceeds this size, do chunks/batches to save memory.",
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


def format_texts_with_prompt(
    prompt_fname: str, texts: Iterable[str]
) -> Iterable[str]:
    if os.path.isfile(prompt_fname):
        prompt_path = prompt_fname
    else:
        # just a file name, combine with known location
        prompt_path = os.path.join(PROMPTS_DIR, prompt_fname)
    with open(prompt_path, "r") as file:
        prompt = file.read().strip()

    texts = [prompt.format(input=t) for t in texts]
    return texts


if __name__ == "__main__":
    args = get_args()
    to_dos = {
        "train": args.do_train,
        "eval": args.do_eval,
        "test": args.do_test,
    }
    mkdir_if_not_exists(args.output_dir)
    # Load Datasets, from disk because
    ## datasets might require a GLIBC version higher than what is available
    datasets = {}
    for split in to_dos:
        if to_dos[split]:
            datasets[split] = pd.read_csv(
                os.path.join(
                    DATA_DIR,
                    "datasets",
                    args.dataset_name_or_path,
                    f"{split}.csv",
                )
            )

    # NOTE: assume we are loading a casualLM
    model = ModelForSentenceLevelRepresentation(
        args.model_name_or_path,
        args.pooler,
        args.load_half_precision,
        args.is_causalLM,
    )
    for split, dataset in datasets.items():
        ## Print some diagnostics
        texts = dataset["text"].tolist()
        max_length_in_dataset = max([len(s) for s in texts])
        print(
            f"The maximum sequence length in the dataset is {max_length_in_dataset}."
            f"The current set maximum sequence length is {args.max_length}"
        )
        if args.max_length < max_length_in_dataset:
            print("WARNING: Sequence truncation will occur")

        if to_dos[split]:
            print(f"***** Computing Representations for {split} dataset *****")

            if args.prompt_path is not None:
                texts = format_texts_with_prompt(
                    prompt_fname=args.prompt_path, texts=texts
                )

            chunk_size = (
                args.chunk_size if len(texts) > args.chunk_size else len(texts)
            )
            num_chunks = (len(texts) // chunk_size) + 1

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(texts))
                texts_chunk = texts[start_idx:end_idx]

                print(
                    f"Processing chunk {i+1}/{num_chunks}, size: {end_idx-start_idx+1}"
                )

                # NOTE: Assume 'text' is the column of inputs to compute reps
                representations = model.extract_representations(
                    texts=texts_chunk,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )

                save_path = os.path.join(
                    args.output_dir,
                    (
                        f"{split}_chunk_{i+1}.npy"
                        if len(texts) > args.chunk_size
                        else f"{split}.npy"
                    ),
                )
                # Save representations as numpy file
                np.save(save_path, representations.numpy())

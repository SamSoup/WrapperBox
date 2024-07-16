"""
Command line runner file to compute representations.

Usage: python3 main.py --help
"""

import argparse
import json
import os
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from utils.constants.directory import CACHE_DIR
from utils.io import mkdir_if_not_exists
from torch.utils.data import DataLoader, Dataset


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


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
        "--batch_size",
        type=int,
        help="Batch size.",
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
    model = SentenceTransformer(args.model_name_or_path, cache_folder=CACHE_DIR)
    for split, dataset in datasets.items():
        if to_dos[split]:
            print(f"***** Computing Representations for {split} dataset *****")
            # Assume that the column to compute representation is for is 'text'
            dataset = SentenceDataset(dataset["text"])
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False
            )

            embeddings = []
            for batch in dataloader:
                embeddings = model.encode(batch, convert_to_numpy=True)
                embeddings.append(embeddings)

            representations = np.vstack(embeddings)
            save_path = os.path.join(args.output_dir, f"{split}.npy")
            # Save representations as numpy file
            np.save(save_path, representations)

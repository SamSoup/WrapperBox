"""
Command line runner file to compute representations.

Usage: python3 main.py --help
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from utils.constants.directory import CACHE_DIR, DATA_DIR
from utils.io import mkdir_if_not_exists
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class InstructionSentenceDataset(Dataset):
    def __init__(self, sentences, instruction):
        self.sentences = sentences
        self.instruction = instruction

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return f"Instruct: {self.instruction}\nText: {self.sentences[idx]}"


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
        "--instruction",
        type=str,
        default=None,
        help="The instruction to append to the input texts",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size.",
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
        datasets[split] = pd.read_csv(
            os.path.join(
                DATA_DIR, "datasets", args.dataset_name_or_path, f"{split}.csv"
            )
        )

    # Load the model, wrapping it for multi-gpu if needed
    model = SentenceTransformer(args.model_name_or_path, cache_folder=CACHE_DIR)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model = model.module  # call on the inner module later

    for split, dataset in datasets.items():
        print(f"***** Computing Representations for {split} dataset *****")
        # Assume that the column to compute representation is for is 'text'
        if args.instruction is not None:
            dataset = InstructionSentenceDataset(
                dataset["text"], args.instruction
            )
        else:
            dataset = SentenceDataset(dataset["text"])
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )

        chunk_size = (
            args.chunk_size if len(dataset) > args.chunk_size else len(dataset)
        )
        num_chunks = (len(dataset) // chunk_size) + 1

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(dataset))
            dataset_chunk = dataset.select(range(start_idx, end_idx))

            print(
                f"Processing chunk {i+1}/{num_chunks} with indices {start_idx} to {end_idx}"
            )

            dataloader = DataLoader(
                dataset_chunk, batch_size=args.batch_size, shuffle=False
            )

            representations = []
            for batch in tqdm(dataloader):
                embeddings = model.encode(batch, convert_to_numpy=True)
                representations.append(embeddings)

            representations = np.vstack(representations)
            save_path = os.path.join(
                args.output_dir,
                (
                    f"{split}_chunk_{i+1}.npy"
                    if len(dataset) > args.chunk_size
                    else f"{split}.npy"
                ),
            )
            # Save representations as numpy file
            np.save(save_path, representations)

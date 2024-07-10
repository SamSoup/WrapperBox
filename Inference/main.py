"""
Loads a generative model and run generations on the test set.
If classification, further translate the text output into a n-way
decision.

Usage: see python3 main.py --help
"""

import argparse
import json
import pickle
import re
import os
import random
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from utils.constants.directory import CACHE_DIR
from utils.datasets import EmbeddingDataset
from utils.hf import get_model_and_tokenizer
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


def prep_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, batch_size: int
) -> DataLoader:
    embed_dataset = EmbeddingDataset(
        texts=dataset["text"], tokenizer=tokenizer, max_length=1024
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return DataLoader(
        embed_dataset, batch_size=batch_size, collate_fn=data_collator
    )


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    args: argparse.Namespace,
):
    # NOTE: the current new simply code to use is `transformers.pipeline`, but
    # I like this current more manual version better for visibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["input_ids"].to(device)
            input_lengths = (batch["input_ids"] != tokenizer.pad_token_id).sum(
                dim=1
            )
            print(batch.keys())
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                num_return_sequences=1,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            decoded_outputs = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            # pprint(decoded_outputs)
            # input()

            if args.is_classification:
                predictions = extract_classification_output(
                    decoded_outputs, args.num_classes, input_lengths
                )
                results.extend(predictions)
            else:
                results.extend(decoded_outputs)

    return results


def extract_classification_output(
    decoded_outputs, num_of_classes, input_lengths
):
    predictions = []
    for outputs, l in zip(decoded_outputs, input_lengths):
        # Only check the real output components with all white space removed
        output = re.sub(r"\s+", "", outputs[l:])
        # Simplistic extraction assuming the first token is the answer
        try:
            decision = int(output.strip()[0])
            if 0 <= decision < num_of_classes:
                predictions.append(decision)
        except (ValueError, IndexError):
            print(
                f"For response: \n\n{outputs}\n\n ;An decision was not clear"
                f" in segmented output {output}"
            )
        predictions.append(random.randint(1, num_of_classes - 1))

    return predictions


def main():
    args = get_args()
    random.seed(args.seed)
    mkdir_if_not_exists(args.output_dir)

    ## Load Dataset
    datasets = load_dataset(
        args.dataset_name_or_path, use_auth_token=True, cache_dir=CACHE_DIR
    )
    test_dataset = datasets["test"]

    ## Load Prompt pre-fix and update 'text' column to use this
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
        args.model_name_or_path, causal_lm=True
    )

    dataloader = prep_dataset(test_dataset, tokenizer, args.batch_size)

    results = generate_responses(model, tokenizer, dataloader, args)

    output_file = os.path.join(args.output_dir, "output.json")
    with open(output_file, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()

"""
Loads a generative model and run generations on the test set.
If classification, further translate the text output into a n-way
decision.

Usage: see python3 main.py --help
"""

import argparse
import json
import pickle
import os
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


def get_args():
    parser = argparse.ArgumentParser()
    # Either give json, or provide all others
    parser.add_argument(
        "--config", type=str, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--prompt_prefix",
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
    parser.add_argument(
        "--is_classification",
        action="store_true",
        help="If set, process the output as classification.",
    )
    parser.add_argument(
        "--num_of_classes",
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
        texts=dataset["text"], tokenizer=tokenizer, max_length=512
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
    prompt_prefix: str,
    is_classification: bool,
    num_of_classes: int,
):
    results = []

    for batch in dataloader:
        inputs = batch["input_ids"]
        for input_ids in inputs:
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            prompt = prompt_prefix.format(input=input_text)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=2 if is_classification else 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=2 if is_classification else 50,
                    top_p=0.95,
                )

            generated_text = tokenizer.decode(
                output[0], skip_special_tokens=True
            )

            if is_classification:
                classification_output = extract_classification_output(
                    generated_text, num_of_classes
                )
                results.append(classification_output)
            else:
                results.append(generated_text)

    return results


def extract_classification_output(generated_text, num_of_classes):
    # Simplistic extraction assuming the first token is the answer
    try:
        decision = int(generated_text.strip()[0])
        if 0 <= decision < num_of_classes:
            return decision
    except (ValueError, IndexError):
        pass
    return None  # Handle unexpected output appropriately


def main():
    args = get_args()
    mkdir_if_not_exists(args.output_dir)

    ## Load Dataset
    datasets = load_dataset(
        args.dataset_name_or_path, use_auth_token=True, cache_dir=CACHE_DIR
    )

    ## Load Model
    model, tokenizer = get_model_and_tokenizer(
        args.model_name_or_path, causal_lm=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader = prep_dataset(datasets["test"], tokenizer, args.batch_size)

    ## Load Prompt pre-fix and generate
    if os.path.isfile(args.prompt_prefix):
        with open(args.prompt_prefix, "r") as file:
            prompt_prefix = file.read().strip()
    else:
        prompt_prefix = args.prompt_prefix

    results = generate_responses(
        model,
        tokenizer,
        dataloader,
        prompt_prefix,
        args.is_classification,
        args.num_of_classes,
    )

    output_file = os.path.join(args.output_dir, "output.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

from typing import Any, Callable, Dict, List
from torch import nn
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
)
from sentence_transformers import SentenceTransformer
from FineTune import (
    get_loss_function,
    load_dataset_from_config,
    startTrainer,
)
from models import (
    ClassificationHead,
    SentenceTransformerForClassification,
    SentenceTransformerDataset,
    SentenceTransformerCollator,
)
from utils.constants.directory import CACHE_DIR
from utils.io import load_yaml
import wandb
import torch
import sys


def load_model(model_config: Dict[str, Any], loss_fct: Callable) -> nn.Module:
    # Initializing a new SentenceTransformerForClassification
    encoder = SentenceTransformer(
        model_name_or_path=model_config["name"], cache_folder=CACHE_DIR
    )
    cls_head = ClassificationHead(
        input_dim=encoder.get_sentence_embedding_dimension(),
        num_classes=model_config["head_params"]["output_dim"],
        dropout_rate=model_config["head_params"]["dropout_rate"],
    )
    model = SentenceTransformerForClassification(
        encoder=encoder,
        classification_head=cls_head,
        batch_size=model_config["encoder_batch_size"],
        loss_fct=loss_fct,
    )

    return model


def main():
    # Load configurations
    config_fname = sys.argv[1]
    config = load_yaml(config_fname)

    # Parse Training Arguments
    training_args = TrainingArguments(**config["training_args"])
    if training_args.report_to == "wandb":
        wandb.init(project=config["project_name"])

    # Load model and tokenizer
    ## Custom loss function
    loss_function = get_loss_function(
        config["loss_function"], device=training_args.device
    )
    model: SentenceTransformerForClassification = load_model(
        model_config=config["model"], loss_fct=loss_function
    )
    tokenizer = model.encoder.tokenizer

    # Load dataset (with prompts)
    train_dataset, eval_dataset, test_dataset = load_dataset_from_config(
        dataset_config=config["dataset"], prompt_config=config["prompt"]
    )

    data_collator = SentenceTransformerCollator()

    ## wrap the datasets
    train_dataset = SentenceTransformerDataset(
        texts=train_dataset["text"],
        labels=train_dataset["label"],
    )
    eval_dataset = SentenceTransformerDataset(
        texts=eval_dataset["text"],
        labels=eval_dataset["label"],
    )
    test_dataset = SentenceTransformerDataset(
        texts=test_dataset["text"],
        labels=test_dataset["label"],
    )

    # Start the Trainer!
    startTrainer(
        config=config,
        training_args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        predict_dataset=test_dataset,
        data_collator=data_collator,
        loss_fct=loss_function,
    )

    if training_args.report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()

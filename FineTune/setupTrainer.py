# Adapted from https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
from typing import Any, Callable, Dict, Tuple, Union
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import PredictionOutput
from transformers.data.data_collator import DataCollator
from torch import nn
from datasets import Dataset, load_dataset
from FineTune import CustomLossTrainer
from utils.constants.directory import CACHE_DIR
import numpy as np
import logging
import datasets
import transformers
import sys
import evaluate
import torch
import random
import os
import json

from utils.hf import set_seed_for_reproducability
from utils.inference import randargmax

logger: logging.Logger = logging.getLogger(__name__)


def init_logging(training_args: TrainingArguments):
    """
    Logging set ups

    Args:
        training_args (TrainingArguments)
    """

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training Arguments:\n")
    logger.info(training_args.to_json_string())


def load_dataset_with_prompt(
    dataset_config: Dict[str, Any],
    prompt_config: Dict[str, Any],
    dataset: Dataset,
):
    template: str = prompt_config["template"]
    input_column = dataset_config["input_column"]
    params = prompt_config["params"]

    dataset = dataset.map(
        lambda example: {
            input_column: template.format(**params, input=example[input_column])
        }
    )

    return dataset


# Avoid name clash with load_dataset from datasets
def load_dataset_from_config(
    dataset_config: Dict[str, Any], prompt_config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    # Preparing the dataset
    # NOTE: this assumes that the online repository contains
    # the train, eval, and test splits already. Otherwise, one
    # can write their own dataset loading codes
    datasets = load_dataset(
        dataset_config["name"], use_auth_token=True, cache_dir=CACHE_DIR
    )
    for split, dataset in datasets.items():
        if prompt_config["use_prompt"]:
            dataset = load_dataset_with_prompt(
                dataset_config=dataset_config,
                prompt_config=prompt_config,
                dataset=dataset,
            )
            datasets[split] = dataset
    return (datasets["train"], datasets["eval"], datasets["test"])


def get_loss_function(
    loss_config: Dict[str, Any], device: Union[str, torch.device]
):
    loss_function_type = loss_config["type"]
    loss_function_params: Dict[str, Any] = loss_config["params"]
    # convert all lists to tensors
    for k, v in loss_function_params.items():
        if isinstance(v, list):
            loss_function_params[k] = torch.tensor(v, device=device)

    if loss_function_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**loss_function_params)
    elif loss_function_type == "MSELoss":
        return nn.MSELoss(**loss_function_params)
    elif loss_function_type == "BCELoss":
        return nn.BCELoss(**loss_function_params)
    elif loss_function_type == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**loss_function_params)
    elif loss_function_type == "NLLLoss":
        return nn.NLLLoss(**loss_function_params)
    else:
        raise ValueError(
            f"Unsupported loss function type: {loss_function_type}"
        )


# NOTE: this function assumes that the configurations, necessary models,
# and datasets is already loaded (and subsetted if needed) elsewhere.
def startTrainer(
    config: Dict[str, Any],
    training_args: TrainingArguments,
    model: nn.Module,
    tokenizer: AutoTokenizer = None,
    train_dataset: Dataset = None,
    eval_dataset: Dataset = None,
    predict_dataset: Dataset = None,
    data_collator: DataCollator = None,  # type: ignore
    loss_fct: Callable = None,
):
    init_logging(training_args)
    set_seed_for_reproducability(training_args.seed)

    # Load evaluation metrics
    metrics_config = config["metrics"]
    metric_fcts = {m: evaluate.load(m) for m in metrics_config["scores"]}

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        labels = p.label_ids
        preds = (
            randargmax(p.predictions)
            if p.predictions.ndim > 1
            else (p.predictions > 0.5).astype(int)
        )

        # For binary classification, preds should be 1D array
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.squeeze()

        # Dictionary to store computed metrics
        computed_metrics = {}

        for metric_name, fct in metric_fcts.items():
            if metric_name == "accuracy":
                # Accuracy does not take average_type
                computed_metrics[metric_name] = fct.compute(
                    predictions=preds, references=labels
                )[metric_name]
            else:
                computed_metrics[metric_name] = fct.compute(
                    predictions=preds,
                    references=labels,
                    average=metrics_config["average"],
                )[metric_name]

        return computed_metrics

    # Initialize our Trainer
    trainer: Trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_fct=loss_fct,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model(training_args.output_dir)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logger.info("*** Predict ***")

        pred_output: PredictionOutput = trainer.predict(predict_dataset)
        predictions = pred_output.predictions
        if pred_output.metrics is not None:
            metrics = pred_output.metrics
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        # Save the test predictions
        predictions = randargmax(predictions).tolist()
        logger.info(f"Some predictions: {predictions[:10]}")

        if trainer.is_world_process_zero():
            logger.info(
                f"*** Saved Predictions to {training_args.output_dir} ***"
            )
            output_fname = os.path.join(
                training_args.output_dir, "test_predictions.json"
            )
            with open(output_fname, "w") as writer:
                json.dump(predictions, writer)

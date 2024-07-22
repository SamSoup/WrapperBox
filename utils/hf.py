from typing import Tuple
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from utils.constants.directory import CACHE_DIR
import torch
import torch.distributed as dist
from accelerate import Accelerator
import os
import random
import numpy as np


def set_seed_for_reproducability(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_model_and_tokenizer(
    model_name: str,
    load_half_precison: bool = False,
    causal_lm: bool = False,
    distributed: bool = False,
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads the model and tokenizer based on the provided model name. Assumes
    that the models can be publicly accessed. Utilizes multiple GPUs if available.

    Args:
        model_name (str): Name of the model to load.
        load_half_precison (bool): Whether to load model in half precision.
        causal_lm (bool): Whether to load a ModelForCausalLM. Default is False.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer.
    """
    print(f"*** Caching to {CACHE_DIR} ***")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    MODEL_CLASS = (
        AutoModelForCausalLM
        if causal_lm
        else AutoModelForSequenceClassification
    )
    ## Use device_map = "auto" for automatic multi-gpu support
    model = MODEL_CLASS.from_pretrained(
        model_name, cache_dir=CACHE_DIR, device_map="auto"
    )
    # model = MODEL_CLASS.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=CACHE_DIR)

    ## Change to half precision, if specified
    if load_half_precison:
        model = model.half()
        if "gemma-2" in model_name:
            model = model.bfloat16()
        print(f"*** Model Loaded in Half Precision ***")

    # Set pad token if not already, assu
    if tokenizer.pad_token is None:
        # NOTE: for llama3, use a special pad token
        if "Llama-3" in model_name:
            tokenizer.pad_token = "<|end_of_text|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.pad_token
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
        if (
            hasattr(model, "generation_config")
            and model.generation_config is not None
        ):
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"eos_token_id: {tokenizer.eos_token}")
    print(f"eos_token: {tokenizer.eos_token_id}")
    print(f"pad_token_id: {tokenizer.pad_token}")
    print(f"pad_token: {tokenizer.pad_token_id}")

    # Check if the model is decoder-only
    print(f"*** Model configurations ***")
    print(model.config)
    print(model.generation_config)
    if (
        model.config.is_decoder
        or "Llama-3" in model_name
        or "Llama-2" in model_name
    ):
        print("Model is a decoder-only architecture.")
        tokenizer.padding_side = "left"
    else:
        print("Model is not a decoder-only architecture.")

    return model, tokenizer

from typing import Tuple
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    set_seed,
)
from utils.constants.directory import CACHE_DIR
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


def init_distributed():
    """
    Initialize the distributed environment if multiple GPUs are available.
    """
    if torch.cuda.device_count() > 1:
        # Check and set necessary environment variables
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"  # or any other free port

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


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

    if distributed:
        init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

    dtype = torch.float16 if load_half_precison else torch.float
    MODEL_CLASS = (
        AutoModelForCausalLM
        if causal_lm
        else AutoModelForSequenceClassification
    )
    model = MODEL_CLASS.from_pretrained(
        model_name, torch_dtype=dtype, cache_dir=CACHE_DIR
    )

    # Set pad token if not already, assu
    if tokenizer.pad_token is None:
        # NOTE: for llama3, use a special pad token
        if isinstance(model.config, LlamaConfig):
            tokenizer.pad_token = "<|end_of_text|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.pad_token
            )
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
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
    if model.config.is_decoder or isinstance(model.config, LlamaConfig):
        print("Model is a decoder-only architecture.")
        tokenizer.padding_side = "left"
    else:
        print("Model is not a decoder-only architecture.")

    device = torch.device(
        f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
        if torch.cuda.device_count() > 1 and distributed
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    if torch.cuda.device_count() > 1 and distributed:
        model = DDP(model, device_ids=[device])

    return model, tokenizer

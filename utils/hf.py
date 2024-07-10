from typing import Tuple
from transformers import AutoTokenizer, AutoModel
from utils.constants.directory import CACHE_DIR
import torch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
import os

CACHE_DIR = "path/to/cache"  # Specify your cache directory here


def init_distributed():
    """
    Initialize the distributed environment if multiple GPUs are available.
    """
    if torch.cuda.device_count() > 1:
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def get_model_and_tokenizer(
    model_name: str, load_half_precison: bool = False
) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads the model and tokenizer based on the provided model name. Assumes
    that the models can be publicly accessed. Utilizes multiple GPUs if available.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer.
    """
    print(f"*** Caching to {CACHE_DIR} ***")
    init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    # Set pad token if not already
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_half_precison:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16, cache_dir=CACHE_DIR
        )
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
    device = torch.device(
        f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
        if torch.cuda.device_count() > 1
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[device])

    return model, tokenizer

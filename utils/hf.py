from typing import Tuple
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from utils.constants.directory import CACHE_DIR
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


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
    model_name: str, load_half_precison: bool = False, causal_lm: bool = False
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
    init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    # Set pad token if not already
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if load_half_precison else torch.float
    MODEL_CLASS = AutoModelForCausalLM if causal_lm else AutoModel
    model = MODEL_CLASS.from_pretrained(
        model_name, torch_dtype=dtype, cache_dir=CACHE_DIR
    )
    device = torch.device(
        f"cuda:{dist.get_rank() % torch.cuda.device_count()}"
        if torch.cuda.device_count() > 1
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[device])

    return model, tokenizer

from typing import Tuple
from transformers import AutoTokenizer, AutoModel
from utils.constants.directory import CACHE_DIR
import torch


def get_model_and_tokenizer(model_name: str) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads the model and tokenizer based on the provided model name. Assumes
    that the models can be publicly accessed.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        Tuple[AutoModel, AutoTokenizer]: Loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model, tokenizer

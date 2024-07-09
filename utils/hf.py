from typing import Tuple
from transformers import AutoTokenizer, AutoModel
from utils.constants.directory import CACHE_DIR


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
    return model, tokenizer

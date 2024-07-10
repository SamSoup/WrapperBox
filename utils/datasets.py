import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict


class EmbeddingDataset(Dataset):
    def __init__(
        self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512
    ):
        """
        Initializes the EmbeddingDataset with texts, tokenizer, and max_length.

        Args:
            texts (List[str]): List of input texts.
            tokenizer (AutoTokenizer): Tokenizer to convert texts to token IDs.
            max_length (int): Maximum length of tokenized input sequences.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Returns the number of texts in the dataset.

        Returns:
            int: Number of texts.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns the tokenized representation of a text at the given index.

        Args:
            idx (int): Index of the text to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Tokenized representation of the text.
        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            **{key: val.squeeze(0) for key, val in encoding.items()},
        }

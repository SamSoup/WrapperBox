from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class TokenizedDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        """
        Initializes the TokenizedDataset with texts, labels, tokenizer, and max_length.

        Args:
            texts (List[str]): List of input texts.
            labels (List[int]): List of labels corresponding to the input texts.
            tokenizer (AutoTokenizer): Tokenizer to convert texts to token IDs.
            max_length (int): Maximum length of tokenized input sequences.
        """
        self.texts = texts
        self.labels = labels
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
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(label)
        return encoding

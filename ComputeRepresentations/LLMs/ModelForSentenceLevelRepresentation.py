import torch
from torch.utils.data import DataLoader
from typing import List, Iterable
from ComputeRepresentations.LLMs.EmbeddingPooler import EmbeddingPooler
from utils.datasets import EmbeddingDataset
from utils.hf import get_model_and_tokenizer
from tqdm import tqdm


class ModelForSentenceLevelRepresentation:
    def __init__(
        self, model_name: str, pooler: str, load_half_precision: bool = False
    ):
        """
        Initializes the model and tokenizer for sentence-level representation extraction.

        Args:
            model_name (str): Name of the model to load.
            pooler (str): Name of the pooling function to use.
        """
        torch.cuda.empty_cache()
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.tokenizer = get_model_and_tokenizer(
            model_name, load_half_precision
        )
        self.model.to(self.device)
        self.model.eval()
        self.pooler_name = pooler
        self.pooler = EmbeddingPooler().get(pooler)

    def create_dataloader(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = False,
    ) -> DataLoader:
        """
        Creates a DataLoader from the given texts.

        Args:
            texts (List[str]): List of input texts.
            batch_size (int): Number of samples per batch.
            max_length (int): Maximum length of tokenized input sequences.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        dataset = EmbeddingDataset(texts, self.tokenizer, max_length)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def extract_representations(
        self,
        texts: Iterable[str],
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = False,
    ) -> torch.Tensor:
        """
        Extracts and averages the hidden states of the final layer from the model.

        Args:
            texts (Iterable[str]): Iterable of input texts to process.
            batch_size (int): Batch size for DataLoader.
            max_length (int): Maximum length of tokenized input sequences.
            shuffle (bool): Whether to shuffle the data in DataLoader.

        Returns:
            torch.Tensor: Averaged representations of the input texts.
        """
        dataloader = self.create_dataloader(
            list(texts), batch_size, max_length, shuffle
        )

        representations = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                torch.cuda.empty_cache()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                output_attentions = (
                    True
                    if self.pooler_name == "mean_with_attention_heads"
                    else False
                )
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    # (num_layers, batch_size, num_heads, sequence_length, sequence_length)
                    attention_mask = outputs.attentions[-1]
                elif self.pooler_name == "last":
                    # the mask here is used to check the last token that is not a pad
                    attention_mask = (
                        input_ids != self.tokenizer.pad_token_id
                    ).long()
                pooled_representation = self.pooler(
                    outputs.last_hidden_state.cpu(), attention_mask.cpu()
                )
                representations.append(pooled_representation)
        return torch.cat(representations)

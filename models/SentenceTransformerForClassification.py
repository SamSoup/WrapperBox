import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from typing import List, Callable


class ClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, num_classes: int, dropout_rate: float = 0.1
    ):
        super(ClassificationHead, self).__init__()
        # NOTE: disbale dropout with 0
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, embeddings):
        # batch_size, hidden_size -> linear
        embeddings = self.dropout(embeddings)
        output = self.classifier(embeddings)
        return output


class SentenceTransformerForClassification(nn.Module):
    def __init__(
        self,
        encoder: SentenceTransformer,
        classification_head: ClassificationHead,
        batch_size: int = 32,
        loss_fct: Callable = nn.CrossEntropyLoss(),
    ):
        super(SentenceTransformerForClassification, self).__init__()
        self.encoder = encoder
        self.classification_head = classification_head
        self.batch_size = batch_size
        self.loss_fct = loss_fct

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, sentences, labels=None):
        embeddings = self.encoder.encode(
            sentences, batch_size=self.batch_size, convert_to_tensor=True
        )
        logits = self.classification_head(embeddings)

        if labels is not None:
            loss = self.loss_fct(
                logits.view(
                    -1, self.classification_head.classifier.out_features
                ),
                labels.view(-1),
            )
            return {"logits": logits, "loss": loss}
        return {"logits": logits}


# For fine-tuning
class SentenceTransformerDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        """
        Initializes the SentenceTransformerDataset with texts and labels.

        Args:
            texts (List[str]): List of input texts.
            labels (List[int]): List of labels corresponding to the input texts.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the number of texts in the dataset.

        Returns:
            int: Number of texts.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns the text and label at the given index.

        Args:
            idx (int): Index of the text to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Text and label at the given index.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        return {"text": text, "labels": torch.tensor(label, dtype=torch.long)}


from typing import List, Dict


class SentenceTransformerCollator:
    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        sentences = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        return {"sentences": sentences, "labels": labels}

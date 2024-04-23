from ExampleBasedExplanations.knn import KNNExampleBasedExplanation
from data.models import load_saved_wrapperbox_model
from datasets import Dataset, DatasetDict
from utils.constants.directory import RESULTS_DIR
from utils.io import mkdir_if_not_exists
import numpy as np
import pandas as pd

# Load datasets from Parquet files
train_dataset = Dataset.from_pandas(pd.read_parquet("train.parquet"))
valid_dataset = Dataset.from_pandas(pd.read_parquet("valid.parquet"))
test_dataset = Dataset.from_pandas(pd.read_parquet("test.parquet"))
dataset_dict = DatasetDict(
    {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
)

# Load Models
knn_clf = load_saved_wrapperbox_model(
    dataset="toxigen",
    model="deberta-large",
    seed=42,
    pooler="mean_with_attention",
    wrapperbox="KNN",
)

# Load Embeddings

output_dir = RESULTS_DIR / "Explanations"
mkdir_if_not_exists(output_dir)

handler = KNNExampleBasedExplanation()

neigh_indices = handler.get_explanations(
    M = 5,
    clf=knn_clf,
    dataset=dataset_dict
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
)
from MinimalSubsetToFlipPredictions.wrappers.knn import FindMinimalSubsetKNN
from utils.constants.directory import RESULTS_DIR
from utils.io import mkdir_if_not_exists
from datasets import concatenate_datasets, DatasetDict

# load embeddings
from data.embeddings import load_saved_embeddings

train_embeddings = load_saved_embeddings(
    dataset="toxigen",
    model="deberta-large",
    seed=42,
    split="train",
    pooler="mean_with_attention",
    layer=24,
)

eval_embeddings = load_saved_embeddings(
    dataset="toxigen",
    model="deberta-large",
    seed=42,
    split="eval",
    pooler="mean_with_attention",
    layer=24,
)

test_embeddings = load_saved_embeddings(
    dataset="toxigen",
    model="deberta-large",
    seed=42,
    split="test",
    pooler="mean_with_attention",
    layer=24,
)

# load classifier
from data.models import load_saved_wrapperbox_model
from data.datasets import load_dataset_from_hf, load_labels_at_split
import numpy as np

dataset = load_dataset_from_hf(dataset="toxigen")
train_labels = load_labels_at_split(dataset, "train")
eval_labels = load_labels_at_split(dataset, "eval")
train_eval_labels = np.concatenate([train_labels, eval_labels])
test_labels = load_labels_at_split(dataset, "test")

output_dir = RESULTS_DIR / "MinimalSubset"
mkdir_if_not_exists(output_dir)

# Running SVM
handler = FindMinimalSubsetKNN()
wrapper_name = "KNN"

clf = load_saved_wrapperbox_model(
    dataset="toxigen",
    model="deberta-large",
    seed=42,
    pooler="mean_with_attention",
    wrapperbox=wrapper_name,
)

train_eval_embeddings = np.vstack([train_embeddings, eval_embeddings])
train_eval_labels = np.concatenate([train_labels, eval_labels])
train_eval_dataset = concatenate_datasets([dataset["train"], dataset["eval"]])
dataset_dict = DatasetDict(
    {"train": train_eval_dataset, "test": dataset["test"]}
)

minimal_subset_indices = handler.find_minimal_subset(
    clf=clf,
    train_embeddings=train_eval_embeddings,
    test_embeddings=test_embeddings,
    train_labels=train_eval_labels,
)

handler.persist_to_disk(
    dataset=dataset_dict,
    dataset_name="toxigen",
    model_name="deberta_large",
    wrapper_name=wrapper_name,
    minimal_subset_indices=minimal_subset_indices,
    output_dir=output_dir,
)

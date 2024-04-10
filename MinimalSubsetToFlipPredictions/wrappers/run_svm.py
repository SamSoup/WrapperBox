from MinimalSubsetToFlipPredictions.Yang2023.interface import (
    compute_minimal_subset_to_flip_predictions,
)
from MinimalSubsetToFlipPredictions.wrappers.svm import FindMinimalSubsetSVM
from utils.constants.directory import RESULTS_DIR
from utils.io import mkdir_if_not_exists


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
handler = FindMinimalSubsetSVM(ITERATIVE_THRESHOLD=10, SPLITS=10)

svm_clf = load_saved_wrapperbox_model(
    dataset="toxigen",
    model="deberta-large",
    seed=42,
    pooler="mean_with_attention",
    wrapperbox="SVM",
)

train_eval_embeddings = np.vstack([train_embeddings, eval_embeddings])
train_eval_labels = np.concatenate([train_labels, eval_labels])

# one example test
test_embeddings = test_embeddings[:25, :]

minimal_subset_indices = handler.find_minimal_subset(
    clf=svm_clf,
    train_embeddings=train_eval_embeddings,
    test_embeddings=test_embeddings,
    train_labels=train_eval_labels,
)

handler.persist_to_disk(
    dataset=dataset,
    dataset_name="toxigen",
    model_name="deberta_large",
    wrapper_name="SVM",
    minimal_subset_indices=minimal_subset_indices,
    output_dir=output_dir,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from ExampleBasedExplanations.knn import KNNExampleBasedExplanation
from data.datasets.load_dataset import load_labels_at_split
from datasets import Dataset, DatasetDict, concatenate_datasets
from utils.constants.directory import RESULTS_DIR
from utils.io import mkdir_if_not_exists
from utils.inference import compute_metrics, cross_validation_with_grid_search
from pprint import pprint
import numpy as np
import pandas as pd

# Load Embeddings
train_embeddings = np.load("AnubrataQA/embeddings/train_embeddings.npy")
eval_embeddings = np.load("AnubrataQA/embeddings/valid_embeddings.npy")
test_embeddings = np.load("AnubrataQA/embeddings/test_embeddings.npy")
train_eval_embeddings = np.vstack([train_embeddings, eval_embeddings])

# Load datasets from Parquet files
train_dataset = Dataset.from_pandas(
    pd.read_parquet("AnubrataQA/dataset/train.parquet")
)
valid_dataset = Dataset.from_pandas(
    pd.read_parquet("AnubrataQA/dataset/valid.parquet")
)
test_dataset = Dataset.from_pandas(
    pd.read_parquet("AnubrataQA/dataset/test.parquet")
)
# For each split, expect Column
# ['id', 'query', 'answer', 'choices', 'gold', 'text']"
dataset_dict = DatasetDict(
    {"train": train_dataset, "valid": valid_dataset, "test": test_dataset}
)

# edit the dataset dict to have a "text" and "label" column
for split in dataset_dict.keys():
    # 1. Rename the 'gold' column to 'label' in all splits, if not test
    dataset_dict[split] = dataset_dict[split].rename_column("gold", "label")
    # 2. Rename the 'text' column to 'fullText' in all splits
    dataset_dict[split] = dataset_dict[split].rename_column("text", "fullText")
    # 3. Rename the 'query' column to 'text' in all splits
    dataset_dict[split] = dataset_dict[split].rename_column("query", "text")

# Create the dataset with combined train and eval, for later use
train_labels = load_labels_at_split(dataset_dict, "train")
eval_labels = load_labels_at_split(dataset_dict, "valid")
test_labels = load_labels_at_split(dataset_dict, "test")
train_eval_labels = np.concatenate([train_labels, eval_labels])
new_dataset = DatasetDict(
    {
        "train": concatenate_datasets(
            [dataset_dict["train"], dataset_dict["valid"]]
        ),
        "test": dataset_dict["test"],
    }
)

# Load Models
M = 5
wrapper_name = "KNN"

# Hyper-parameter search with KNN
best_knn, best_score_knn, best_params_knn = cross_validation_with_grid_search(
    estimator=KNeighborsClassifier(n_jobs=-1),
    param_grid={
        "n_neighbors": [10],
    },
    X_train=train_embeddings,
    y_train=dataset_dict["train"]["label"],
    X_eval=eval_embeddings,
    y_eval=dataset_dict["valid"]["label"],
    scoring=lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
)
print(f"Best {wrapper_name} Valid Weighted F1:", best_score_knn)
print(f"Best {wrapper_name} Params:", best_params_knn)

predictions = best_knn.predict(test_embeddings)

# Print some metrics
testset_perfm = compute_metrics(
    y_true=test_labels, y_pred=predictions, is_multiclass=False, prefix="test"
)
pprint(testset_perfm)

# Obtain Example Based Explanations
handler = KNNExampleBasedExplanation()

neigh_indices = handler.get_explanation_indices(
    M=M,
    clf=best_knn,
    train_embeddings=train_eval_embeddings,
    test_embeddings=test_embeddings,
)

output_dir = "./AnubrataQA/results"
mkdir_if_not_exists(output_dir)

handler.persist_to_disk(
    dataset=new_dataset,
    dataset_name="AnubrataQA",
    model_name="SentenceT5Large",
    wrapper_name=wrapper_name,
    predictions=predictions,
    explanation_indices=neigh_indices,
    output_dir=output_dir,
)

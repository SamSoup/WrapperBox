"""
Runs pairwise comparisons between all transformers with results
"""

from itertools import combinations
from binomial import compute_binomial_p_value
from tqdm.auto import tqdm
from datasets import load_dataset
from constants import (
    DATASETS,
    MODELS,
    WORK_DIR,
    SEEDS,
    METRICS,
    MODEL_METADATAS,
    WRAPPER_BOXES,
)
from utils import load_predictions
import pickle
import pandas as pd
import numpy as np
import os

WORK_DIR = "/home/samsoup/Work/DkNN"
# for now, only do seed 42, last layer, mean_with_attention
SEEDS = list(filter(lambda x: x == 42, SEEDS))
pooler_config = "mean_with_attention"

for dataset in tqdm(DATASETS, desc="datasets"):
    # each metric will have their unique matrix of pairwise-comparison metrics
    results = {
        m: pd.DataFrame(np.nan, index=MODELS, columns=MODELS) for m in METRICS
    }
    data = load_dataset(f"Samsoup/{dataset}", token=True)
    y_test = np.array(data["test"]["label"])
    is_multiclass = np.unique(y_test).size > 2
    # run all pairwise comparsions
    for m1, m2 in combinations(MODELS, 2):
        for seed in SEEDS:
            m1_preds = np.array(
                load_predictions(WORK_DIR, dataset, f"{m1}-seed-{seed}")
            )
            m2_preds = np.array(
                load_predictions(WORK_DIR, dataset, f"{m2}-seed-{seed}")
            )
            p_values = compute_binomial_p_value(
                m1_preds,
                m2_preds,
                y_test,
                is_multiclass,
            )
            for metric in METRICS:
                df = results[metric]
                df.loc[m1, m2] = p_values[f"test_{metric}"]
    with open(
        os.path.join(
            WORK_DIR,
            "data",
            dataset,
            f"{dataset}_binomial_significance_tests.pkl",
        ),
        "wb",
    ) as handle:
        pickle.dump(results, handle)

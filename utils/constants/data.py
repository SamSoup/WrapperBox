from .directory import WORK_DIR
import os

# Data Related Variables

SPLITS = ["train", "eval", "test"]

DATASETS = {"asd": ["Last Only"], "esnli": ["Last Only"]}

LAYER_CONFIGS = ["All", "Embedding Only", "Embedding + Last", "Last Only"]

DATA_PATH = os.path.join(
    WORK_DIR, "data/{dataset}/{model}/{mode}/{pooler_config}/layer_{layer}.csv"
)

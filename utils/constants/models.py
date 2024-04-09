from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from .directory import WORK_DIR
from .data import DATASETS
import os

# Model Related Variables

SEEDS = [42, 365, 469, 4399, 3012023]

BASELINES = [
    "Glove-Twitter-200",
    "FastText-300",
    "Google-news-300",
    "SentenceBert",
]

MODELS = [
    "bart-large",
    "deberta-large",
    "flan-t5-large",
    # "t5-large"
    # "llama7B",
]

# defines where models outputs are stored
MODEL_CONFIGS = {
    dataset: {
        model: {
            seed: os.path.join(
                WORK_DIR, "output", dataset, f"{model}-seed-{seed}"
            )
            for seed in SEEDS
        }
        for model in MODELS
    }
    for dataset in DATASETS
}

# defines metadata specific to each model
MODEL_METADATAS = {
    "bart-large": {
        "num_layers": 26,
        "available_poolers": [
            "mean_with_attention",
            # "mean_with_attention_and_eos"
        ],
    },
    "deberta-large": {
        "num_layers": 25,
        "available_poolers": [
            "mean_with_attention",
            # "mean_with_attention_and_cls"
        ],
    },
    "flan-t5-large": {
        "num_layers": 50,
        "available_poolers": [
            "mean_with_attention",
            # "encoder_mean_with_attention_and_decoder_flatten"
        ],
    },
    "t5-large": {
        "num_layers": 50,
        "available_poolers": [
            "mean_with_attention",
            # "encoder_mean_with_attention_and_decoder_flatten"
        ],
    },
    "llama7B": {
        "num_layers": 33,
        "available_poolers": ["mean_with_attention"],
    },
}

WRAPPER_BOXES_NAMES = ["KNN", "SVM", "Decision_Tree", "L_Means"]

# Defines the base wrapper boxes
SVM_BASE = SVC(
    gamma="auto",
    class_weight="balanced",
    kernel="linear",
    random_state=42,
)

DT_BASE = DecisionTreeClassifier(
    random_state=42,
)

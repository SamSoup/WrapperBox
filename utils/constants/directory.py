# These are known locations from which data, models, and other should be lodaed
from pathlib import Path

WORK_DIR = Path(__file__).parent.parent.parent.absolute()

CACHE_DIR = WORK_DIR / ".cache"
RESULTS_DIR = WORK_DIR / "results"
DATA_DIR = WORK_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
SAVED_MODELS_DIR = DATA_DIR / "models"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PREDICTIONS_DIR = DATA_DIR / "predictions"

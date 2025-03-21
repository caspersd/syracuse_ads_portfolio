from pathlib import Path
from loguru import logger


##################### Shared Paths ##########################
PROJ_ROOT = Path(__file__).resolve().parents[0]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
DATA_DIR = PROJ_ROOT / "data"
OUTPUT_DIR = PROJ_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = PROJ_ROOT / "models"

##################### OPINION Paths ##########################

OPINION_DATA_DIR = DATA_DIR / "opinion_extraction"
OPINION_RAW_DATA_DIR = OPINION_DATA_DIR / "raw"
OPINION_EXTERNAL_DATA_DIR = OPINION_DATA_DIR / "external"
NEWSSD_DATA_DIR = OPINION_EXTERNAL_DATA_DIR / "newssd"
OPINION_SCRAPED_DATA_DIR = OPINION_RAW_DATA_DIR / "politician_public_statements"
OPINION_PROCESSED_DATA_DIR = OPINION_DATA_DIR / "processed"

MPQA_DIR = OPINION_EXTERNAL_DATA_DIR / "MPQA_subjclueslen1.tff"

OPINION_MODELS_DIR= MODELS_DIR / "opinion_extraction"

BERT_MODEL_DIR = OPINION_MODELS_DIR / "bert_opinion_model.keras"

OPINION_RESULTS_DIR = OUTPUT_DIR / "opinion_results.txt"

##################### Ideology Paths ##########################
IDIOLOGY_DATA_DIR = DATA_DIR / "political_ideology_ranking"
IDIOLOGY_RAW_DATA_DIR = IDIOLOGY_DATA_DIR / "raw"
IDEOLOGY_RAW_POLITICAL_STATEMENTS_DIR = IDIOLOGY_RAW_DATA_DIR / "scraped_political_text"

IDEOLOGY_INTERIM_DATA_DIR = IDIOLOGY_DATA_DIR / "interim"

IDEOLOGY_FEATURES_DATA_DIR = IDIOLOGY_DATA_DIR / "features"
IDIOLOGY_PROCESSED_DIR = IDIOLOGY_DATA_DIR / "processed"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

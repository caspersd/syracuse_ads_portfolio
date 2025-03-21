
import typer
from loguru import logger
from pathlib import Path

# Import constants from config.py
from config import NEWSSD_DATA_DIR, OPINION_PROCESSED_DATA_DIR, MPQA_DIR

# Import processing modules
from src.opinion_extraction.dataset_processor import DatasetProcessor
from src.opinion_extraction.mpqa_processor import MPQAFeatureExtractor
from src.opinion_extraction.tfidf_processor import FeatureExtractor
from src.opinion_extraction.bert_processor import BERTDatasetProcessor


def preprocess_data():
    """
    Runs the full preprocessing pipeline:
    1. Load & Process Dataset
    2. Process MPQA Lexicon
    3. Extract Features (TF-IDF and MPQA-based)
    4. Prepare BERT Datasets
    """

    logger.info("Starting data preprocessing pipeline...")

    ## 1. Load & Process Dataset
    logger.info("Loading and processing datasets...")
    dataset_processor = DatasetProcessor(NEWSSD_DATA_DIR, OPINION_PROCESSED_DATA_DIR)
    dataset_processor.load_datasets()
    dataset_processor.save_datasets()  # Ensures processed datasets are stored for later use

    ## 2. Process MPQA Lexicon
    logger.info("Processing MPQA Subjectivity Lexicon...")
    mpqa_processor = MPQAFeatureExtractor(MPQA_DIR)
    mpqa_processor.save_subjectivity_dict(OPINION_PROCESSED_DATA_DIR / "mpqa_subjectivity_dict.pkl")
    mpqa_processor.generate_mpqa_matrix(dataset_processor.opinions_train)
    mpqa_processor.save_mpqa_matrix(OPINION_PROCESSED_DATA_DIR)

    ## 3. Extract Features (TF-IDF & MPQA)
    logger.info("Extracting features using TF-IDF...")
    feature_extractor = FeatureExtractor(OPINION_PROCESSED_DATA_DIR)
    
    # Extract TF-IDF features
    tfidf_features = feature_extractor.extract_features(dataset_processor.opinions_train)
    feature_extractor.save_features()  # Save the extracted features and vectorizers

    ## 4. Process BERT Dataset
    logger.info("Processing BERT-based datasets...")
    bert_processor = BERTDatasetProcessor(OPINION_PROCESSED_DATA_DIR)
    bert_processor.process_datasets(
        dataset_processor.opinions_train,
        dataset_processor.opinions_val,
        dataset_processor.opinions_test
    )

    logger.success("Data preprocessing pipeline completed successfully!")


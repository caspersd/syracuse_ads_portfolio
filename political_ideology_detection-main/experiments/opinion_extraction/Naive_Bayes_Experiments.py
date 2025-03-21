from pathlib import Path
import pickle
from loguru import logger
from src.opinion_extraction.data_loader import DatasetLoader
from src.opinion_extraction.NBTrainer import NaiveBayesTrainer

def run_naive_bayes_experiment(processed_path: Path, results_path: Path, model_save_path: Path, mpqa_path: Path):
    """
    Runs the Naive Bayes experiment pipeline from start to finish.

    Args:
        processed_path (Path): Path to the processed dataset directory.
        results_path (Path): Path to save evaluation results.
        model_save_path (Path): Path to save trained models.
        mpqa_path (Path): Path to MPQA subjectivity lexicon.

    Returns:
        None
    """
    logger.info("Starting Naive Bayes Experiment...")

    # Load datasets
    dataset_loader = DatasetLoader(processed_path)
    train_df, test_df, _ = dataset_loader.load_base_datasets()

    # Load TF-IDF matrices and vectorizers
    tfidf_data = dataset_loader.load_tfidf_datasets()

    # Train and evaluate Naive Bayes models
    logger.info("Training and Evaluating Naive Bayes models...")
    nb_trainer = NaiveBayesTrainer(processed_path, results_path, model_save_path, mpqa_path)

    # Train and evaluate Naive Bayes models using TF-IDF features
    nb_trainer.run(train_df, test_df, tfidf_data)
    
    # Save trained models
    nb_trainer.save_models()

    logger.success("Naive Bayes Experiment completed successfully!")

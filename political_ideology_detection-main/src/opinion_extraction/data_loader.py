from pathlib import Path
import pickle
import tensorflow as tf
from loguru import logger
from src.opinion_extraction.text_processing_helpers import spacy_tokenizer, spacy_pos_tokenizer, spacy_pos_count_tokenizer


class DatasetLoader:
    """Handles loading of different datasets including raw, TF-IDF, and BERT datasets."""

    def __init__(self, processed_data_dir: Path):
        self.processed_data_dir = processed_data_dir

    def load_pickle_file(self, filename):
        """Helper function to load a pickle file."""
        file_path = self.processed_data_dir / filename
        if file_path.exists():
            with open(file_path, "rb") as file:
                return pickle.load(file)
        else:
            logger.warning(f"File {filename} not found.")
            return None

    def load_base_datasets(self):
        """Loads pre-processed base datasets (train, test, val)."""
        logger.info("Loading base datasets...")
        train_df = self.load_pickle_file("train_df.pkl")
        test_df = self.load_pickle_file("test_df.pkl")
        val_df = self.load_pickle_file("val_df.pkl")
        logger.success("Base datasets loaded successfully.")
        return train_df, test_df, val_df

    def load_tfidf_datasets(self):
        """Loads TF-IDF matrices and vectorizers as tuples (matrix, vectorizer)."""
        try:
            logger.info("Loading TF-IDF datasets...")

            # List of TF-IDF feature types
            feature_names = [
                "unigram",
                "bigram_trigram",
                "all_grams",
                "pos_tagged",
                "pos_counts"
            ]

            tfidf_data = []

            for name in feature_names:
                matrix = self.load_pickle_file(f"{name}_matrix.pkl")
                vectorizer = self.load_pickle_file(f"{name}_vectorizer.pkl")
                
                if matrix is not None and vectorizer is not None:
                    tfidf_data.append((name, matrix, vectorizer))
                    logger.success(f"Loaded TF-IDF dataset: {name}")
                else:
                    logger.warning(f"Skipping {name} due to missing files.")

            return tfidf_data

        except Exception as e:
            logger.error(f"Error loading TF-IDF datasets: {e}")
            raise

    def load_bert_datasets(self):
        """Loads preprocessed BERT datasets."""
        logger.info("Loading BERT datasets...")
        try:
            bert_train = tf.data.experimental.load(str(self.processed_data_dir / "train_BERT_dataset"))
            bert_test = tf.data.experimental.load(str(self.processed_data_dir / "test_BERT_dataset"))
            bert_val = tf.data.experimental.load(str(self.processed_data_dir / "val_BERT_dataset"))
            logger.success("BERT datasets loaded successfully.")
            return bert_train, bert_test, bert_val
        except Exception as e:
            logger.error(f"Error loading BERT datasets: {e}")
            return None, None, None

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from loguru import logger
import spacy
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src.opinion_extraction.text_processing_helpers import (
    spacy_tokenizer, spacy_pos_tokenizer, spacy_pos_count_tokenizer
)


class FeatureExtractor:
    """
    Extracts TF-IDF features from text data using different tokenization strategies.

    Attributes:
        output_path (Path): Path to save extracted feature matrices and vectorizers.
        vectorizers (dict): Dictionary to store trained vectorizers.
        features (dict): Dictionary to store extracted TF-IDF feature matrices.
    """

    def __init__(self, output_path: Path):
        """
        Initializes the FeatureExtractor.

        Args:
            output_path (Path): Directory where extracted features will be stored.
        """
        self.output_path = output_path
        self.vectorizers = {}  # Dictionary to store vectorizers
        self.features = {}  # Dictionary to store feature matrices

        # Load Spacy NLP Model
        self.nlp = spacy.load("en_core_web_sm")

    def extract_features(self, dataset):
        """
        Extracts TF-IDF features using different tokenization strategies.

        Args:
            dataset (pd.DataFrame): The dataset containing the text column "Sentence".

        Returns:
            dict: A dictionary where keys are feature names and values are TF-IDF matrices.
        """
        feature_configs = {
            "unigram": (spacy_tokenizer, (1, 1)),
            "bigram_trigram": (spacy_tokenizer, (2, 3)),
            "all_grams": (spacy_tokenizer, (1, 3)),
            "pos_tagged": (spacy_pos_tokenizer, (1, 3)),
            "pos_counts": (spacy_pos_count_tokenizer, (1, 3))
        }

        for feature_name, (tokenizer, ngram_range) in feature_configs.items():
            matrix, vectorizer = self._generate_tfidf(tokenizer, dataset, ngram_range)
            self.features[feature_name] = matrix
            self.vectorizers[feature_name] = vectorizer

        logger.success("TF-IDF datasets generated successfully.")
        return self.features  # Now returning a dictionary

    def _generate_tfidf(self, tokenizer, dataset, ngram_range):
        """
        Generates a TF-IDF feature matrix for the given dataset.

        Args:
            tokenizer (function): Tokenization function.
            dataset (pd.DataFrame): The dataset containing the text column "Sentence".
            ngram_range (tuple): The n-gram range for feature extraction.

        Returns:
            tuple: (TF-IDF matrix, trained TfidfVectorizer).
        """
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, max_features=2000)
        tfidf_matrix = vectorizer.fit_transform(dataset["Sentence"])
        return tfidf_matrix, vectorizer

    def save_features(self):
        """
        Saves the extracted feature matrices and vectorizers as pickle files.
        """
        for name, matrix in self.features.items():
            file_path = self.output_path / f"{name}_matrix.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(matrix, file)
            logger.success(f"Saved TF-IDF matrix: {file_path}")

        for name, vectorizer in self.vectorizers.items():
            file_path = self.output_path / f"{name}_vectorizer.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(vectorizer, file)
            logger.success(f"Saved vectorizer: {file_path}")

    def load_features(self):
        """
        Loads previously saved feature matrices and vectorizers from the output path.

        Returns:
            tuple: (dict of loaded matrices, dict of loaded vectorizers)
        """
        loaded_features = {}
        loaded_vectorizers = {}

        for name in ["unigram", "bigram_trigram", "all_grams", "pos_tagged", "pos_counts"]:
            matrix_path = self.output_path / f"{name}_matrix.pkl"
            vectorizer_path = self.output_path / f"{name}_vectorizer.pkl"

            if matrix_path.exists():
                with open(matrix_path, "rb") as file:
                    loaded_features[name] = pickle.load(file)

            if vectorizer_path.exists():
                with open(vectorizer_path, "rb") as file:
                    loaded_vectorizers[name] = pickle.load(file)

        return loaded_features, loaded_vectorizers

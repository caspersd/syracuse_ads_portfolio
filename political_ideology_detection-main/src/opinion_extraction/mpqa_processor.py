import pickle
import numpy as np
from pathlib import Path
from loguru import logger
from scipy.sparse import csr_matrix, hstack
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from src.opinion_extraction.text_processing_helpers import (
    spacy_tokenizer, spacy_pos_tokenizer
)


class MPQAFeatureExtractor:
    """Handles loading and processing of MPQA subjectivity lexicon and feature matrix generation."""

    def __init__(self, mpqa_path: Path):
        self.mpqa_path = mpqa_path
        self.subjectivity_dict = self._load_subjectivity_lexicon()
        self.nlp = spacy.load("en_core_web_sm")  # Load Spacy NLP model
        self.mpqa_matrix = None
        self.mpqa_vectorizer = None

    def _load_subjectivity_lexicon(self):
        """Loads MPQA subjectivity lexicon and creates a dictionary."""
        logger.info("Loading MPQA subjectivity lexicon...")
        subjectivity_dict = {}
        with open(self.mpqa_path, "r") as file:
            for line in file:
                elements = line.strip().split(" ")
                word = elements[2].split("=")[1]
                strength = elements[0].split("=")[1]
                subjectivity_dict[word] = strength
        logger.success("MPQA subjectivity lexicon loaded successfully.")
        return subjectivity_dict

    def save_subjectivity_dict(self, output_path):
        """Save the MPQA subjectivity dictionary as a pickle file."""
        with open(output_path, "wb") as file:
            pickle.dump(self.subjectivity_dict, file)
        logger.success(f"MPQA subjectivity dictionary saved at {output_path}")

    def _compute_mpqa_features(self, text_tokens):
        """
        Computes sentiment features using the MPQA subjectivity lexicon.

        Args:
        - text_tokens (list of list): Tokenized sentences.

        Returns:
        - np.array: Array of sentiment scores for each sentence.
        """
        features = []
        for doc in text_tokens:
            weakSubj = sum(1 for word in doc if word in self.subjectivity_dict and self.subjectivity_dict[word] == 'weaksubj')
            strongSubj = sum(1 for word in doc if word in self.subjectivity_dict and self.subjectivity_dict[word] == 'strongsubj')
            doc_len = max(len(doc), 1)  # Avoid division by zero
            feature = (weakSubj + (2 * strongSubj)) / doc_len
            features.append(feature)
        return np.array(features).reshape(-1, 1)  # Reshape to column vector

    def _generate_pos_count_matrix(self, dataset):
        """
        Generate a POS count feature matrix from the dataset.

        Args:
        - dataset (pandas.DataFrame): DataFrame containing a "Sentence" column.

        Returns:
        - csr_matrix: POS count feature matrix.
        """
        logger.info("Generating POS count matrix...")

        # Tokenize and extract POS tags
        pos_sentences = [" ".join(spacy_pos_tokenizer(sentence)) for sentence in dataset["Sentence"]]

        # Vectorize POS counts
        vectorizer = CountVectorizer()
        pos_count_matrix = vectorizer.fit_transform(pos_sentences)
        self.mpqa_vectorizer = pos_count_matrix
        logger.success("POS count matrix generated successfully.")
        return pos_count_matrix

    def generate_mpqa_matrix(self, dataset):
        """
        Generate an MPQA feature matrix and automatically generate the POS count matrix.

        Args:
        - dataset (pandas.DataFrame): DataFrame containing a "Sentence" column.

        Returns:
        - csr_matrix: Combined MPQA + POS count feature matrix.
        """
        logger.info("Generating MPQA feature matrix...")

        # Tokenize sentences
        tokenized_sentences = [spacy_tokenizer(sentence) for sentence in dataset["Sentence"]]

        # Compute MPQA sentiment features
        mpqa_features = self._compute_mpqa_features(tokenized_sentences)

        # Convert MPQA features to a sparse matrix
        mpqa_sparse_matrix = csr_matrix(mpqa_features)

        # Generate POS count matrix
        pos_count_matrix = self._generate_pos_count_matrix(dataset)

        # Combine MPQA features with the POS count matrix
        combined_matrix = hstack([pos_count_matrix, mpqa_sparse_matrix])

        logger.success("MPQA + POS count feature matrix generated successfully.")
        return combined_matrix

    def save_mpqa_matrix(self, output_path):
        """Save the MPQA subjectivity dictionary as a pickle file."""
        with open(output_path / "mpqa.pkl", "wb") as file:
            pickle.dump(self.mpqa_matrix, file)
        with open(output_path / "mpqa_vectorizer.pkl", "wb") as file:
            pickle.dump(self.mpqa_vectorizer, file)
        logger.success(f"MPQA subjectivity dictionary saved at {output_path}")

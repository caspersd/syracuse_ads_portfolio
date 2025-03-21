import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from loguru import logger
from src.opinion_extraction.data_loader import DatasetLoader
from src.opinion_extraction.mpqa_processor import MPQAFeatureExtractor


class NaiveBayesTrainer:
    """Trains and evaluates Naive Bayes models using different feature representations, including MPQA."""

    def __init__(self, processed_path, results_path, model_save_path, mpqa_path):
        """
        Initializes the Naive Bayes trainer.

        Args:
            processed_path (Path): Path to the processed dataset.
            results_path (Path): Path to save model evaluation results.
            model_save_path (Path): Path to save trained models.
            mpqa_path (Path): Path to MPQA lexicon.
        """
        self.processed_path = processed_path
        self.results_path = results_path
        self.model_save_path = model_save_path
        self.results = {}

    def train_model(self, matrix, labels, model_name):
        """
        Train Naive Bayes models using cross-validation.

        Args:
            matrix (csr_matrix): Feature matrix.
            labels (pd.Series): Labels for training.
            model_name (str): Name of the model.

        Returns:
            dict: Dictionary with trained models and evaluation metrics.
        """
        models = {'MNB': MultinomialNB(), 'BNB': BernoulliNB()}
        kf = StratifiedKFold(n_splits=5)
        self.results[model_name] = {}

        for name, model in models.items():
            y_pred = cross_val_predict(model, matrix, labels, cv=kf)
            precision, recall, fscore, _ = precision_recall_fscore_support(labels, y_pred, average='weighted')
            accuracy = accuracy_score(labels, y_pred)

            # Store trained model and metrics
            self.results[model_name][name] = {
                'precision': precision,
                'recall': recall,
                'fscore': fscore,
                'accuracy': accuracy,
                'fitted_model': model.fit(matrix, labels)
            }

        return self.results[model_name]

    def evaluate_model(self, model, vectorizer, dataset, model_name):
        """
        Evaluate a trained model on the test set.

        Args:
            model (sklearn model): Trained Naive Bayes model.
            vectorizer (TfidfVectorizer): Corresponding vectorizer.
            dataset (pd.DataFrame): Test dataset containing 'Sentence' and 'Label'.
            model_name (str): Name of the model for logging.

        Returns:
            dict: Evaluation metrics.
        """
        matrix = vectorizer.transform(dataset["Sentence"])
        predictions = model.predict(matrix)

        precision, recall, fscore, _ = precision_recall_fscore_support(dataset["Label"], predictions, average='weighted')
        accuracy = accuracy_score(dataset["Label"], predictions)
        conf_matrix = confusion_matrix(dataset["Label"], predictions)
        report = classification_report(dataset["Label"], predictions)

        results = {
            "precision": precision, "recall": recall, "fscore": fscore,
            "accuracy": accuracy, "conf_matrix": conf_matrix, "report": report
        }

        self._log_results(results, model_name)
        return results

    def save_models(self):
        """
        Iterates through the trained models and saves them as pickle files.
        """
        logger.info("Saving trained models...")
        for model_name, model_data in self.results.items():
            for model_type, model_info in model_data.items():
                model = model_info['fitted_model']
                save_path = self.model_save_path / f"{model_name}_{model_type}.pkl"
                with open(save_path, "wb") as file:
                    pickle.dump(model, file)
                logger.success(f"Saved {model_name} {model_type} model to {save_path}")

    def _log_results(self, results, model_name):
        """
        Logs classification results to a file.

        Args:
            results (dict): Evaluation metrics.
            model_name (str): Name of the model.
        """
        with open(self.results_path, "a") as file:
            file.write(f"\n{model_name} Model Results\n")
            file.write(f"Precision: {results['precision']}\n")
            file.write(f"Recall: {results['recall']}\n")
            file.write(f"Accuracy: {results['accuracy']}\n")
            file.write(f"F1 Score: {results['fscore']}\n")
            file.write(f"Confusion Matrix:\n{results['conf_matrix']}\n")
            file.write(f"Classification Report:\n{results['report']}\n")
        logger.success(f"Results saved to {self.results_path}")


    def run(self, train_df, test_df, tfidf_data):
        """
        Train, evaluate, and save all Naive Bayes models.

        Args:
            train_df (pd.DataFrame): Training dataset.
            test_df (pd.DataFrame): Test dataset.
            tfidf_data (list of tuples): [(name, matrix, vectorizer)].
            vectorizer_data (dict): Dictionary of vectorizers.

        Returns:
            None
        """
        for model_name, matrix, vectorizer in tfidf_data:
            logger.info(f"Training {model_name} model...")
            self.train_model(matrix, train_df["Label"], model_name)

            logger.info(f"Evaluating {model_name} model...")
            self.evaluate_model(
                self.results[model_name]['MNB']['fitted_model'], vectorizer, test_df, f"{model_name} MNB"
            )
            self.evaluate_model(
                self.results[model_name]['BNB']['fitted_model'], vectorizer, test_df, f"{model_name} BNB"
            )

        self.save_models()
        logger.success("All Naive Bayes models trained, evaluated, and saved successfully.")

'''
# Run the trainer
if __name__ == "__main__":
    trainer = NaiveBayesTrainer(
        processed_path=Path("data/processed"),
        results_path=Path("models/results.txt"),
        model_save_path=Path("models"),
        mpqa_path=Path("data/mpqa")
    )
    trainer.run()
'''
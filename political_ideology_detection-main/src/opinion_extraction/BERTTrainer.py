from pathlib import Path
import typer
from loguru import logger
import numpy as np
import pickle
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

from config import OPINION_PROCESSED_DATA_DIR, BERT_MODEL_DIR, OPINION_RESULTS_DIR
from src.opinion_extraction.data_loader import DatasetLoader

# Initialize Typer CLI
app = typer.Typer()


class BERTTrainer:
    def __init__(self, processed_path=OPINION_PROCESSED_DATA_DIR, model_path=BERT_MODEL_DIR, results_path=OPINION_RESULTS_DIR):
        """Initialize paths and model parameters."""
        self.processed_path = processed_path
        self.model_path = model_path
        self.results_path = results_path
        self.model = None
        self.train_dataset=self.test_dataset= self.val_dataset = None
        self.train_df= self.test_df= self.val_df = None


    def load_datasets(self):
        """Load TensorFlow datasets."""
        logger.info("Loading TensorFlow datasets . . .")
        loader = DatasetLoader(self.processed_path)
        self.train_dataset, self.test_dataset, self.val_dataset = loader.load_bert_datasets()
        self.train_df, self.test_df, self.val_df = loader.load_base_datasets()
        logger.success("TensorFlow Datasets Loaded")

    def build_model(self):
        """Initialize the BERT model."""
        logger.info("Generating BERT Model . . .")
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

        # Freeze all layers except the last two
        for layer in self.model.distilbert.transformer.layer[:-2]:
            layer.trainable = False

        # Load model checkpoint if it exists
        if self.model_path.exists():
            logger.info(f"Checkpoint found at {self.model_path}, loading weights . . .")
            self.model.load_weights(str(self.model_path))
            logger.success("Weights loaded successfully from checkpoint.")
        else:
            logger.info("No checkpoint found. Loading stock model.")


    def train_model(self, epochs=5):
        """Compute class weights to handle class imbalance."""
        logger.info("Computing class weights . . .")
        with open(self.processed_path / "train_df.pkl", "rb") as file:
            train_labels = pickle.load(file)["Label"]
            train_labels = np.array(train_labels)

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        logger.info(f"Class Weights: {self.class_weights_dict}")


        """Train the model with class weights and validation dataset."""
        logger.info("Training the model . . .")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )


        self.model.fit(
            self.train_dataset,
            class_weight=self.class_weights_dict,
            epochs=epochs,
            validation_data=self.val_dataset,
            verbose=1
        )

        logger.success("Model Training complete.")

    def save_model(self):
        """Save trained model."""
        logger.info(f"Saving the model to {self.model_path} . . .")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        logger.success(f"Model saved successfully at {self.model_path}")

    def evaluate_model(self, dataset = None):
        if dataset is None:
            dataset = self.test_dataset
        
        """Evaluate the trained model and save results."""
        logger.info("Evaluating model . . .")
        results = self.model.predict(dataset)
        predictions = tf.nn.softmax(tf.convert_to_tensor(results.logits), axis=-1).numpy()
        predictions = np.argmax(predictions, axis=1)

        # Extract labels from test_dataset
        def extract_labels(dataset):
            labels = []
            for _, y in dataset:
                labels.extend(y.numpy())  
            return np.array(labels)

        y_test = extract_labels(dataset)

        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predictions, average="weighted")
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)

        # Save results
        with open(self.results_path, "a") as file:
            file.write("\n" + "=" * 80 + "\n")
            file.write(f"BERT Model:\n")
            file.write(f"Precision: {precision}\n")
            file.write(f"Recall: {recall}\n")
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"F1 Score: {fscore}\n")
            file.write(f"Confusion Matrix:\n{conf_matrix}\n")
            file.write(f"Classification Report:\n{report}\n\n")

        logger.success(f"Results saved to {self.results_path}")
        print(f"Results appended to {self.results_path}")
    
    def predict(self, dataset):
        results = self.model.predict(dataset)
        predictions = tf.nn.softmax(tf.convert_to_tensor(results.logits), axis=-1).numpy()
        predictions = np.argmax(predictions, axis=1)

        return predictions


    def run(self):
        """Run the full BERT pipeline."""
        self.load_datasets()
        self.build_model()
        self.train_model()
        self.save_model()
        self.evaluate_model()


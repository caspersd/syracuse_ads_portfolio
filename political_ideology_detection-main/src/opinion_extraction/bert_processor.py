from transformers import DistilBertTokenizerFast
from datasets import Dataset
import tensorflow as tf
from loguru import logger
from pathlib import Path

class BERTDatasetProcessor:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_function(self, examples):
        return self.tokenizer(examples["Sentence"], padding="max_length", truncation=True, max_length=512)

    def process_datasets(self, opinions_train, opinions_val, opinions_test):
        """Tokenizes datasets and converts them to TensorFlow format."""
        datasets = {
            "train": Dataset.from_pandas(opinions_train).map(self.tokenize_function, batched=True),
            "val": Dataset.from_pandas(opinions_val).map(self.tokenize_function, batched=True),
            "test": Dataset.from_pandas(opinions_test).map(self.tokenize_function, batched=True),
        }

        for name, dataset in datasets.items():
            tf_dataset = dataset.to_tf_dataset(columns=["input_ids", "attention_mask"], label_cols="Label", batch_size=16)
            tf.data.experimental.save(tf_dataset, str(self.output_path / f"{name}_BERT_dataset"))
            logger.success(f"{name} BERT dataset saved.")

import pandas as pd
import csv
import pickle
from pathlib import Path
from loguru import logger

class DatasetProcessor:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_coding = {"OBJ": 0, "SUBJ": 1}

    def load_datasets(self):
        """Loads training, validation, and test datasets."""
        self.opinions_train = pd.read_csv(self.input_path / "train_newssd.csv", quoting=csv.QUOTE_ALL)
        self.opinions_test = pd.read_csv(self.input_path / "test_newssd.csv", quoting=csv.QUOTE_ALL)
        self.opinions_val = pd.read_csv(self.input_path / "val_newssd.csv", quoting=csv.QUOTE_ALL)

        # Convert labels to numerical values
        self.opinions_train["Label"] = self.opinions_train["Label"].map(self.label_coding)
        self.opinions_test["Label"] = self.opinions_test["Label"].map(self.label_coding)
        self.opinions_val["Label"] = self.opinions_val["Label"].map(self.label_coding)

        logger.success("Datasets successfully loaded and labels encoded.")

    def save_datasets(self):
        """Saves processed datasets."""
        datasets = {
            "train_df.pkl": self.opinions_train,
            "test_df.pkl": self.opinions_test,
            "val_df.pkl": self.opinions_val,
        }
        for file_name, dataset in datasets.items():
            with open(self.output_path / file_name, "wb") as file:
                pickle.dump(dataset, file)
        logger.success("Processed datasets saved.")

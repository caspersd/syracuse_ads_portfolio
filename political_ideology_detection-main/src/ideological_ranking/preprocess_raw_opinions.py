import os
import csv
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
BATCH_SIZE = 500  # Process in batches

def extract_opinions(input_directory: Path, output_directory: Path, predict_function):
    """
    Processes all folders and extracts opinion sentences.
    """
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for year_folder in sorted(input_directory.iterdir()):
        if year_folder.is_dir():
            logger.info(f"Processing folder: {year_folder.name}")
            process_folder(year_folder, output_directory, predict_function)

def process_folder(folder_path: Path, output_directory: Path, predict_function):
    """
    Processes all CSV files within a folder and extracts opinion sentences.
    """
    output_file = output_directory / f"extracted_opinion_sentences_{folder_path.name}.csv"

    if not output_file.exists():
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(["topic", "date", "title", "politician", "sentence"])
    
    accumulated_sentences = []
    file_num = 0
    for file_path in tqdm(list(folder_path.glob("*.csv")), desc=f"Processing {folder_path.name}", unit="file"):
        file_num += 1
        new_sentences = process_file(file_path)
        accumulated_sentences.extend(new_sentences)
        
        if len(accumulated_sentences) >= BATCH_SIZE:
            process_and_write_opinions(accumulated_sentences, output_file, predict_function)
            accumulated_sentences = []  # Clear batch
    
    if accumulated_sentences:
        process_and_write_opinions(accumulated_sentences, output_file, predict_function)
    
    logger.success(f"Finished processing folder: {folder_path.name}")

def process_file(file_path: Path):
    """
    Processes a single CSV file and extracts opinion sentences.
    """
    df = pd.read_csv(file_path, dtype=str, usecols=["topic", "date", "title", "politician", "full_text"])
    
    sentences_data = []
    
    for _, row in df.iterrows():
        if pd.isna(row["full_text"]):
            continue
        sentences = row["full_text"].split(". ")
        sentences_data.extend([[row["topic"], row["date"], row["title"], row["politician"], sentence] for sentence in sentences])
    
    return sentences_data

def process_and_write_opinions(sentences_data, output_file, predict_function):
    """
    Runs prediction in batches and writes results to the file.
    """
    opinion_sentences = predict_opinions(sentences_data, predict_function)
    
    if opinion_sentences:
        with open(output_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerows(opinion_sentences)

def predict_opinions(sentences_data, predict_function):
    """
    Converts sentences to TensorFlow format and runs predictions.
    """
    if not sentences_data:
        return []
    sentences = [sentence[4] for sentence in sentences_data]
    encodings = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="tf")
    predictions = predict_function(encodings)
    
    return [sentences_data[i] for i, pred in enumerate(predictions) if pred == 1]

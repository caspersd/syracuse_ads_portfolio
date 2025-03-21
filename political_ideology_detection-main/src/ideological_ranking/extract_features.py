import os
import csv
import pandas as pd
import tensorflow as tf
import pickle
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from transformers import DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def filter_embed_senator_sentences(file_path_2022: Path, file_path_2023: Path, saving_path: Path, feature_path: Path):
    """
    Filters sentences to only retain those from senators who have at least 50 sentences in both years.
    """
    politican_speeches_2022 = pd.read_csv(file_path_2022)
    politican_speeches_2023 = pd.read_csv(file_path_2023)
    
    # Compute sentence counts for 2022
    sentence_counts_2022 = politican_speeches_2022.groupby("politician")["sentence"].count().reset_index()
    sentence_counts_2022.rename(columns={"sentence": "count_2022"}, inplace=True)

    # Compute sentence counts for 2023
    sentence_counts_2023 = politican_speeches_2023.groupby("politician")["sentence"].count().reset_index()
    sentence_counts_2023.rename(columns={"sentence": "count_2023"}, inplace=True)

    # Merge counts
    merged_counts = pd.merge(sentence_counts_2022, sentence_counts_2023, on="politician", how="inner")

    # Filter for politicians with more than 50 sentences in both years
    valid_politicians = merged_counts[(merged_counts["count_2022"] > 50) & (merged_counts["count_2023"] > 50)]["politician"]

    # Retain sentences from both datasets
    filtered_2022 = politican_speeches_2022[politican_speeches_2022["politician"].isin(valid_politicians)]
    filtered_2023 = politican_speeches_2023[politican_speeches_2023["politician"].isin(valid_politicians)]

    # Filter to keep only senators
    filtered_2022 = filtered_2022[filtered_2022["politician"].str.startswith("Sen.")]
    filtered_2023 = filtered_2023[filtered_2023["politician"].str.startswith("Sen.")]

    # Reset index
    filtered_2022 = filtered_2022.reset_index(drop=True)
    filtered_2023 = filtered_2023.reset_index(drop=True)  

    save_path_2022 = saving_path / "filtered_sentences_2022.csv"
    save_path_2023 = saving_path / "filtered_sentences_2023.csv"
    filtered_2022.to_csv(save_path_2022, index=False)
    filtered_2023.to_csv(save_path_2023, index=False)

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sbert_model.encode(filtered_2022["sentence"].tolist(), show_progress_bar=True)
    # Convert embeddings to a DataFrame
    embeddings_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
    # Add politician names and topics
    embeddings_df["politician"] = filtered_2022["politician"]
    embeddings_df["topic"] = filtered_2022["topic"]

    # Save embedded sentences to file
    feature_save_path = feature_path / "embeddings_df.pkl"
    with open(feature_save_path, 'wb') as file:
        pickle.dump(embeddings_df, file)
    
    mean_embeddings = embeddings_df.groupby(["politician"], as_index=False).mean(numeric_only=True)
    feature_save_path = feature_path / "mean_embeddings_df.pkl"
    with open(feature_save_path, 'wb') as file:
        pickle.dump(mean_embeddings, file)

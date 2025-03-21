import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import pickle
import typer
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger


from config import IDEOLOGY_FEATURES_DATA_DIR, OUTPUT_DIR


def rank_politicians(base_politician_1: str = "Sen. Dan Sullivan", base_politician_2: str = "Sen. Chuck Schumer") -> pd.DataFrame:
    """
    Ranks politicians based on their vector position in the embedding space relative to two base politicians.
    """
    # Load embeddings
    features_path = IDEOLOGY_FEATURES_DATA_DIR / "mean_embeddings_df.pkl"
    with open(features_path, 'rb') as file:
        mean_embeddings = pickle.load(file)
    print(mean_embeddings)

    # Get embeddings for the two base politicians
    base1_embedding = mean_embeddings.loc[mean_embeddings['politician'] == base_politician_1, mean_embeddings.columns[1:]].values.flatten()
    base2_embedding = mean_embeddings.loc[mean_embeddings['politician'] == base_politician_2, mean_embeddings.columns[1:]].values.flatten()
    
    # Compute direction vector
    v = base2_embedding - base1_embedding
    v_norm = np.linalg.norm(v)
    
    # Function to compute normalized magnitude
    def calculate_magnitude(embedding):
        projection = base1_embedding + (np.dot(embedding - base1_embedding, v) / np.dot(v, v)) * v
        raw_magnitude = np.dot(projection - base1_embedding, v) / v_norm
        return raw_magnitude
    
    # Compute magnitudes for all politicians
    rankings = [
        {"Senator": row["politician"], "Magnitude": calculate_magnitude(row[1:].values)}
        for _, row in mean_embeddings.iterrows()
    ]
    
    df = pd.DataFrame(rankings)
    
    # Normalize magnitude to 0-100%
    df["Magnitude"] = (df["Magnitude"] - df["Magnitude"].min()) / (df["Magnitude"].max() - df["Magnitude"].min()) * 100

    return df



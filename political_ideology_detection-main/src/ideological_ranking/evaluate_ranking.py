import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger

from config import OUTPUT_DIR


def eval_ranking(rank_df) -> dict:
    """
    Computes Spearman's correlation coefficient and Kendall's tau correlation coefficient
    between the actual legislative ranking and predicted ranking.
    
    Parameters:
        rank_df (pd.DataFrame): DataFrame containing Senator and Magnitude columns.
    
    Returns:
        dict: Dictionary containing Spearman's correlation coefficient and Kendall's tau correlation coefficient.
    """

    
    # Extract senator rankings from Legislative Analysis
    senator_data = [
        ("Sen. Mike Lee", 1),
        ("Sen. Rand Paul", 1),
        ("Sen. Ron Johnson", 3),
        ("Sen. Mike Braun", 4),
        ("Sen. Marsha Blackburn", 5),
        ("Sen. Roger Marshall", 6),
        ("Sen. Patrick Toomey", 7),
        ("Sen. Cynthia Lummis", 8),
        ("Sen. James Lankford", 8),
        ("Sen. Tim Scott", 10),
        ("Sen. Jim Risch", 11),
        ("Sen. Ted Cruz", 12),
        ("Sen. Tommy Tuberville", 13),
        ("Sen. Michael Crapo", 14),
        ("Sen. Steve Daines", 14),
        ("Sen. Bill Hagerty", 16),
        ("Sen. Kevin Cramer", 17),
        ("Sen. Deb Fischer", 18),
        ("Sen. John Hoeven", 18),
        ("Sen. Rick Scott", 18),
        ("Sen. Josh Hawley", 21),
        ("Sen. John Barrasso", 22),
        ("Sen. Marco Rubio", 23),
        ("Sen. John Thune", 24),
        ("Sen. Ben Sasse", 25),
        ("Sen. Tom Cotton", 26),
        ("Sen. Mike Rounds", 27),
        ("Sen. Joni Ernst", 28),
        ("Sen. James Inhofe", 29),
        ("Sen. John Cornyn", 30),
        ("Sen. John Boozman", 31),
        ("Sen. Dan Sullivan", 31),
        ("Sen. Cindy Hyde-Smith", 33),
        ("Sen. John Kennedy", 33),
        ("Sen. Charles Grassley", 35),
        ("Sen. Richard Shelby", 36),
        ("Sen. Willard Romney", 37),
        ("Sen. Thom Tillis", 38),
        ("Sen. Todd Young", 39),
        ("Sen. Jerry Moran", 40),
        ("Sen. Rob Portman", 41),
        ("Sen. Roger Wicker", 41),
        ("Sen. Bill Cassidy", 43),
        ("Sen. Shelley Capito", 44),
        ("Sen. Mitch McConnell", 45),
        ("Sen. Lindsey Graham", 46),
        ("Sen. Lisa Murkowski", 47),
        ("Sen. Roy Blunt", 48),
        ("Sen. Susan Collins", 49),
        ("Sen. Joe Manchin", 50),
        ("Sen. Kyrsten Sinema", 51),
        ("Sen. Jon Tester", 51),
        ("Sen. Mark Kelly", 53),
        ("Sen. Catherine Cortez Masto", 54),
        ("Sen. Margaret Hassan", 55),
        ("Sen. Jacky Rosen", 56),
        ("Sen. Amy Klobuchar", 57),
        ("Sen. Angus King", 57),
        ("Sen. Tina Smith", 57),
        ("Sen. Patrick Leahy", 60),
        ("Sen. Dianne Feinstein", 61),
        ("Sen. Michael Bennet", 62),
        ("Sen. Robert Menendez", 63),
        ("Sen. Raphael Warnock", 63),
        ("Sen. Cynthia Shaheen", 65),
        ("Sen. John Hickenlooper", 65),
        ("Sen. Chris Coons", 67),
        ("Sen. Thomas Carper", 67),
        ("Sen. Jon Ossoff", 67),
        ("Sen. Debbie Stabenow", 67),
        ("Sen. Cory Booker", 67),
        ("Sen. Ron Wyden", 67),
        ("Sen. Bob Casey", 67),
        ("Sen. Mark Warner", 67),
        ("Sen. Tim Kaine", 67),
        ("Sen. Bernie Sanders", 67),
        ("Sen. Patty Murray", 67),
        ("Sen. Maria Cantwell", 67),
        ("Sen. Ben Ray Luján", 80),
        ("Sen. Jeff Merkley", 80),
        ("Sen. Chris Van Hollen", 82),
        ("Sen. Alex Padilla", 83),
        ("Sen. Sherrod Brown", 83),
        ("Sen. Christopher Murphy", 85),
        ("Sen. Ladda Duckworth", 85),
        ("Sen. Martin Heinrich", 85),
        ("Sen. Tammy Baldwin", 85),
        ("Sen. Chuck Schumer", 89),
        ("Sen. Richard Blumenthal", 90),
        ("Sen. Brian Schatz", 90),
        ("Sen. Mazie Hirono", 90),
        ("Sen. Dick Durbin", 90),
        ("Sen. Elizabeth Warren", 90),
        ("Sen. Ed Markey", 90),
        ("Sen. Ben Cardin", 90),
        ("Sen. Gary Peters", 90),
        ("Sen. Kirsten Gillibrand", 90),
        ("Sen. Sheldon Whitehouse", 90),
        ("Sen. Jack Reed", 90),
    ]

    # Create dataframe and filter to only include available embeddings
    senator_df = pd.DataFrame(senator_data, columns=["Senator", "actual_rank"])
    filtered_senator_df = senator_df[senator_df["Senator"].isin(rank_df["Senator"])].reset_index(drop=True)

    # Reorder rankings while preserving ties
    filtered_senator_df["actual_rank"] = filtered_senator_df["actual_rank"].rank(method="dense").astype(int)

    # Rank provided dataframe
    rank_df["predicted_rank"] = rank_df["Magnitude"].round(3).rank(method="dense", ascending=True).astype(int)


    merged_dataframe = rank_df.merge(filtered_senator_df, on="Senator")

    # Compute correlation
    spearman_corr, _ = spearmanr(merged_dataframe['actual_rank'], merged_dataframe['predicted_rank'])
    kendall_corr, _ = kendalltau(merged_dataframe['actual_rank'], merged_dataframe['predicted_rank'])

    # Compute error metrics
    mae_rank = mean_absolute_error(merged_dataframe['actual_rank'], merged_dataframe['predicted_rank'])
    rmse_rank = np.sqrt(mean_squared_error(merged_dataframe['actual_rank'], merged_dataframe['predicted_rank']))

    # Percentage of perfect matches
    perfect_matches = (merged_dataframe['actual_rank'] == merged_dataframe['predicted_rank']).sum()
    total_senators = len(merged_dataframe)
    percent_perfect = (perfect_matches / total_senators) * 100

    results = {
        "Spearman_Correlation": spearman_corr,
        "Kendall_Tau": kendall_corr,
        "MAE_Rank": mae_rank,
        "RMSE_Rank": rmse_rank,
        "Percent_Perfect_Matches": percent_perfect,
    }
    
    #log results
    results_path = OUTPUT_DIR / "ranking_results.txt"
    with open(results_path, "a") as file:
        file.write(f"\nIdeology Project Model Results:\n")
        file.write(f"        Spearman_Correlation: {results['Spearman_Correlation']}\n")
        file.write(f"        Kendall_Tau: {results['Kendall_Tau']}\n")
        file.write(f"        MAE_Rank: {results['MAE_Rank']}\n")
        file.write(f"        RMSE_Rank: {results['RMSE_Rank']}\n")
        file.write(f"        Percent_Perfect_Matches: {results['Percent_Perfect_Matches']}\n")
    logger.success(f"Results saved to {results_path}")

    # Return results as dictionary
    return results







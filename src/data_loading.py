"""
Data loading utilities for Steam interactions and games metadata.
Functions are intentionally flexible so column names can be tweaked in config.py after EDA.
"""

from typing import Tuple

import pandas as pd


def load_interactions(path: str) -> pd.DataFrame:
    """
    Load the user–game interactions dataset from the given CSV path.
    Prints basic info for a quick sanity check.
    """
    interactions = pd.read_csv(path)
    print("Loaded interactions:", interactions.shape)
    print("Columns:", interactions.columns.tolist())
    print(interactions.head())
    return interactions


def load_games_metadata(path: str) -> pd.DataFrame:
    """
    Load the games metadata dataset (Steam games).
    Prints basic info for a quick sanity check.
    """
    games = pd.read_csv(path)
    print("Loaded games metadata:", games.shape)
    print("Columns:", games.columns.tolist())
    print(games.head())
    return games


def merge_datasets(
    interactions: pd.DataFrame,
    games: pd.DataFrame,
    user_col: str,
    item_col: str,
    game_id_col_in_games: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and merge interactions and games data.
    - Standardizes item identifiers so they match across datasets.
    - Filters interactions to items present in the games metadata.
    - Returns cleaned interactions and games metadata.
    """
    games_clean = games.copy()
    interactions_clean = interactions.copy()

    # Align item IDs
    if item_col != game_id_col_in_games:
        games_clean = games_clean.rename(columns={game_id_col_in_games: item_col})

    # Drop rows with missing critical identifiers
    interactions_clean = interactions_clean.dropna(subset=[user_col, item_col])
    games_clean = games_clean.dropna(subset=[item_col])

    # Keep only interactions with known games
    valid_items = set(games_clean[item_col].unique())
    interactions_clean = interactions_clean[interactions_clean[item_col].isin(valid_items)]

    # Optional de-duplication
    interactions_clean = interactions_clean.drop_duplicates(subset=[user_col, item_col])
    games_clean = games_clean.drop_duplicates(subset=[item_col])

    print(
        f"After merge alignment: {len(interactions_clean)} interactions, "
        f"{len(games_clean)} unique games"
    )
    return interactions_clean, games_clean

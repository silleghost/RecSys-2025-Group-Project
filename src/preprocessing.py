"""
Preprocessing helpers: filtering sparse users/items and building train/test splits.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def filter_users_and_items(
    interactions: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    user_col: str = "user_id",
    item_col: str = "game_id",
) -> pd.DataFrame:
    """
    Keep only users and items with at least the given number of interactions.
    Iteratively filters until all users/items satisfy the thresholds.
    """
    filtered = interactions.copy()
    while True:
        user_counts = filtered[user_col].value_counts()
        item_counts = filtered[item_col].value_counts()

        before = len(filtered)
        filtered = filtered[
            filtered[user_col].isin(user_counts[user_counts >= min_user_interactions].index)
            & filtered[item_col].isin(item_counts[item_counts >= min_item_interactions].index)
        ]
        if len(filtered) == before:
            break
    return filtered


def train_test_split_by_time(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str | None,
    n_test_items: int = 1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user:
    - sort interactions by timestamp and keep the last `n_test_items` as test;
    - if no timestamp is available, randomly sample n_test_items per user with a fixed seed.
    Returns train_df, test_df.
    """
    train_parts = []
    test_parts = []

    for _, user_df in interactions.groupby(user_col):
        if timestamp_col and timestamp_col in user_df.columns:
            user_df = user_df.sort_values(timestamp_col)
            test_samples = user_df.tail(n_test_items)
        else:
            test_samples = user_df.sample(
                n=min(n_test_items, len(user_df)), random_state=random_state
            )
        train_samples = user_df.drop(test_samples.index)

        # If user has fewer interactions than n_test_items, keep at least one in train
        if train_samples.empty and not test_samples.empty:
            test_samples = test_samples.iloc[:-1]
            train_samples = user_df.drop(test_samples.index)

        train_parts.append(train_samples)
        test_parts.append(test_samples)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df

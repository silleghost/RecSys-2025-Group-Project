"""
Central configuration for paths, column names, and defaults.
Update the column names after inspecting the raw Kaggle data in the EDA notebook.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File locations (adjust as needed)
INTERACTIONS_FILE = RAW_DATA_DIR / "reviews.csv"
GAMES_METADATA_FILE = RAW_DATA_DIR / "games_march2025_cleaned.csv"
GAME_ID_COL_IN_GAMES = "appid"  # item id column in games metadata

# Column names (replace with actual names after EDA)
USER_COL = "playerid"
ITEM_COL = "gameid"  # aligns with reviews.csv; games metadata uses `appid`
TIMESTAMP_COL = "posted"  # set to None if unavailable
INTERACTION_VALUE_COL = "helpful"  # not required for implicit models

# Minimum interaction thresholds
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5

# Random seed for reproducibility
RANDOM_STATE = 42

# Feature engineering defaults (override in notebooks if needed)
GENRE_COL = "genres"
TAGS_COL = "tags"
PRICE_COL = "price"
RATING_COL = "user_score"
RELEASE_DATE_COL = "release_date"

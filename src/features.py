"""
Item (game) feature construction helpers.
Builds numeric feature matrices from genres, tags, price, rating, and release date.
"""

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def _tokenize_cell(cell: object) -> List[str]:
    """Split a cell containing comma/semicolon-separated values into tokens."""
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        tokens = []
        for val in cell:
            tokens.extend(str(val).split(","))
        return [t.strip() for t in tokens if t.strip()]
    return [t.strip() for t in str(cell).replace(";", ",").split(",") if t.strip()]


def _tfidf_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Build TF-IDF features from a string/list column."""
    texts = series.fillna("").apply(lambda x: " ".join(_tokenize_cell(x)))
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    feature_names = [f"{prefix}{term}" for term in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=feature_names)


def build_item_features(
    games_df: pd.DataFrame,
    item_id_col: str,
    genre_col: str | None,
    tags_col: str | None = None,
    price_col: str | None = None,
    rating_col: str | None = None,
    release_date_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build numeric feature matrix for items (games).
    Suggested features:
    - TF-IDF for genres and tags,
    - normalized price (and flag for free-to-play),
    - normalized rating,
    - release year.
    Returns:
    - item_features: DataFrame with one row per item and numeric columns only,
    - item_meta: original subset with item_id_col and a few human-readable fields.
    """
    games_df = games_df.copy()
    features = []

    # Genres
    if genre_col and genre_col in games_df.columns:
        genre_df = _tfidf_features(games_df[genre_col], prefix="genre::")
        features.append(genre_df)

    # Tags
    if tags_col and tags_col in games_df.columns:
        tags_df = _tfidf_features(games_df[tags_col], prefix="tag::")
        features.append(tags_df)

    # Price
    if price_col and price_col in games_df.columns:
        price = pd.to_numeric(games_df[price_col], errors="coerce").fillna(0)
        scaler = MinMaxScaler()
        price_norm = scaler.fit_transform(price.values.reshape(-1, 1))
        features.append(pd.DataFrame(price_norm, columns=["price_norm"]))
        features.append(pd.DataFrame((price == 0).astype(int), columns=["is_free"]))

    # Rating
    if rating_col and rating_col in games_df.columns:
        rating = pd.to_numeric(games_df[rating_col], errors="coerce").fillna(0)
        scaler = MinMaxScaler()
        rating_norm = scaler.fit_transform(rating.values.reshape(-1, 1))
        features.append(pd.DataFrame(rating_norm, columns=["rating_norm"]))

    # Release year
    if release_date_col and release_date_col in games_df.columns:
        year = pd.to_datetime(games_df[release_date_col], errors="coerce").dt.year.fillna(0)
        features.append(pd.DataFrame(year.astype(int), columns=["release_year"]))

    if features:
        feature_matrix = pd.concat(features, axis=1)
    else:
        feature_matrix = pd.DataFrame(index=games_df.index)

    feature_matrix.insert(0, item_id_col, games_df[item_id_col].values)
    item_features = feature_matrix

    item_meta_cols = [c for c in [item_id_col, genre_col, tags_col, price_col, rating_col, release_date_col] if c]
    item_meta = games_df[item_meta_cols].copy()
    return item_features, item_meta

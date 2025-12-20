"""
Shared evaluation utilities for top-N recommendation.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Function that returns top-k item ids for a user_id
RecommenderFn = Callable[[int, int], List[int]]


def _dcg(relevances: List[int]) -> float:
    return float(np.sum((2**np.array(relevances) - 1) / np.log2(np.arange(2, len(relevances) + 2))))


def hit_rate_at_k(
    ground_truth: Dict[int, List[int]],
    recommendations: Dict[int, List[int]],
    k: int,
) -> float:
    """Compute HitRate@k over users."""
    hits = 0
    total = 0
    for user, true_items in ground_truth.items():
        recs = recommendations.get(user, [])[:k]
        if not true_items:
            continue
        hits += int(any(item in recs for item in true_items))
        total += 1
    return hits / total if total > 0 else 0.0


def recall_at_k(
    ground_truth: Dict[int, List[int]],
    recommendations: Dict[int, List[int]],
    k: int,
) -> float:
    """Compute Recall@k over users."""
    recalls = []
    for user, true_items in ground_truth.items():
        if not true_items:
            continue
        recs = recommendations.get(user, [])[:k]
        hit_count = len(set(true_items) & set(recs))
        recalls.append(hit_count / len(true_items))
    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    ground_truth: Dict[int, List[int]],
    recommendations: Dict[int, List[int]],
    k: int,
) -> float:
    """Compute NDCG@k over users."""
    ndcgs = []
    for user, true_items in ground_truth.items():
        if not true_items:
            continue
        recs = recommendations.get(user, [])[:k]
        relevances = [1 if item in true_items else 0 for item in recs]
        ideal_relevances = [1] * min(len(true_items), k)
        dcg = _dcg(relevances)
        idcg = _dcg(ideal_relevances)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def evaluate_topk(
    ground_truth: Dict[int, List[int]],
    recommend_fn: RecommenderFn,
    users: List[int],
    ks: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    For each k in ks, compute HitRate@k, Recall@k, NDCG@k by calling recommend_fn(user_id, k).
    Returns a DataFrame with metrics per k.
    """
    recommendations = {
        user: recommend_fn(user, max(ks))  # generate up to the largest k once
        for user in users
    }

    rows = []
    for k in ks:
        metrics = {
            "k": k,
            "hit_rate": hit_rate_at_k(ground_truth, recommendations, k),
            "recall": recall_at_k(ground_truth, recommendations, k),
            "ndcg": ndcg_at_k(ground_truth, recommendations, k),
        }
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_model(
    model: object,
    ground_truth: Dict[int, List[int]],
    users: List[int],
    ks: List[int] = [5, 10, 20],
    known_items: Dict[int, List[int]] | None = None,
    exclude_known: bool = True,
    recommend_fn: RecommenderFn | None = None,
) -> pd.DataFrame:
    """
    Uniform wrapper to evaluate any model with a `.recommend` method.
    The recommender is expected to support either:
    - recommend(user_id, k), or
    - recommend(user_id, known_items, k).
    Alternatively, pass a custom recommend_fn(user_id, k) to override model.recommend.
    """

    def _recommend(user_id: int, k: int) -> List[int]:
        known = known_items.get(user_id, []) if known_items else []
        if recommend_fn is not None:
            recs = recommend_fn(user_id, k)
        else:
            try:
                recs = model.recommend(user_id, known, k)
            except TypeError:
                recs = model.recommend(user_id, k)  # type: ignore[arg-type]
        if exclude_known and known:
            known_set = set(known)
            recs = [r for r in recs if r not in known_set]
        return recs[:k]

    return evaluate_topk(ground_truth, _recommend, users, ks=ks)


def build_ground_truth(
    interactions: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> Dict[int, List[int]]:
    """
    Build a mapping from user_id to list of held-out item_ids from a test interactions DataFrame.
    """
    return (
        interactions.groupby(user_col)[item_col]
        .apply(list)
        .to_dict()
    )


def make_recommend_fn_from_topk_model(topk_model: object, known_items: Dict[int, List[int]] | None = None) -> Callable[[int, int], List[int]]:
    """
    Adapter for models exposing `.recommend(user_id, top_k=k)` (and optionally `.user_map`).
    Returns a recommend_fn(user_id, k) compatible with evaluate_model.
    Known items are excluded if provided.
    """
    def recommend_fn(user_id: int, k: int) -> List[int]:
        if hasattr(topk_model, "user_map") and getattr(topk_model, "user_map"):
            if user_id not in topk_model.user_map:
                return []
        recs = topk_model.recommend(user_id, top_k=k)
        if known_items:
            recs = [r for r in recs if r not in set(known_items.get(user_id, []))]
        return recs[:k]

    return recommend_fn

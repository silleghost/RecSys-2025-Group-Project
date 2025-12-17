import numpy as np
import pandas as pd
import math
from tqdm import tqdm


def calculate_metrics(test_df, recommender_model, k=10):
    """
    Calculates HitRate@K, Recall@K, NDCG@K
    """
    known_users = set(recommender_model.user_map.keys())
    test_df_filtered = test_df[test_df['playerid'].isin(known_users)].copy()

    ground_truth = test_df_filtered.groupby('playerid')['appid'].apply(list).to_dict()

    hits = 0
    total_recall = 0
    total_ndcg = 0
    n_users = len(ground_truth)

    if n_users == 0:
        return {"Error": "No overlapping users in test set"}

    for user, actual_items in tqdm(ground_truth.items()):
        recs = recommender_model.recommend(user, top_k=k)

        hit = any(item in actual_items for item in recs)
        if hit:
            hits += 1

        intersect = set(recs).intersection(set(actual_items))
        recall = len(intersect) / len(actual_items) if len(actual_items) > 0 else 0
        total_recall += recall

        dcg = 0
        idcg = 0

        for i, item in enumerate(recs):
            if item in actual_items:
                dcg += 1 / math.log2((i + 1) + 1)

        num_relevant_in_top_k = min(len(actual_items), k)
        for i in range(num_relevant_in_top_k):
            idcg += 1 / math.log2((i + 1) + 1)

        ndcg = dcg / idcg if idcg > 0 else 0
        total_ndcg += ndcg

    return {
        f"HitRate@{k}": hits / n_users,
        f"Recall@{k}": total_recall / n_users,
        f"NDCG@{k}": total_ndcg / n_users
    }
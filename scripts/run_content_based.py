#!/usr/bin/env python
"""
Content-based recommender evaluation script for full-data/HPC runs.
- Enriched features (price/owners buckets + BM25 tags/categories/developers).
- Optional SVD for dimensionality reduction.
- Hybrid content+pop model evaluation with shared metrics.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import config
from src.evaluation import build_ground_truth, evaluate_model
from src.models.content_based import ContentHybridRecommender


def bm25_block(series: pd.Series, prefix: str, max_features: int, k1: float = 1.6, b: float = 0.75):
    texts = series.fillna("").astype(str).tolist()
    vec = CountVectorizer(max_features=max_features)
    X = vec.fit_transform(texts)
    tf = X
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avg_dl = dl.mean() + 1e-8
    idf = np.log((tf.shape[0] - tf.astype(bool).sum(axis=0) + 0.5) / (tf.astype(bool).sum(axis=0) + 0.5)) + 1
    idf = np.asarray(idf).ravel()
    denom = tf + k1 * (1 - b + b * (dl / avg_dl))[:, None]
    numer = tf.multiply(k1 + 1)
    bm25 = numer.multiply(1 / denom)
    bm25 = bm25.multiply(idf)
    names = [f"{prefix}{t}" for t in vec.get_feature_names_out()]
    return bm25.tocsr(), names


def build_feature_matrix(
    base_feats: pd.DataFrame,
    meta: pd.DataFrame,
    use_enriched: bool,
    use_svd: bool,
    svd_components: Optional[int],
    vocab_tags: int,
    vocab_categories: int,
    vocab_developers: int,
) -> Tuple[List[int], Dict[int, int], csr_matrix]:
    items = base_feats[config.ITEM_COL].astype(int).tolist()
    meta_aligned = meta.set_index(config.ITEM_COL).reindex(base_feats[config.ITEM_COL]).reset_index()

    blocks = []

    # Base dense features
    base_dense = csr_matrix(base_feats.drop(columns=[config.ITEM_COL]).to_numpy(dtype=np.float32))
    blocks.append(base_dense)

    if use_enriched:
        price_col = config.PRICE_COL
        if price_col in meta_aligned.columns:
            prices = pd.to_numeric(meta_aligned[price_col], errors="coerce").fillna(0)
            bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
            labels = [f"price_bin_{i}" for i in range(len(bins) - 1)]
            price_bins = pd.get_dummies(pd.cut(prices, bins=bins, labels=labels, include_lowest=True))
        else:
            price_bins = pd.DataFrame(index=meta_aligned.index)

        if "estimated_owners" in meta_aligned.columns:
            owners_raw = meta_aligned["estimated_owners"].fillna("")

            def parse_owner(val):
                if isinstance(val, str) and "-" in val:
                    try:
                        low = val.split("-")[0].replace(",", "").strip()
                        return float(low)
                    except Exception:
                        return np.nan
                try:
                    return float(val)
                except Exception:
                    return np.nan

            owners_num = owners_raw.apply(parse_owner)
            bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf]
            labels = [f"owners_bin_{i}" for i in range(len(bins) - 1)]
            owner_bins = pd.get_dummies(pd.cut(owners_num, bins=bins, labels=labels, include_lowest=True))
        else:
            owner_bins = pd.DataFrame(index=meta_aligned.index)

        extra_dense = pd.concat([price_bins, owner_bins], axis=1).fillna(0)
        blocks.append(csr_matrix(extra_dense.to_numpy(dtype=np.float32)))

        # BM25 text blocks
        if "categories" in meta_aligned.columns:
            mat, _ = bm25_block(meta_aligned["categories"], prefix="cat::", max_features=vocab_categories)
            blocks.append(mat)
        if "developers" in meta_aligned.columns:
            mat, _ = bm25_block(meta_aligned["developers"], prefix="dev::", max_features=vocab_developers)
            blocks.append(mat)
        if "tags" in meta_aligned.columns:
            mat, _ = bm25_block(meta_aligned["tags"], prefix="tag::", max_features=vocab_tags)
            blocks.append(mat)

    matrix = hstack(blocks).tocsr()

    if use_svd and svd_components:
        svd = TruncatedSVD(n_components=svd_components, random_state=config.RANDOM_STATE)
        matrix = svd.fit_transform(matrix)
        matrix = normalize(matrix)
        matrix = csr_matrix(matrix)

    matrix = normalize(matrix, norm="l2", axis=1)
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    return items, item_to_idx, matrix


def load_data(sample_users: Optional[int], min_interactions: int, seed: int):
    user_col = config.USER_COL
    item_col = config.ITEM_COL

    train = pd.read_parquet(config.PROCESSED_DATA_DIR / "train_interactions.parquet")
    test = pd.read_parquet(config.PROCESSED_DATA_DIR / "test_interactions.parquet")
    feats = pd.read_parquet(config.PROCESSED_DATA_DIR / "item_features.parquet").fillna(0)
    meta = pd.read_parquet(config.PROCESSED_DATA_DIR / "games_metadata.parquet")

    user_counts = train[user_col].value_counts()
    eligible = user_counts[user_counts >= min_interactions].index
    train = train[train[user_col].isin(eligible)]
    test = test[test[user_col].isin(eligible)]

    if sample_users:
        rng = np.random.default_rng(seed)
        picked = rng.choice(eligible, size=min(sample_users, len(eligible)), replace=False)
        train = train[train[user_col].isin(picked)]
        test = test[test[user_col].isin(picked)]

    items_in_split = set(train[item_col]) | set(test[item_col])
    feats = feats[feats[item_col].isin(items_in_split)].copy()
    meta = meta[meta[item_col].isin(items_in_split)].copy()

    return train, test, feats, meta


def main(
    alpha: float,
    min_interactions: int,
    sample_users: Optional[int],
    seed: int,
    use_svd: bool,
    svd_components: Optional[int],
    vocab_tags: int,
    vocab_categories: int,
    vocab_developers: int,
    run_lightfm: bool,
    lightfm_factors: int,
    lightfm_epochs: int,
    lightfm_loss: str,
    run_logreg: bool,
    logreg_neg_per_pos: int,
    logreg_max_users: Optional[int],
):
    user_col = config.USER_COL
    item_col = config.ITEM_COL

    train_df, test_df, feats, meta = load_data(sample_users, min_interactions, seed)

    item_ids, item_to_idx, item_matrix = build_feature_matrix(
        base_feats=feats,
        meta=meta,
        use_enriched=True,
        use_svd=use_svd,
        svd_components=svd_components,
        vocab_tags=vocab_tags,
        vocab_categories=vocab_categories,
        vocab_developers=vocab_developers,
    )

    # Popularity priors
    pop_counts = train_df[item_col].value_counts()
    pop_ranking = pop_counts.index.tolist()
    pop_scores = np.zeros(len(item_ids), dtype=np.float32)
    max_pop = pop_counts.max()
    for iid, count in pop_counts.items():
        idx = item_to_idx.get(iid)
        if idx is not None:
            pop_scores[idx] = count / max_pop

    ground_truth = build_ground_truth(test_df, user_col=user_col, item_col=item_col)
    users = list(ground_truth.keys())
    known = train_df.groupby(user_col)[item_col].apply(list).to_dict()

    model = ContentHybridRecommender(
        item_ids=item_ids,
        item_to_idx=item_to_idx,
        item_matrix=item_matrix,
        pop_scores=pop_scores,
        pop_ranking=pop_ranking,
        user_col=user_col,
        item_col=item_col,
        alpha=alpha,
    )
    model.fit(train_df)

    metrics = evaluate_model(model, ground_truth, users, ks=[5, 10, 20], known_items=known)
    metrics["model"] = f"content_alpha_{alpha}_svd_{svd_components}"
    print(metrics)

    if run_lightfm:
        try:
            from lightfm import LightFM
            from lightfm.data import Dataset as LFMDataset

            lfm_ds = LFMDataset()
            lfm_ds.fit(users=train_df[user_col].unique(), items=train_df[item_col].unique())
            interactions, _ = lfm_ds.build_interactions(train_df[[user_col, item_col]].itertuples(index=False, name=None))

            lfm_item_features = csr_matrix(item_matrix)
            user_id_map, user_feature_map, item_id_map, _ = lfm_ds.mapping()
            inv_item_map = {v: k for k, v in item_id_map.items()}

            lfmm = LightFM(loss=lightfm_loss, no_components=lightfm_factors, random_state=seed)
            lfmm.fit(interactions, item_features=lfm_item_features, epochs=lightfm_epochs, num_threads=4)

            class LFMWrapper:
                def recommend(self, uid: int, known_items: List[int], k: int) -> List[int]:
                    if uid not in user_id_map:
                        return []
                    uidx = user_id_map[uid]
                    scores = lfmm.predict(uidx, np.arange(len(inv_item_map)), item_features=lfm_item_features)
                    ranked = np.argsort(-scores)
                    recs = []
                    known_set = set(known_items)
                    for idx in ranked:
                        itm = inv_item_map[idx]
                        if itm in known_set:
                            continue
                        recs.append(itm)
                        if len(recs) >= k:
                            break
                    return recs

            lfm_wrapper = LFMWrapper()
            metrics_lfm = evaluate_model(lfm_wrapper, ground_truth, users, ks=[5, 10, 20], known_items=known)
            metrics_lfm["model"] = f"lightfm_{lightfm_loss}_f{lightfm_factors}_e{lightfm_epochs}"
            print(metrics_lfm)
        except Exception as e:
            print("LightFM failed or not installed:", e)

    if run_logreg:
        # Train logistic regression on sampled positives + negatives
        rng = np.random.default_rng(seed)
        user_profiles = {}
        for user, grp in train_df.groupby(user_col):
            idxs = [item_to_idx[i] for i in grp[item_col] if i in item_to_idx]
            if not idxs:
                continue
            profile = item_matrix[idxs].mean(axis=0)
            arr = np.asarray(profile).ravel()
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            user_profiles[user] = arr

        X_feat = []
        y = []
        users_iter = list(user_profiles.keys())
        if logreg_max_users:
            users_iter = users_iter[:logreg_max_users]

        all_items = np.array(item_ids)
        for user in tqdm(users_iter, desc="LogReg samples"):
            known_list = train_df[train_df[user_col] == user][item_col].tolist()
            known_set = set(known_list)
            profile = user_profiles[user]
            # positives
            for pos in known_list:
                idx = item_to_idx.get(pos)
                if idx is None:
                    continue
                content_score = float(item_matrix[idx].dot(profile))
                X_feat.append([content_score, pop_scores[idx]])
                y.append(1)
                # negatives
                neg_candidates = np.setdiff1d(all_items, np.array(list(known_set)), assume_unique=True)
                if len(neg_candidates) == 0:
                    continue
                neg_sample = rng.choice(neg_candidates, size=min(logreg_neg_per_pos, len(neg_candidates)), replace=False)
                for neg in neg_sample:
                    nidx = item_to_idx.get(int(neg))
                    if nidx is None:
                        continue
                    content_score_neg = float(item_matrix[nidx].dot(profile))
                    X_feat.append([content_score_neg, pop_scores[nidx]])
                    y.append(0)

        if X_feat:
            X_feat = np.array(X_feat, dtype=np.float32)
            y_arr = np.array(y, dtype=np.int8)
            clf = LogisticRegression(max_iter=200, n_jobs=4)
            clf.fit(X_feat, y_arr)

            def logreg_recommend(uid: int, known_items: List[int], k: int) -> List[int]:
                if uid not in user_profiles:
                    return []
                profile = user_profiles[uid]
                # compute scores in batches to limit memory
                batch_size = 50000
                scores = np.empty(len(item_ids), dtype=np.float32)
                for start in range(0, len(item_ids), batch_size):
                    end = min(start + batch_size, len(item_ids))
                    sub_idx = np.arange(start, end)
                    content_scores = np.asarray(item_matrix[sub_idx].dot(profile)).ravel()
                    pop_sub = pop_scores[sub_idx]
                    feats = np.stack([content_scores, pop_sub], axis=1)
                    proba = clf.predict_proba(feats)[:, 1]
                    scores[sub_idx] = proba
                known_set = set(known_items)
                for itm in known_items:
                    idx = item_to_idx.get(itm)
                    if idx is not None:
                        scores[idx] = -np.inf
                top_idx = np.argpartition(scores, -k)[-k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                return [item_ids[i] for i in top_idx]

            metrics_lr = evaluate_model(None, ground_truth, users, ks=[5, 10, 20], known_items=known, recommend_fn=logreg_recommend)
            metrics_lr["model"] = f"logreg_neg{logreg_neg_per_pos}_users{logreg_max_users or 'all'}_svd_{svd_components}"
            print(metrics_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.2, help="Content weight (0=popularity,1=content)")
    parser.add_argument("--min-interactions", type=int, default=config.MIN_USER_INTERACTIONS)
    parser.add_argument("--sample-users", type=int, default=None, help="Optional user sample for quick tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-svd", action="store_true", help="Enable TruncatedSVD")
    parser.add_argument("--svd-components", type=int, default=None, help="Number of SVD components when --use-svd is set")
    parser.add_argument("--vocab-tags", type=int, default=4000)
    parser.add_argument("--vocab-categories", type=int, default=2000)
    parser.add_argument("--vocab-developers", type=int, default=1000)
    parser.add_argument("--lightfm", action="store_true", help="Run LightFM hybrid with item features")
    parser.add_argument("--lightfm-factors", type=int, default=64)
    parser.add_argument("--lightfm-epochs", type=int, default=10)
    parser.add_argument("--lightfm-loss", type=str, default="warp", choices=["warp", "bpr", "logistic"])
    parser.add_argument("--logreg", action="store_true", help="Train/eval logistic regression scorer")
    parser.add_argument("--logreg-neg-per-pos", type=int, default=3, help="Negatives per positive for logreg training")
    parser.add_argument("--logreg-max-users", type=int, default=None, help="Cap users for logreg training (None=all)")
    args = parser.parse_args()
    main(
        alpha=args.alpha,
        min_interactions=args.min_interactions,
        sample_users=args.sample_users,
        seed=args.seed,
        use_svd=args.use_svd,
        svd_components=args.svd_components,
        vocab_tags=args.vocab_tags,
        vocab_categories=args.vocab_categories,
        vocab_developers=args.vocab_developers,
        run_lightfm=args.lightfm,
        lightfm_factors=args.lightfm_factors,
        lightfm_epochs=args.lightfm_epochs,
        lightfm_loss=args.lightfm_loss,
        run_logreg=args.logreg,
        logreg_neg_per_pos=args.logreg_neg_per_pos,
        logreg_max_users=args.logreg_max_users,
    )

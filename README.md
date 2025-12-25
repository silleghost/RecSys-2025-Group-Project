# Steam Games Recommender: Project Overview

## Problem statement

We build a recommender system for Steam games that suggests titles each user is likely to engage with. The task uses implicit feedback (purchases) and item metadata to train and compare classic collaborative filtering, content-based filtering and neural recommenders. Evaluation focuses on top-N ranking quality (HitRate, Recall, NDCG).

## Dataset
  - Sources: [Gaming Profiles 2025 (Steam)](https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox) and [Steam Games Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset).
  - Key files used:
      - purchased_games.csv: playerid, library (list of purchased appids).
      - games_march2025_cleaned.csv: game metadata (appid, name, release_date, price, genres, tags, categories, developers,
        estimated_owners, ratings, playtime stats, etc.).
  - Size (after processing in this repo):
      - Interactions: ~9.1M train rows, ~44K test rows (purchases), users filtered by min interactions (default 5).
      - Items: ~89K games with metadata/features.
  - EDA/Preprocessing (notebooks/01_eda_and_preprocessing.ipynb):
      - Explode purchase lists into (playerid, appid) interactions.
      - Align IDs with metadata (appid), drop missing IDs, filter users/items by min interactions.
      - Train/test split: per-user holdout of last item (or random fallback if no timestamp).
      - Save processed data to data/processed/: train_interactions.parquet, test_interactions.parquet, games_metadata.parquet.
  - Feature engineering (notebooks/02_feature_engineering.ipynb):
      - Base item features: TF-IDF over genres/tags, normalized price, user_score, release year.
      - Enriched features in later notebooks/scripts: BM25-weighted text (tags, categories, developers), bucketed price and
        estimated_owners, optional SVD for dimensionality reduction.
      - Saved to data/processed/item_features.parquet and item_meta.parquet.

  ## Methods

 ### Collaborative filtering (notebooks/03_cf_baselines_and_evaluation.ipynb):
  - User-based CF: similarity over user interaction profiles, recommends items from nearest users, excluding seen items, evaluated with shared HitRate/Recall/NDCG.
  - Item-based CF: similarity over item interaction profiles (cosine), recommends items similar to those a user interacted with, evaluated with the same metrics.

 ### Content-based filtering (notebooks/04_content_based_experiments.ipynb):
  - Popularity: rank by global interaction counts, filter seen items.
  - Content+pop hybrids: user profiles from item content vectors (mean of BM25/SVD features), cosine scores blended with popularity (parameter alpha controls blend).
  - Feature-kNN: precomputed item–item cosine neighbors on content features, recommend by aggregating neighbor sims from a user’s items.
  - LightFM hybrid: WARP/BPR with item content features plus implicit interactions.
  - Logistic regression scorer: pointwise model on content+pop features with negative sampling, ranks by predicted probability.
  - Main evaluation script: src/evaluation.py computes HitRate/Recall/NDCG@K with seen-item filtering

  ### Neural / transformer-based (notebooks/05-transformer-based.ipynb):
  - SBERT + KNN: semantic embeddings from SBERT on concatenated metadata fields, KNN over embeddings for content-only recommendations.
  - Two-tower transformer: item tower uses SBERT+metadata embeddings (PCA-compressed), user tower processes purchase sequences via Transformer, trained with BPR loss and negative sampling, inference via dot-product ranking of precomputed item vectors.

## Results
Top models on a 2K-user sample (min 10 interactions), sorted by NDCG@10:

| Model                | NDCG@10 | HitRate@10 | Recall@10 |
| -------------------- | ------- | ---------- | ---------- |
| hybrid_alpha_0.45    | 0.0779  | 0.1370     | 0.1370     |
| hybrid_alpha_0.40    | 0.0779  | 0.1355     | 0.1355     |
| hybrid_alpha_0.35    | 0.0775  | 0.1345     | 0.1345     |
| User-based CF        | 0.1263  | 0.2028     | 0.2028     |
| Item-based CF        | 0.1214  | 0.1932     | 0.1932     |
| Transformer-based    | 0.0157  | 0.0483     | 0.0483     |
| popularity (content) | 0.0701  | 0.1190     | 0.1190     |
| feature_kNN (sample) | 0.0093  | 0.0170     | 0.0170     |

## How to run

  - Environment: pip install -r requirements.txt.
  - Data: place raw CSVs in data/raw/, run notebooks in order for preprocessing/feature building, or use processed data already present.

## Conclusion

  - Content+pop hybrids with BM25/SVD features are strong classic baselines for this Steam purchase data, improving over popularity.
  - Feature-kNN and LightFM provide complementary signals, the logistic scorer offers a simple pointwise alternative.
  - Metrics are consistent across models via the shared evaluator, with single-item holdout.
  - Next steps: refine feature vocab/bins, tune alpha on validation, consider larger LightFM/logreg sweeps on HPC.

## References

  - Data: Steam metadata and interactions from Kaggle (data/raw/).
  - Methods: LightFM (https://making.lyst.com/lightfm/docs/), BM25 weighting, standard top-N metrics (HitRate/Recall/NDCG).

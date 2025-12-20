# Steam Games Recommender: Project Overview

  ## Problem statement

  We build a recommender system for Steam games that suggests titles each user is likely to engage with. The task uses
  implicit feedback (purchases) and item metadata to train and compare classic content-based and hybrid recommenders.
  Evaluation focuses on top-N ranking quality (HitRate, Recall, NDCG).

  ## Dataset

  - Source: Local Kaggle-style Steam data placed in data/raw/.
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

  - Popularity: rank by global interaction counts, filter seen items.
  - Content+pop hybrids: user profiles from item content vectors (mean of BM25/SVD features), cosine scores blended with
    popularity (alpha controls blend).
  - Feature-kNN: precomputed item–item cosine neighbors on content features; recommend by aggregating neighbor sims from a
    user’s items.
  - LightFM hybrid (optional): WARP/BPR with item content features plus implicit interactions; factors/epochs/loss swept in
    notebook and script.
  - Logistic regression scorer: pointwise model on content+pop features with negative sampling; ranks by predicted
    probability.
  - Shared evaluation: src/evaluation.py computes HitRate/Recall/NDCG@K with seen-item filtering; adapters provided for
    different model APIs.

  ## Results (sampled runs in notebooks/04_content_based_experiments.ipynb)

  - On a 2K-user sample (≥10 interactions), BM25+SVD hybrids outperform pure popularity; LightFM and logistic provide
    additional baselines.
  - With one held-out item per user, HitRate and Recall coincide; NDCG differentiates ranking quality.
  - Top models are reported in the notebook summary (NDCG@10 table).

  ## How to run

  - Environment: pip install -r requirements.txt (use .venv).
  - Data: place raw CSVs in data/raw/, run notebooks in order for preprocessing/feature building, or use processed data
    already present.
  - Notebook experiments (sampled for speed): notebooks/04_content_based_experiments.ipynb (BM25/SVD hybrids, kNN, LightFM,
    logistic).
  - HPC/full-data runs: scripts/run_content_based.py with flags:
      - Hybrids: --alpha {0.15..0.35} --use-svd --svd-components {128,256}
      - LightFM: add --lightfm --lightfm-factors {64,128} --lightfm-epochs 10 --lightfm-loss warp
      - Logistic: add --logreg --logreg-neg-per-pos 3 --logreg-max-users 50000
      - Common: --min-interactions 5 --vocab-tags 4000 --vocab-categories 2000 --vocab-developers 1000
      - Use sbatch arrays to sweep params; set OMP_NUM_THREADS/MKL_NUM_THREADS as needed.

  ## Conclusion

  - Content+pop hybrids with BM25/SVD features are strong classic baselines for this Steam purchase data, improving over
    popularity.
  - Feature-kNN and LightFM provide complementary signals; the logistic scorer offers a simple pointwise alternative.
  - Metrics are consistent across models via the shared evaluator; with single-item holdout, consider multi-item holdout for
    richer Recall/HitRate separation.
  - Next steps: refine feature vocab/bins, tune alpha on validation, consider larger LightFM/logreg sweeps on HPC.

  ## References

  - Data: Steam metadata and interactions from provided Kaggle-style dumps (data/raw/).
  - Methods: LightFM (https://making.lyst.com/lightfm/docs/), BM25 weighting, standard top-N metrics (HitRate/Recall/NDCG).

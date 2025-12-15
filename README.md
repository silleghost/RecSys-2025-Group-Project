# Steam Games Recommender Systems Project

Reproducible coursework project for a 3‑person team on Steam game recommendation. It sets up data loading, preprocessing, feature engineering, shared evaluation, and several recommender baselines plus templates for teammates to extend.

## Repository layout
- `data/raw/` — put raw Kaggle CSVs here (user–game interactions and games metadata).
- `data/processed/` — outputs from preprocessing and feature engineering.
- `notebooks/01_eda_and_preprocessing.ipynb` — inspect raw data, choose column names, filter, and split train/test.
- `notebooks/02_feature_engineering.ipynb` — build item (content) feature vectors.
- `notebooks/03_cf_baselines_and_evaluation.ipynb` — classical recommenders: popularity, item-based CF, implicit MF.
- `notebooks/04_content_based_template.ipynb` — template for teammate 1 to refine content-based model.
- `notebooks/05_neural_cf_template.ipynb` — template for teammate 2 to refine a neural CF model.
- `src/` — reusable Python modules (config, data loading, preprocessing, features, evaluation, models).

## Getting started
1. Install dependencies: `pip install -r requirements.txt`.
2. Place downloaded Kaggle CSVs into `data/raw/`.
3. Run notebooks in order:
   - `01_eda_and_preprocessing` to inspect raw data, decide `USER_COL`, `ITEM_COL`, `TIMESTAMP_COL`, `INTERACTION_VALUE`, and set them in `src/config.py`. Saves cleaned train/test splits to `data/processed/`.
   - `02_feature_engineering` to build item features and save them to `data/processed/`.
   - `03_cf_baselines_and_evaluation` to train/evaluate popularity, item-based CF, and implicit MF using shared evaluation helpers.
4. Teammates can start from:
   - `04_content_based_template` (content-based recommender).
   - `05_neural_cf_template` (neural CF). Both reuse the shared evaluation pipeline from `src/evaluation.py`.

## Implemented models
- Popularity baseline.
- Item-based collaborative filtering.
- Implicit matrix factorization for implicit feedback.

Planned/extendable:
- Content-based recommender (template provided).
- Neural collaborative filtering (template provided).

## Notes for the team
- All key paths and column names live in `src/config.py` for easy adjustment after inspecting the datasets.
- Follow the project guidance in `Eng-IndGroupTask.pdf` for scope, evaluation, and reporting expectations.

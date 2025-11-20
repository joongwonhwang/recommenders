"""Collaborative filtering recommender for AMOREPACIFIC beauty products."""

import numpy as np
import pandas as pd
from recommenders.models.surprise.surprise_utils import compute_ranking_predictions
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate


class BeautyProductRecommender:
    def __init__(self, n_factors: int = 50, n_epochs: int = 20):
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42)
        self.trainset = None

    def fit(self, ratings_df: pd.DataFrame):
        """Fit SVD on user-product ratings.

        Args:
            ratings_df: DataFrame with columns [user_id, product_sku, rating]
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[["user_id", "product_sku", "rating"]], reader)
        self.trainset = data.build_full_trainset()
        self.algo.fit(self.trainset)
        cv_results = cross_validate(self.algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
        print(f"CV RMSE: {cv_results['test_rmse'].mean():.4f}")

    def recommend(self, user_id: str, all_skus: list[str], top_k: int = 5) -> list[dict]:
        """Return top-k product recommendations for a user."""
        preds = [(sku, self.algo.predict(user_id, sku).est) for sku in all_skus]
        preds.sort(key=lambda x: x[1], reverse=True)
        return [{"sku": sku, "score": round(score, 3)} for sku, score in preds[:top_k]]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    users = [f"U{i:04d}" for i in range(200)]
    skus = [f"SKU{i:03d}" for i in range(50)]
    ratings_df = pd.DataFrame({
        "user_id": rng.choice(users, 1000),
        "product_sku": rng.choice(skus, 1000),
        "rating": rng.integers(1, 6, 1000),
    })
    rec = BeautyProductRecommender()
    rec.fit(ratings_df)
    print(rec.recommend("U0001", skus))

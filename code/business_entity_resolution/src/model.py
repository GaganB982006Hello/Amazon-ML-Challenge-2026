"""Supervised pair classification models for Amazon ML Challenge 2026.

Implements:
- Model A: Calibrated rule-based similarity baseline
- Model B: Regularized Logistic Regression pair classifier
- Model C: LightGBM gradient boosted decision tree classifier
Includes hard-negative mining from blocking candidates and controlled negative sampling.
"""

import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, PairFeatureExtractor


class HeuristicSimilarityModel:
    """Model A: Weighted heuristic similarity baseline."""

    def __init__(self):
        self.weights = {
            "name_token_set_ratio": 0.35,
            "name_ratio": 0.20,
            "addr_token_set_ratio": 0.25,
            "addr_numbers_jaccard": 0.15,
            "country_exact_match": 0.05,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute heuristic matching score from features."""
        # Find column indices
        col_map = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
        scores = np.zeros(len(X), dtype=np.float32)

        for feat_name, w in self.weights.items():
            if feat_name in col_map:
                scores += w * X[:, col_map[feat_name]]

        # Country mismatch penalization
        if "country_exact_match" in col_map:
            c_match = X[:, col_map["country_exact_match"]]
            scores = scores * c_match

        # Normalize score into probability [0, 1]
        probs = np.clip(scores, 0.0, 1.0)
        return probs


class LogisticRegressionPairModel:
    """Model B: Scaled L2-regularized Logistic Regression classifier."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit scaler and logistic regression classifier."""
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of match."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]


class LightGBMPairModel:
    """Model C: LightGBM gradient boosted decision tree classifier."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42,
    ):
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=-1,
            importance_type="gain",
            objective="binary",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit LightGBM model."""
        self.clf.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict matching probability."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.clf.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dict of feature names to gain importances."""
        if not self.is_fitted:
            return {}
        imp = self.clf.feature_importances_
        return {name: float(imp[i]) for i, name in enumerate(FEATURE_NAMES)}


def build_training_pairs_and_labels(
    s1_ids: List[str],
    candidates_map: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    negative_to_positive_ratio: int = 5,
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Build positive and hard-negative training pairs from blocking candidates.

    Hard negatives come directly from blocking candidates that do not appear in ground truth.
    """
    rng = random.Random(random_seed)
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []

    pos_count = 0
    neg_count = 0

    for s1_id in s1_ids:
        true_set = ground_truth.get(s1_id, set())
        cand_set = candidates_map.get(s1_id, set())

        # Positives: All true matches that were retrieved by blocking
        # (plus any true matches we want to guarantee the model sees)
        true_in_cands = true_set & cand_set
        for tid in true_in_cands:
            pairs.append((s1_id, tid))
            labels.append(1)
            pos_count += 1

        # Hard Negatives: Candidates generated by blocking that are NOT true matches
        hard_negs = list(cand_set - true_set)
        if hard_negs:
            # Sample hard negatives with controlled ratio
            n_sample = max(1, len(true_in_cands) * negative_to_positive_ratio)
            if len(hard_negs) > n_sample:
                sampled_negs = rng.sample(hard_negs, n_sample)
            else:
                sampled_negs = hard_negs

            for tid in sampled_negs:
                pairs.append((s1_id, tid))
                labels.append(0)
                neg_count += 1

    # Shuffle training pairs
    combined = list(zip(pairs, labels))
    rng.shuffle(combined)
    pairs = [p for p, _ in combined]
    labels_arr = np.array([l for _, l in combined], dtype=np.int32)

    print(f"Constructed training dataset: {len(pairs):,} pairs ({pos_count:,} positive, {neg_count:,} hard negative)")
    return pairs, labels_arr

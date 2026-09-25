"""Configuration module for Amazon ML Challenge 2026 Business Entity Resolution."""

from dataclasses import dataclass, field
import os
from typing import List, Optional


@dataclass
class PathConfig:
    """Paths to datasets, reports, outputs, and artifacts."""
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def __post_init__(self) -> None:
        """Use the canonical dataset directory or the extracted bundle fallback."""
        canonical_dir = os.path.join(self.base_dir, "dataset")
        fallback_dir = os.path.join(
            self.base_dir,
            "6ab10eb3b23ba_student_resource",
            "student_resource",
            "dataset",
        )
        if not os.path.isfile(os.path.join(canonical_dir, "train", "train_source1.tsv")):
            if os.path.isfile(os.path.join(fallback_dir, "train", "train_source1.tsv")):
                self.train_dir = os.path.join(fallback_dir, "train")
                self.test_dir = os.path.join(fallback_dir, "test")
                self.train_source1 = os.path.join(self.train_dir, "train_source1.tsv")
                self.train_source2 = os.path.join(self.train_dir, "train_source2.tsv")
                self.train_source3 = os.path.join(self.train_dir, "train_source3.tsv")
                self.train_ground_truth = os.path.join(self.train_dir, "train_ground_truth.tsv")
                self.test_source1 = os.path.join(self.test_dir, "test_source1.tsv")
                self.test_source2 = os.path.join(self.test_dir, "test_source2.tsv")
                self.test_source3 = os.path.join(self.test_dir, "test_source3.tsv")

    # Dataset paths
    train_dir: str = os.path.join(base_dir, "dataset", "train")
    test_dir: str = os.path.join(base_dir, "dataset", "test")

    train_source1: str = os.path.join(train_dir, "train_source1.tsv")
    train_source2: str = os.path.join(train_dir, "train_source2.tsv")
    train_source3: str = os.path.join(train_dir, "train_source3.tsv")
    train_ground_truth: str = os.path.join(train_dir, "train_ground_truth.tsv")

    test_source1: str = os.path.join(test_dir, "test_source1.tsv")
    test_source2: str = os.path.join(test_dir, "test_source2.tsv")
    test_source3: str = os.path.join(test_dir, "test_source3.tsv")

    # Output paths
    output_dir: str = os.path.join(base_dir, "output")
    matching_output: str = os.path.join(output_dir, "matching_results.tsv")
    candidate_output: str = os.path.join(output_dir, "candidate_pairs.tsv")

    # Reports
    reports_dir: str = os.path.join(base_dir, "reports")
    profiling_report: str = os.path.join(reports_dir, "profiling_report.md")
    blocking_report_json: str = os.path.join(reports_dir, "blocking_report.json")
    blocking_report_md: str = os.path.join(reports_dir, "blocking_report.md")
    experiments_csv: str = os.path.join(reports_dir, "experiments.csv")
    experiments_md: str = os.path.join(reports_dir, "experiments.md")
    validation_report_md: str = os.path.join(reports_dir, "validation_report.md")


@dataclass
class PipelineConfig:
    """Hyperparameters and configuration settings."""
    random_seed: int = 42

    # Validation split parameters
    val_sample_size: int = 15000  # Number of S1 entities for validation experiments
    train_sample_size: int = 60000  # Number of S1 entities for training pair model

    # Blocking parameters
    max_candidates_per_entity: int = 40
    min_token_length_for_block: int = 3
    tfidf_max_features: int = 50000
    tfidf_ngram_range: tuple = (3, 4)
    tfidf_top_k: int = 15

    # Negative sampling parameters
    negative_to_positive_ratio: int = 5  # Realistic difficult negatives ratio
    hard_negative_sample_ratio: float = 0.8  # 80% hard negatives from blocking, 20% random from candidates

    # Model parameters
    lgb_n_estimators: int = 300
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 31
    lgb_min_child_samples: int = 20

    # Decision threshold parameters
    threshold_search_start: float = 0.30
    threshold_search_end: float = 0.95
    threshold_search_step: float = 0.02
    default_threshold: float = 0.70

    # Source-specific thresholds (can be tuned on validation)
    s2_threshold: float = 0.70
    s3_threshold: float = 0.70

    # Singleton prediction margin
    singleton_score_margin: float = 0.15

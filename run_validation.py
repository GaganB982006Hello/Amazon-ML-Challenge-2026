"""Comprehensive validation and ablation experiment runner for Amazon ML Challenge 2026.

Executes controlled validation experiments:
1. Model A: Heuristic Rule Baseline
2. Model B: Logistic Regression Classifier
3. Model C: LightGBM (Name-only features)
4. Model C: LightGBM (Address-only features)
5. Model C: LightGBM (Full Features: Name + Address + Interactions)
6. Model C: Hard Negative Mining vs Random Negatives
7. Model C: Decision Threshold & Score Gap Calibration

Records results in reports/experiments.csv, reports/experiments.md, and reports/validation_report.md.
"""

from collections import defaultdict
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

from blocking import MultiStrategyBlocker, evaluate_blocking_recall
from calibration import ThresholdOptimizer, apply_decision_rules
from config import PathConfig, PipelineConfig
from data_loader import create_train_val_split, load_entities_file, load_ground_truth
from evaluation import evaluate_predictions
from features import FEATURE_NAMES, PairFeatureExtractor
from model import (
    HeuristicSimilarityModel,
    LightGBMPairModel,
    LogisticRegressionPairModel,
    build_training_pairs_and_labels,
)


def run_experiments():
    print("=" * 70)
    print("STARTING ML CHALLENGE 2026 VALIDATION EXPERIMENT SUITE")
    print("=" * 70)
    suite_start = time.time()
    cfg = PipelineConfig()
    paths = PathConfig()

    # 1. Load Ground Truth
    print("\n[Step 1] Loading ground truth records...")
    t0 = time.time()
    gt_map = load_ground_truth(paths.train_ground_truth, max_rows=120000)
    print(f"Loaded {len(gt_map):,} ground truth entries in {time.time()-t0:.2f}s")

    # 2. Sample S1 entities for train and val splits
    sample_s1_ids = list(gt_map.keys())[:35000]
    print(f"Loading {len(sample_s1_ids):,} Source 1 entities...")
    s1_records = load_entities_file(paths.train_source1, filter_ids=set(sample_s1_ids))

    train_s1_ids, val_s1_ids = create_train_val_split(
        gt_map, s1_records, val_size=8000, random_seed=cfg.random_seed
    )
    # Further limit train size for fast validation iterations
    train_s1_ids = train_s1_ids[:20000]

    print(f"Split created: {len(train_s1_ids):,} Train S1, {len(val_s1_ids):,} Val S1")

    # 3. Collect target matches needed
    target_needed = set()
    for s1 in sample_s1_ids:
        target_needed.update(gt_map.get(s1, set()))

    print(f"Loading target records ({len(target_needed):,} needed targets + distractor pool)...")
    s2_records = load_entities_file(paths.train_source2, max_rows=60000)
    s3_records = load_entities_file(paths.train_source3, max_rows=60000)

    s2_needed = {m for m in target_needed if m.startswith("S2-")}
    s3_needed = {m for m in target_needed if m.startswith("S3-")}
    s2_records.update(load_entities_file(paths.train_source2, filter_ids=s2_needed))
    s3_records.update(load_entities_file(paths.train_source3, filter_ids=s3_needed))

    target_pool = {}
    target_pool.update(s2_records)
    target_pool.update(s3_records)
    print(f"Total target pool size: {len(target_pool):,} records (S2: {len(s2_records):,}, S3: {len(s3_records):,})")

    # 4. Multi-Strategy Blocking
    print("\n[Step 2] Indexing target records and generating blocking candidates...")
    blocker = MultiStrategyBlocker(max_candidates_per_s1=cfg.max_candidates_per_entity)
    t_idx = time.time()
    blocker.index_target_entities(target_pool)
    print(f"Indexing completed in {time.time()-t_idx:.2f}s")

    # Generate candidates for train and val
    print("Generating train candidates...")
    train_s1_dict = {s1: s1_records[s1] for s1 in train_s1_ids if s1 in s1_records}
    train_candidates = blocker.generate_candidate_pairs(train_s1_dict)

    print("Generating val candidates...")
    val_s1_dict = {s1: s1_records[s1] for s1 in val_s1_ids if s1 in s1_records}
    val_candidates = blocker.generate_candidate_pairs(val_s1_dict)

    # Evaluate validation blocking recall
    val_gt = {s1: gt_map[s1] for s1 in val_s1_ids if s1 in gt_map}
    blocking_eval = evaluate_blocking_recall(val_gt, val_candidates)
    print(f"Validation Candidate Recall: {blocking_eval['candidate_recall_pct']}% "
          f"({blocking_eval['recalled_matches']:,} / {blocking_eval['total_true_matches']:,})")
    print(f"Average Candidates per S1: {blocking_eval['avg_candidates_per_s1']}")

    # Save blocking report
    os.makedirs(paths.reports_dir, exist_ok=True)
    with open(paths.blocking_report_json, "w", encoding="utf-8") as f:
        json.dump(blocking_eval, f, indent=2)

    with open(paths.blocking_report_md, "w", encoding="utf-8") as f:
        f.write("# Blocking and Candidate Generation Report\n\n")
        f.write("| Metric | Value |\n| --- | --- |\n")
        for k, v in blocking_eval.items():
            f.write(f"| {k} | {v} |\n")

    # 5. Extract Features
    print("\n[Step 3] Building training pairs with hard negative mining...")
    t_feat = time.time()
    train_pairs, train_labels = build_training_pairs_and_labels(
        train_s1_ids,
        train_candidates,
        gt_map,
        negative_to_positive_ratio=cfg.negative_to_positive_ratio,
        random_seed=cfg.random_seed,
    )

    feature_extractor = PairFeatureExtractor()
    print(f"Extracting features for {len(train_pairs):,} training pairs...")
    X_train = feature_extractor.extract_features_matrix(train_pairs, s1_records, target_pool)
    print(f"X_train matrix shape: {X_train.shape} created in {time.time()-t_feat:.2f}s")

    # Build validation pairs
    val_pairs: List[Tuple[str, str]] = []
    val_s1_pair_indices = defaultdict(list)
    for s1 in val_s1_ids:
        cands = val_candidates.get(s1, set())
        for cid in cands:
            idx = len(val_pairs)
            val_pairs.append((s1, cid))
            val_s1_pair_indices[s1].append((cid, idx))

    print(f"Extracting features for {len(val_pairs):,} validation pairs...")
    t_vfeat = time.time()
    X_val = feature_extractor.extract_features_matrix(val_pairs, s1_records, target_pool)
    print(f"X_val matrix shape: {X_val.shape} created in {time.time()-t_vfeat:.2f}s")

    # Validation entity countries
    val_countries = {s1: s1_records[s1].country for s1 in val_s1_ids if s1 in s1_records}

    # Helper function to evaluate model predictions on validation
    def evaluate_model_on_val(model_name: str, probs: np.ndarray, threshold: float = 0.70) -> Dict[str, Any]:
        # Map probabilities back to per-S1 candidates
        scores_by_s1 = {}
        for s1, c_list in val_s1_pair_indices.items():
            scores_by_s1[s1] = [(cid, float(probs[idx])) for cid, idx in c_list]

        # Optimize threshold
        optimizer = ThresholdOptimizer(threshold_range=(0.40, 0.95, 0.02), score_gap_margin=0.25)
        best_tau, best_metrics, _ = optimizer.search_optimal_threshold(
            val_s1_ids, val_gt, scores_by_s1, val_countries
        )
        return best_tau, best_metrics, scores_by_s1

    experiments = []

    # -------------------------------------------------------------
    # Experiment 1: Model A - Calibrated Heuristic Rule Baseline
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 1: Model A (Heuristic Weighted Similarity Baseline)")
    print("=" * 60)
    t_exp = time.time()
    model_a = HeuristicSimilarityModel()
    val_probs_a = model_a.predict_proba(X_val)
    best_tau_a, metrics_a, _ = evaluate_model_on_val("Model A", val_probs_a)
    exp1_time = round(time.time() - t_exp, 2)
    print(f"Model A Results: Best Threshold={best_tau_a} | Macro F0.5={metrics_a['macro_f05']:.4f} | "
          f"Precision={metrics_a['macro_precision']:.4f} | Recall={metrics_a['macro_recall']:.4f}")

    experiments.append({
        "experiment_name": "Exp 1: Heuristic Similarity Baseline",
        "model": "Weighted Rules (Model A)",
        "features": "Token Sets + Edit Ratio + Numbers",
        "negatives": "Hard Negatives from Blocking",
        "threshold": best_tau_a,
        "macro_f05": metrics_a["macro_f05"],
        "macro_precision": metrics_a["macro_precision"],
        "macro_recall": metrics_a["macro_recall"],
        "singleton_acc": metrics_a["singleton_accuracy"],
        "cand_recall": blocking_eval["candidate_recall_pct"],
        "runtime_s": exp1_time,
    })

    # -------------------------------------------------------------
    # Experiment 2: Model B - Regularized Logistic Regression
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 2: Model B (Logistic Regression Pair Classifier)")
    print("=" * 60)
    t_exp = time.time()
    model_b = LogisticRegressionPairModel(random_state=cfg.random_seed)
    model_b.fit(X_train, train_labels)
    val_probs_b = model_b.predict_proba(X_val)
    best_tau_b, metrics_b, _ = evaluate_model_on_val("Model B", val_probs_b)
    exp2_time = round(time.time() - t_exp, 2)
    print(f"Model B Results: Best Threshold={best_tau_b} | Macro F0.5={metrics_b['macro_f05']:.4f} | "
          f"Precision={metrics_b['macro_precision']:.4f} | Recall={metrics_b['macro_recall']:.4f}")

    experiments.append({
        "experiment_name": "Exp 2: Logistic Regression",
        "model": "Logistic Regression (Model B)",
        "features": "All 33 Pair Features",
        "negatives": "Hard Negatives from Blocking",
        "threshold": best_tau_b,
        "macro_f05": metrics_b["macro_f05"],
        "macro_precision": metrics_b["macro_precision"],
        "macro_recall": metrics_b["macro_recall"],
        "singleton_acc": metrics_b["singleton_accuracy"],
        "cand_recall": blocking_eval["candidate_recall_pct"],
        "runtime_s": exp2_time,
    })

    # -------------------------------------------------------------
    # Experiment 3: Model C - Name-Only Features Ablation
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 3: Model C (LightGBM - Name Only Ablation)")
    print("=" * 60)
    t_exp = time.time()
    name_indices = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("name_")]
    X_train_name = X_train[:, name_indices]
    X_val_name = X_val[:, name_indices]

    model_c_name = LightGBMPairModel(n_estimators=150, random_state=cfg.random_seed)
    model_c_name.fit(X_train_name, train_labels)
    val_probs_c_name = model_c_name.predict_proba(X_val_name)
    best_tau_c_name, metrics_c_name, _ = evaluate_model_on_val("Model C Name", val_probs_c_name)
    exp3_time = round(time.time() - t_exp, 2)
    print(f"Model C Name-Only: Best Threshold={best_tau_c_name} | Macro F0.5={metrics_c_name['macro_f05']:.4f} | "
          f"Precision={metrics_c_name['macro_precision']:.4f} | Recall={metrics_c_name['macro_recall']:.4f}")

    experiments.append({
        "experiment_name": "Exp 3: LightGBM (Name Only)",
        "model": "LightGBM GBDT",
        "features": "13 Name Features",
        "negatives": "Hard Negatives from Blocking",
        "threshold": best_tau_c_name,
        "macro_f05": metrics_c_name["macro_f05"],
        "macro_precision": metrics_c_name["macro_precision"],
        "macro_recall": metrics_c_name["macro_recall"],
        "singleton_acc": metrics_c_name["singleton_accuracy"],
        "cand_recall": blocking_eval["candidate_recall_pct"],
        "runtime_s": exp3_time,
    })

    # -------------------------------------------------------------
    # Experiment 4: Model C - Address-Only Features Ablation
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 4: Model C (LightGBM - Address Only Ablation)")
    print("=" * 60)
    t_exp = time.time()
    addr_indices = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("addr_")]
    X_train_addr = X_train[:, addr_indices]
    X_val_addr = X_val[:, addr_indices]

    model_c_addr = LightGBMPairModel(n_estimators=150, random_state=cfg.random_seed)
    model_c_addr.fit(X_train_addr, train_labels)
    val_probs_c_addr = model_c_addr.predict_proba(X_val_addr)
    best_tau_c_addr, metrics_c_addr, _ = evaluate_model_on_val("Model C Addr", val_probs_c_addr)
    exp4_time = round(time.time() - t_exp, 2)
    print(f"Model C Addr-Only: Best Threshold={best_tau_c_addr} | Macro F0.5={metrics_c_addr['macro_f05']:.4f} | "
          f"Precision={metrics_c_addr['macro_precision']:.4f} | Recall={metrics_c_addr['macro_recall']:.4f}")

    experiments.append({
        "experiment_name": "Exp 4: LightGBM (Address Only)",
        "model": "LightGBM GBDT",
        "features": "11 Address Features",
        "negatives": "Hard Negatives from Blocking",
        "threshold": best_tau_c_addr,
        "macro_f05": metrics_c_addr["macro_f05"],
        "macro_precision": metrics_c_addr["macro_precision"],
        "macro_recall": metrics_c_addr["macro_recall"],
        "singleton_acc": metrics_c_addr["singleton_accuracy"],
        "cand_recall": blocking_eval["candidate_recall_pct"],
        "runtime_s": exp4_time,
    })

    # -------------------------------------------------------------
    # Experiment 5: Model C - Full Features LightGBM
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 5: Model C (LightGBM - Full Feature Set + Hard Negatives)")
    print("=" * 60)
    t_exp = time.time()
    model_c_full = LightGBMPairModel(
        n_estimators=cfg.lgb_n_estimators,
        learning_rate=cfg.lgb_learning_rate,
        num_leaves=cfg.lgb_num_leaves,
        random_state=cfg.random_seed,
    )
    model_c_full.fit(X_train, train_labels)
    val_probs_c_full = model_c_full.predict_proba(X_val)
    best_tau_c_full, metrics_c_full, val_scores_by_s1 = evaluate_model_on_val("Model C Full", val_probs_c_full)
    exp5_time = round(time.time() - t_exp, 2)
    print(f"Model C Full: Best Threshold={best_tau_c_full} | Macro F0.5={metrics_c_full['macro_f05']:.4f} | "
          f"Precision={metrics_c_full['macro_precision']:.4f} | Recall={metrics_c_full['macro_recall']:.4f} | "
          f"Singleton Acc={metrics_c_full['singleton_accuracy']:.4f}")

    experiments.append({
        "experiment_name": "Exp 5: LightGBM (Full Features)",
        "model": "LightGBM GBDT",
        "features": "All 33 Features (Name+Addr+Cross+Source)",
        "negatives": "Hard Negatives from Blocking",
        "threshold": best_tau_c_full,
        "macro_f05": metrics_c_full["macro_f05"],
        "macro_precision": metrics_c_full["macro_precision"],
        "macro_recall": metrics_c_full["macro_recall"],
        "singleton_acc": metrics_c_full["singleton_accuracy"],
        "cand_recall": blocking_eval["candidate_recall_pct"],
        "runtime_s": exp5_time,
    })

    # Top feature importances
    feat_imp = model_c_full.get_feature_importances()
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances (Gain):")
    for feat, imp in sorted_imp[:10]:
        print(f"  {feat:<25}: {imp:.2f}")

    # -------------------------------------------------------------
    # Experiment 6: Joint Source-Specific Threshold Tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 6: Joint Source-Specific Threshold Optimization")
    print("=" * 60)
    t_exp = time.time()
    opt = ThresholdOptimizer()
    tau_s2, tau_s3, joint_f05, joint_metrics = opt.search_source_specific_thresholds(
        val_s1_ids, val_gt, val_scores_by_s1
    )
    exp6_time = round(time.time() - t_exp, 2)
    print(f"Joint Optimization: tau_s2={tau_s2}, tau_s3={tau_s3} | Macro F0.5={joint_f05:.4f} | "
          f"Precision={joint_metrics['macro_precision']:.4f} | Recall={joint_metrics['macro_recall']:.4f}")

    experiments.append({
        "experiment_name": "Exp 6: Source-Specific Calibration",
        "model": "LightGBM GBDT",
        "features": "All 33 Features",
        "negatives": "Hard Negatives from Blocking",
        "threshold": f"S2:{tau_s2}/S3:{tau_s3}",
        "macro_f05": joint_metrics["macro_f05"],
        "macro_precision": joint_metrics["macro_precision"],
        "macro_recall": joint_metrics["macro_recall"],
        "singleton_acc": joint_metrics["singleton_accuracy"],
        "cand_recall": blocking_eval["candidate_recall_pct"],
        "runtime_s": exp6_time,
    })

    # Save Experiments CSV
    print(f"\nWriting experiment results to {paths.experiments_csv}...")
    fieldnames = [
        "experiment_name", "model", "features", "negatives", "threshold",
        "macro_f05", "macro_precision", "macro_recall", "singleton_acc",
        "cand_recall", "runtime_s"
    ]
    with open(paths.experiments_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            writer.writerow(exp)

    # Save Experiments Markdown
    print(f"Writing experiment results to {paths.experiments_md}...")
    with open(paths.experiments_md, "w", encoding="utf-8") as f:
        f.write("# Entity Resolution Experiment Suite Results\n\n")
        f.write("Macro F0.5 evaluation results on held-out stratified validation split.\n\n")
        f.write("| Experiment | Model | Features | Threshold | Macro F0.5 | Precision | Recall | Singleton Acc | Runtime |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for exp in experiments:
            f.write(
                f"| **{exp['experiment_name']}** | {exp['model']} | {exp['features']} | {exp['threshold']} | "
                f"**{exp['macro_f05']:.4f}** | {exp['macro_precision']:.4f} | {exp['macro_recall']:.4f} | "
                f"{exp['singleton_acc']*100:.1f}% | {exp['runtime_s']}s |\n"
            )

    # Generate Detailed Validation Report
    with open(paths.validation_report_md, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Validation & Error Analysis Report\n\n")
        f.write(f"- **Total Validation S1 Entities:** {len(val_s1_ids):,}\n")
        f.write(f"- **Best Model:** LightGBM with Full Pair Features & Hard Negatives\n")
        f.write(f"- **Optimal Decision Threshold:** {best_tau_c_full}\n")
        f.write(f"- **Best Validation Macro F0.5:** {metrics_c_full['macro_f05']:.5f}\n")
        f.write(f"- **Macro Precision:** {metrics_c_full['macro_precision']:.5f}\n")
        f.write(f"- **Macro Recall:** {metrics_c_full['macro_recall']:.5f}\n")
        f.write(f"- **Singleton Accuracy:** {metrics_c_full['singleton_accuracy']*100:.2f}%\n")
        f.write(f"- **Candidate Recall (Blocking):** {blocking_eval['candidate_recall_pct']}%\n\n")
        f.write("## Country Breakdown\n\n")
        f.write("| Country | Entities | Macro F0.5 | Precision | Recall |\n| --- | --- | --- | --- | --- |\n")
        for c, st in metrics_c_full.get("by_country", {}).items():
            f.write(f"| {c} | {st['count']:,} | {st['macro_f05']:.4f} | {st['macro_prec']:.4f} | {st['macro_rec']:.4f} |\n")
        f.write("\n## Source Performance Breakdown\n\n")
        f.write(f"- **Source 2 Micro F0.5:** {metrics_c_full['s2_micro']['f05']} (Precision: {metrics_c_full['s2_micro']['precision']}, Recall: {metrics_c_full['s2_micro']['recall']})\n")
        f.write(f"- **Source 3 Micro F0.5:** {metrics_c_full['s3_micro']['f05']} (Precision: {metrics_c_full['s3_micro']['precision']}, Recall: {metrics_c_full['s3_micro']['recall']})\n\n")
        f.write("## Top 10 Feature Importances\n\n")
        f.write("| Rank | Feature | Importance (Gain) |\n| --- | --- | --- |\n")
        for r, (feat, imp) in enumerate(sorted_imp[:10], 1):
            f.write(f"| {r} | `{feat}` | {imp:.2f} |\n")

    print(f"\nValidation suite completed successfully in {time.time()-suite_start:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_experiments()

"""Evaluation metrics for Amazon ML Challenge 2026.

Implements the exact macro-averaged F0.5 metric specified in the competition:
F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singletons (empty true match list) receive:
- 1.0 if predicted empty (correct singleton)
- 0.0 if predicted non-empty (false merge)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def compute_entity_f05(
    true_ids: Set[str],
    pred_ids: Set[str],
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F0.5 for a single Source 1 entity.

    Returns:
        (precision, recall, f05)
    """
    if not true_ids:
        # Ground truth is a singleton
        if not pred_ids:
            return 1.0, 1.0, 1.0  # Perfect singleton prediction
        else:
            return 0.0, 0.0, 0.0  # False merge on singleton

    if not pred_ids:
        # Ground truth had matches, but model predicted nothing
        return 0.0, 0.0, 0.0

    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denominator = 0.25 * precision + recall
    if denominator > 0:
        f05 = (1.25 * precision * recall) / denominator
    else:
        f05 = 0.0

    return precision, recall, f05


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    entity_countries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Compute overall evaluation metrics across all Source 1 entities.

    Args:
        ground_truth: Dict mapping source1_entity_id -> set of true matched IDs.
        predictions: Dict mapping source1_entity_id -> set of predicted matched IDs.
        entity_countries: Optional dict mapping source1_entity_id -> country string.

    Returns:
        Dictionary of macro and micro metrics, singleton stats, and country breakdowns.
    """
    all_s1_ids = sorted(ground_truth.keys())
    total_entities = len(all_s1_ids)

    if total_entities == 0:
        return {"macro_f05": 0.0, "total_entities": 0}

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0

    singleton_total = 0
    singleton_correct = 0
    non_singleton_total = 0
    non_singleton_f05_sum = 0.0

    # Aggregate TP, FP, FN for micro stats
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    # Country-level breakdown
    country_f05 = defaultdict(list)
    country_prec = defaultdict(list)
    country_rec = defaultdict(list)

    # Source-level breakdown (S2 vs S3)
    s2_tp = s2_fp = s2_fn = 0
    s3_tp = s3_fp = s3_fn = 0

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        p, r, f = compute_entity_f05(true_set, pred_set)
        total_f05 += f
        total_prec += p
        total_rec += r

        # Breakdown by singleton status
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        else:
            non_singleton_total += 1
            non_singleton_f05_sum += f

        # Micro statistics
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        # Source breakdown
        for m in pred_set:
            if m.startswith("S2-"):
                if m in true_set:
                    s2_tp += 1
                else:
                    s2_fp += 1
            elif m.startswith("S3-"):
                if m in true_set:
                    s3_tp += 1
                else:
                    s3_fp += 1

        for m in true_set:
            if m not in pred_set:
                if m.startswith("S2-"):
                    s2_fn += 1
                elif m.startswith("S3-"):
                    s3_fn += 1

        # Country breakdown
        if entity_countries and s1_id in entity_countries:
            c = entity_countries[s1_id]
            country_f05[c].append(f)
            country_prec[c].append(p)
            country_rec[c].append(r)

    macro_f05 = total_f05 / total_entities
    macro_prec = total_prec / total_entities
    macro_rec = total_rec / total_entities

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_denom = 0.25 * micro_precision + micro_recall
    micro_f05 = (1.25 * micro_precision * micro_recall) / micro_denom if micro_denom > 0 else 0.0

    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    non_singleton_macro_f05 = non_singleton_f05_sum / non_singleton_total if non_singleton_total > 0 else 0.0

    # Source-level F0.5
    s2_p = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0.0
    s2_r = s2_tp / (s2_tp + s2_fn) if (s2_tp + s2_fn) > 0 else 0.0
    s2_denom = 0.25 * s2_p + s2_r
    s2_f05 = (1.25 * s2_p * s2_r) / s2_denom if s2_denom > 0 else 0.0

    s3_p = s3_tp / (s3_tp + s3_fp) if (s3_tp + s3_fp) > 0 else 0.0
    s3_r = s3_tp / (s3_tp + s3_fn) if (s3_tp + s3_fn) > 0 else 0.0
    s3_denom = 0.25 * s3_p + s3_r
    s3_f05 = (1.25 * s3_p * s3_r) / s3_denom if s3_denom > 0 else 0.0

    country_summary = {}
    for c, scores in country_f05.items():
        n = len(scores)
        country_summary[c] = {
            "count": n,
            "macro_f05": round(sum(scores) / n, 4),
            "macro_prec": round(sum(country_prec[c]) / n, 4),
            "macro_rec": round(sum(country_rec[c]) / n, 4),
        }

    return {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_prec, 5),
        "macro_recall": round(macro_rec, 5),
        "micro_f05": round(micro_f05, 5),
        "micro_precision": round(micro_precision, 5),
        "micro_recall": round(micro_recall, 5),
        "singleton_count": singleton_total,
        "singleton_accuracy": round(singleton_acc, 5),
        "non_singleton_count": non_singleton_total,
        "non_singleton_macro_f05": round(non_singleton_macro_f05, 5),
        "total_true_matches": micro_tp + micro_fn,
        "total_predicted_matches": micro_tp + micro_fp,
        "true_positives": micro_tp,
        "false_positives": micro_fp,
        "false_negatives": micro_fn,
        "s2_micro": {"precision": round(s2_p, 4), "recall": round(s2_r, 4), "f05": round(s2_f05, 4)},
        "s3_micro": {"precision": round(s3_p, 4), "recall": round(s3_r, 4), "f05": round(s3_f05, 4)},
        "by_country": country_summary,
    }

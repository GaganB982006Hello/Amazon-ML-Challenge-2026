"""Threshold calibration and entity-level match decision logic for Amazon ML Challenge 2026.

Optimizes decision thresholds specifically for the precision-heavy macro F0.5 metric.
Handles singleton detection, score gap filtering, and optional source-specific thresholds.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from evaluation import compute_entity_f05, evaluate_predictions


class ThresholdOptimizer:
    """Finds the optimal decision threshold for macro F0.5 on validation data."""

    def __init__(
        self,
        threshold_range: Tuple[float, float, float] = (0.30, 0.95, 0.02),
        score_gap_margin: float = 0.25,
    ):
        self.start, self.end, self.step = threshold_range
        self.score_gap_margin = score_gap_margin

    def search_optimal_threshold(
        self,
        val_s1_ids: List[str],
        val_ground_truth: Dict[str, Set[str]],
        val_candidate_scores: Dict[str, List[Tuple[str, float]]],
        entity_countries: Optional[Dict[str, str]] = None,
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Search for the best global threshold maximizing validation macro F0.5.

        Args:
            val_s1_ids: List of all validation Source 1 IDs (including singletons).
            val_ground_truth: Dict mapping S1 ID -> set of true matched target IDs.
            val_candidate_scores: Dict mapping S1 ID -> list of (target_id, probability).
            entity_countries: Optional country mapping.

        Returns:
            (best_threshold, best_metrics_dict, all_threshold_evaluations)
        """
        thresholds = np.arange(self.start, self.end + 1e-5, self.step)
        all_evals = []

        best_tau = 0.70
        best_f05 = -1.0
        best_metrics: Dict[str, Any] = {}

        for tau in thresholds:
            tau = round(float(tau), 3)
            # Generate predictions at threshold tau
            preds: Dict[str, Set[str]] = {}
            for s1_id in val_s1_ids:
                cand_list = val_candidate_scores.get(s1_id, [])
                if not cand_list:
                    preds[s1_id] = set()
                    continue

                # Filter by threshold tau and optional gap margin relative to top candidate
                top_prob = cand_list[0][1] if cand_list else 0.0
                matched = set()
                for tid, prob in cand_list:
                    if prob >= tau and (top_prob - prob) <= self.score_gap_margin:
                        matched.add(tid)

                preds[s1_id] = matched

            eval_res = evaluate_predictions(val_ground_truth, preds, entity_countries)
            eval_res["threshold"] = tau
            all_evals.append(eval_res)

            if eval_res["macro_f05"] > best_f05:
                best_f05 = eval_res["macro_f05"]
                best_tau = tau
                best_metrics = eval_res

        return best_tau, best_metrics, all_evals

    def search_source_specific_thresholds(
        self,
        val_s1_ids: List[str],
        val_ground_truth: Dict[str, Set[str]],
        val_candidate_scores: Dict[str, List[Tuple[str, float]]],
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """Search for joint (tau_s2, tau_s3) optimal thresholds."""
        grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        best_s2 = 0.70
        best_s3 = 0.70
        best_f05 = -1.0
        best_metrics: Dict[str, Any] = {}

        for tau_s2 in grid:
            for tau_s3 in grid:
                preds: Dict[str, Set[str]] = {}
                for s1_id in val_s1_ids:
                    cand_list = val_candidate_scores.get(s1_id, [])
                    matched = set()
                    for tid, prob in cand_list:
                        cutoff = tau_s2 if tid.startswith("S2-") else tau_s3
                        if prob >= cutoff:
                            matched.add(tid)
                    preds[s1_id] = matched

                eval_res = evaluate_predictions(val_ground_truth, preds)
                if eval_res["macro_f05"] > best_f05:
                    best_f05 = eval_res["macro_f05"]
                    best_s2 = tau_s2
                    best_s3 = tau_s3
                    best_metrics = eval_res

        return best_s2, best_s3, best_f05, best_metrics


def apply_decision_rules(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
    threshold: float,
    score_gap_margin: float = 0.25,
    s2_threshold: Optional[float] = None,
    s3_threshold: Optional[float] = None,
) -> Dict[str, List[str]]:
    """Apply calibrated decision rules to produce final matches per S1 entity.

    Returns:
        Dict mapping s1_id -> list of matched target IDs (sorted, deduplicated).
    """
    final_matches: Dict[str, List[str]] = {}

    for s1_id in all_s1_ids:
        cand_list = candidate_scores.get(s1_id, [])
        if not cand_list:
            final_matches[s1_id] = []
            continue

        # Sort candidates descending by probability
        cand_list_sorted = sorted(cand_list, key=lambda x: x[1], reverse=True)
        top_prob = cand_list_sorted[0][1]

        matched = []
        for tid, prob in cand_list_sorted:
            # Source-specific cutoff if provided, else global threshold
            if tid.startswith("S2-") and s2_threshold is not None:
                cutoff = s2_threshold
            elif tid.startswith("S3-") and s3_threshold is not None:
                cutoff = s3_threshold
            else:
                cutoff = threshold

            if prob >= cutoff and (top_prob - prob) <= score_gap_margin:
                matched.append(tid)

        # Deduplicate while preserving score order
        seen = set()
        deduped = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                deduped.append(m)

        final_matches[s1_id] = deduped

    return final_matches

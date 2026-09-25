"""Country-partitioned scalable batch inference module for Amazon ML Challenge 2026.

Streams test records country-by-country and in chunks:
1. Loads target records (S2, S3) for a specific country
2. Indexes target records using MultiStrategyBlocker
3. Streams Source 1 records in batches
4. Generates candidate pairs
5. Computes pair features using RapidFuzz
6. Scores pairs using the trained LightGBM model
7. Applies calibrated decision rules
8. Streams matches and candidates directly to output TSVs
Maintains low memory footprint across multi-million entity datasets.
"""

import csv
import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from blocking import MultiStrategyBlocker
from calibration import apply_decision_rules
from config import PathConfig, PipelineConfig
from data_loader import EntityRecord
from features import PairFeatureExtractor
from model import LightGBMPairModel


def stream_source1_by_country(
    source1_path: str,
    target_country: str,
    chunk_size: int = 50000,
):
    """Generator yielding chunks of EntityRecord for a specific country from Source 1."""
    chunk: List[EntityRecord] = []
    with open(source1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            country = row[3].strip() if len(row) > 3 else "UNKNOWN"
            if country == target_country:
                chunk.append(
                    EntityRecord(
                        row[0].strip(),
                        row[1].strip() if len(row) > 1 else "",
                        row[2].strip() if len(row) > 2 else "",
                        country,
                    )
                )
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk


def load_targets_for_country(
    source2_path: str,
    source3_path: str,
    target_country: str,
    max_records_per_source: Optional[int] = None,
) -> Dict[str, EntityRecord]:
    """Load S2 and S3 records for a specific country."""
    targets: Dict[str, EntityRecord] = {}

    for src_path in (source2_path, source3_path):
        count = 0
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                country = row[3].strip() if len(row) > 3 else "UNKNOWN"
                if country == target_country:
                    eid = row[0].strip()
                    targets[eid] = EntityRecord(
                        eid,
                        row[1].strip() if len(row) > 1 else "",
                        row[2].strip() if len(row) > 2 else "",
                        country,
                    )
                    count += 1
                    if max_records_per_source and count >= max_records_per_source:
                        break

    return targets


def run_country_inference(
    model: LightGBMPairModel,
    feature_extractor: PairFeatureExtractor,
    country: str,
    s1_chunk: List[EntityRecord],
    blocker: MultiStrategyBlocker,
    targets_dict: Dict[str, EntityRecord],
    threshold: float,
    score_gap: float = 0.25,
    s2_threshold: Optional[float] = None,
    s3_threshold: Optional[float] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """Run candidate retrieval, feature extraction, scoring, and matching for an S1 chunk."""
    s1_dict = {rec.entity_id: rec for rec in s1_chunk}

    # 1. Blocking / Candidate retrieval
    candidates_map = blocker.generate_candidate_pairs(s1_dict)

    # 2. Build candidate pairs list
    pair_list: List[Tuple[str, str]] = []
    s1_pair_map = {rec.entity_id: [] for rec in s1_chunk}

    for rec in s1_chunk:
        cands = candidates_map.get(rec.entity_id, set())
        for cid in cands:
            idx = len(pair_list)
            pair_list.append((rec.entity_id, cid))
            s1_pair_map[rec.entity_id].append((cid, idx))

    if not pair_list:
        # All singletons in this chunk
        empty_matches = {rec.entity_id: [] for rec in s1_chunk}
        return empty_matches, candidates_map

    # 3. Extract features
    X = feature_extractor.extract_features_matrix(pair_list, s1_dict, targets_dict)

    # 4. Predict probabilities
    probs = model.predict_proba(X)

    # 5. Group scores by S1
    scores_by_s1: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, c_indices in s1_pair_map.items():
        scores_by_s1[s1_id] = [(cid, float(probs[idx])) for cid, idx in c_indices]

    # 6. Apply decision rules
    all_s1_ids = [rec.entity_id for rec in s1_chunk]
    matches = apply_decision_rules(
        scores_by_s1,
        all_s1_ids,
        threshold=threshold,
        score_gap_margin=score_gap,
        s2_threshold=s2_threshold,
        s3_threshold=s3_threshold,
    )

    return matches, candidates_map

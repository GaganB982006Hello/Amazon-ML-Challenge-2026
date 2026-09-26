"""End-to-end entity resolution pipeline for Amazon ML Challenge 2026.

Orchestrates training, candidate blocking, feature extraction, scoring,
threshold calibration, submission generation, and validation.
"""

import argparse
import atexit
import csv
import gc
import json
import os
import sqlite3
import sys
import tempfile
import time
from typing import Dict, List, Optional, Set, Tuple
import joblib
import numpy as np

from blocking import MultiStrategyBlocker, evaluate_blocking_recall
from calibration import apply_decision_rules
from config import PathConfig, PipelineConfig
from data_loader import (
    EntityRecord,
    create_train_val_split,
    load_entities_file,
    load_ground_truth,
)
from features import PairFeatureExtractor
from inference import load_targets_for_country, run_country_inference, stream_source1_by_country
from model import LightGBMPairModel, build_training_pairs_and_labels
from submission import run_official_validator, write_candidate_pairs, write_matching_results


def train_production_model(
    config: PipelineConfig,
    paths: PathConfig,
    train_size: int = 40000,
    model_save_path: Optional[str] = None,
) -> Tuple[LightGBMPairModel, PairFeatureExtractor]:
    """Train the production LightGBM model using all appropriate training data with hard negatives."""
    print("=" * 60)
    print("Training Production Entity Resolution Model...")
    print("=" * 60)
    t0 = time.time()

    # 1. Load Ground Truth
    gt_map = load_ground_truth(paths.train_ground_truth, max_rows=train_size * 2)
    s1_ids = list(gt_map.keys())[:train_size]
    print(f"Sampled {len(s1_ids):,} Source 1 entities for training...")

    # 2. Load S1 entities
    s1_records = load_entities_file(paths.train_source1, filter_ids=set(s1_ids))

    # 3. Collect targets needed
    needed_targets = set()
    for s1 in s1_ids:
        needed_targets.update(gt_map.get(s1, set()))

    # Load targets + background pool for realistic hard negatives
    s2_needed = {m for m in needed_targets if m.startswith("S2-")}
    s3_needed = {m for m in needed_targets if m.startswith("S3-")}

    print(f"Loading {len(needed_targets):,} target records + background pool...")
    s2_records = load_entities_file(paths.train_source2, max_rows=50000)
    s3_records = load_entities_file(paths.train_source3, max_rows=50000)
    s2_records.update(load_entities_file(paths.train_source2, filter_ids=s2_needed))
    s3_records.update(load_entities_file(paths.train_source3, filter_ids=s3_needed))

    target_pool = {}
    target_pool.update(s2_records)
    target_pool.update(s3_records)
    print(f"Target pool contains {len(target_pool):,} records.")

    # 4. Multi-strategy blocking to mine realistic hard negatives
    print("Mining hard negatives from blocking candidates...")
    blocker = MultiStrategyBlocker(max_candidates_per_s1=config.max_candidates_per_entity)
    blocker.index_target_entities(target_pool)
    train_candidates = blocker.generate_candidate_pairs(s1_records)

    # 5. Build training pairs with hard negative sampling
    train_pairs, train_labels = build_training_pairs_and_labels(
        s1_ids,
        train_candidates,
        gt_map,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
        random_seed=config.random_seed,
    )

    # 6. Extract features
    print(f"Extracting features for {len(train_pairs):,} training pairs...")
    extractor = PairFeatureExtractor()
    X_train = extractor.extract_features_matrix(train_pairs, s1_records, target_pool)

    # 7. Fit LightGBM model
    print(f"Fitting LightGBM classifier on {X_train.shape[0]:,} pairs...")
    model = LightGBMPairModel(
        n_estimators=config.lgb_n_estimators,
        learning_rate=config.lgb_learning_rate,
        num_leaves=config.lgb_num_leaves,
        random_state=config.random_seed,
    )
    model.fit(X_train, train_labels)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"Saved model to {model_save_path}")

    print(f"Production model training completed in {time.time()-t0:.1f}s.")
    return model, extractor


def run_full_inference(
    model: LightGBMPairModel,
    feature_extractor: PairFeatureExtractor,
    config: PipelineConfig,
    paths: PathConfig,
    threshold: float = 0.70,
    chunk_size: int = 500,
) -> None:
    """Run full test inference country-by-country and write final outputs."""
    print("=" * 60)
    print("Running Full Test Inference across all Countries...")
    print("=" * 60)
    t_start = time.time()

    # 1. Discover all unique countries in test_source1.tsv (strictly open-set)
    print("Discovering open-set countries in test_source1.tsv...")
    countries_found: List[str] = []
    seen_countries = set()
    total_test_s1 = 0

    with open(paths.test_source1, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            total_test_s1 += 1
            c = row[3].strip() if len(row) > 3 else "UNKNOWN"
            if c not in seen_countries:
                seen_countries.add(c)
                countries_found.append(c)

    print(f"Found {len(countries_found)} countries across {total_test_s1:,} test S1 entities: {countries_found}")

    # Existing output rows are checkpoints, allowing inference to resume after interruption.
    os.makedirs(paths.output_dir, exist_ok=True)

    checkpoint_file = tempfile.NamedTemporaryFile(
        prefix="entity_resolution_", suffix=".sqlite", delete=False
    )
    checkpoint_path = checkpoint_file.name
    checkpoint_file.close()
    atexit.register(lambda: os.path.exists(checkpoint_path) and os.remove(checkpoint_path))
    checkpoint = sqlite3.connect(checkpoint_path)
    checkpoint.execute("PRAGMA journal_mode=OFF")
    checkpoint.execute("PRAGMA synchronous=OFF")
    checkpoint.execute("PRAGMA cache_size=-32768")
    checkpoint.execute(
        "CREATE TABLE completed (entity_id TEXT PRIMARY KEY, matching INTEGER NOT NULL, candidate INTEGER NOT NULL)"
    )

    def index_existing_output(path: str, expected_header: List[str], column: str) -> None:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as output_file:
                output_file.write("\t".join(expected_header) + "\n")
            return
        with open(path, "r", encoding="utf-8", newline="") as output_file:
            reader = csv.reader(output_file, delimiter="\t")
            header = next(reader, [])
            if header != expected_header:
                raise ValueError(f"Unexpected header in {path}: {header}")
            batch = []
            for row in reader:
                if row and row[0].strip():
                    batch.append((row[0].strip(),))
                if len(batch) >= 10000:
                    checkpoint.executemany(
                        f"INSERT INTO completed (entity_id, matching, candidate) VALUES (?, {1 if column == 'matching' else 0}, {1 if column == 'candidate' else 0}) "
                        f"ON CONFLICT(entity_id) DO UPDATE SET {column}=1",
                        batch,
                    )
                    batch.clear()
            if batch:
                checkpoint.executemany(
                    f"INSERT INTO completed (entity_id, matching, candidate) VALUES (?, {1 if column == 'matching' else 0}, {1 if column == 'candidate' else 0}) "
                    f"ON CONFLICT(entity_id) DO UPDATE SET {column}=1",
                    batch,
                )
        checkpoint.commit()

    index_existing_output(
        paths.matching_output, ["source1_entity_id", "matched_entity_ids"], "matching"
    )
    index_existing_output(
        paths.candidate_output, ["source1_entity_id", "candidate_entity_ids"], "candidate"
    )
    matching_count, candidate_count = checkpoint.execute(
        "SELECT COALESCE(SUM(matching), 0), COALESCE(SUM(candidate), 0) FROM completed"
    ).fetchone()

    if matching_count == total_test_s1 and candidate_count == total_test_s1:
        print("All test Source 1 IDs are already present; validating existing TSVs.")
        is_pass, val_out = run_official_validator(
            paths.matching_output, paths.candidate_output, paths.test_dir, check_ids=False
        )
        print(val_out)
        checkpoint.close()
        os.remove(checkpoint_path)
        return

    total_matches_written = 0
    total_singletons_predicted = 0
    total_s1_processed = 0

    # Process each country independently to preserve memory
    for country in countries_found:
        print(f"\n--- Processing Country: {country} ---")
        t_country = time.time()

        # Load target records (S2, S3) for this country
        print(f"Loading test target records (S2 + S3) for {country}...")
        country_targets = load_targets_for_country(
            paths.test_source2, paths.test_source3, country
        )
        print(f"Loaded {len(country_targets):,} target records for {country}.")

        # Build country blocking index
        blocker = MultiStrategyBlocker(max_candidates_per_s1=config.max_candidates_per_entity)
        blocker.index_target_entities(country_targets)

        # Stream S1 chunks for this country
        chunk_idx = 0
        country_s1_processed = 0

        with open(paths.matching_output, "a", encoding="utf-8") as f_match, \
             open(paths.candidate_output, "a", encoding="utf-8") as f_cand:

            for s1_chunk in stream_source1_by_country(paths.test_source1, country, chunk_size=chunk_size):
                chunk_idx += 1
                chunk_start = time.time()
                chunk_ids = [rec.entity_id for rec in s1_chunk]
                placeholders = ",".join("?" for _ in chunk_ids)
                written_status = {
                    row[0]: (row[1], row[2])
                    for row in checkpoint.execute(
                        f"SELECT entity_id, matching, candidate FROM completed WHERE entity_id IN ({placeholders})",
                        chunk_ids,
                    )
                }
                pending_chunk = [
                    rec for rec in s1_chunk
                    if written_status.get(rec.entity_id, (0, 0)) != (1, 1)
                ]
                if not pending_chunk:
                    continue

                matches, candidates = run_country_inference(
                    model=model,
                    feature_extractor=feature_extractor,
                    country=country,
                    s1_chunk=pending_chunk,
                    blocker=blocker,
                    targets_dict=country_targets,
                    threshold=threshold,
                    score_gap=config.singleton_score_margin,
                )

                # Write chunk rows
                for rec in pending_chunk:
                    s1_id = rec.entity_id
                    matched_list = matches.get(s1_id, [])
                    cand_set = set(candidates.get(s1_id, set()))
                    # Guarantee that every matched ID is present in candidates
                    cand_set.update(matched_list)

                    if not matched_list:
                        total_singletons_predicted += 1
                    else:
                        total_matches_written += len(matched_list)

                    # Write matching row
                    matching_done, candidate_done = written_status.get(s1_id, (0, 0))
                    if not matching_done:
                        f_match.write(f"{s1_id}\t{','.join(matched_list)}\n")
                        matching_done = 1
                    # Write candidate row
                    if not candidate_done:
                        f_cand.write(f"{s1_id}\t{','.join(sorted(cand_set))}\n")
                        candidate_done = 1
                    written_status[s1_id] = (matching_done, candidate_done)

                f_match.flush()
                f_cand.flush()
                os.fsync(f_match.fileno())
                os.fsync(f_cand.fileno())
                checkpoint.executemany(
                    "INSERT INTO completed (entity_id, matching, candidate) VALUES (?, ?, ?) "
                    "ON CONFLICT(entity_id) DO UPDATE SET matching=excluded.matching, candidate=excluded.candidate",
                    [(entity_id, status[0], status[1]) for entity_id, status in written_status.items()],
                )
                checkpoint.commit()
                country_s1_processed += len(pending_chunk)
                total_s1_processed += len(pending_chunk)
                print(f"  [{country}] Chunk {chunk_idx}: Processed {len(pending_chunk):,} entities in {time.time()-chunk_start:.1f}s "
                      f"(Country Progress: {country_s1_processed:,})")
                del matches, candidates, pending_chunk
                del written_status, chunk_ids, s1_chunk
                if chunk_idx % 10 == 0:
                    gc.collect()

        print(f"Country {country} completed in {time.time()-t_country:.1f}s.")

        # Clean memory before next country
        del country_targets
        del blocker
        gc.collect()

    print(f"\nAll countries completed in {time.time()-t_start:.1f}s!")
    print(f"Total test entities processed: {total_test_s1:,}")
    print(f"Total final matched pairs: {total_matches_written:,}")
    print(f"Total predicted singletons: {total_singletons_predicted:,}")

    matching_count, candidate_count = checkpoint.execute(
        "SELECT COALESCE(SUM(matching), 0), COALESCE(SUM(candidate), 0) FROM completed"
    ).fetchone()
    checkpoint.close()
    os.remove(checkpoint_path)
    if matching_count != total_test_s1 or candidate_count != total_test_s1:
        raise RuntimeError(
            f"Output coverage is incomplete: matching has {matching_count:,}, "
            f"candidate has {candidate_count:,}, expected {total_test_s1:,} rows."
        )

    print(f"Wrote matching results to: {paths.matching_output}")
    print(f"Wrote candidate pairs to:  {paths.candidate_output}")

    # Run official validator
    print("\n[Step 4] Running Official Competition Validator...")
    is_pass, val_out = run_official_validator(
        matching_path=paths.matching_output,
        candidate_path=paths.candidate_output,
        test_dir=paths.test_dir,
        check_ids=False,
    )
    print(val_out)
    if is_pass:
        print(">>> SUCCESS: Submission output passed all validation checks! <<<")
    else:
        print(">>> ERROR: Validation checks failed! Inspect errors above. <<<")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument("--mode", choices=["train", "inference", "all"], default="all")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    cfg = PipelineConfig()
    paths = PathConfig()

    model_path = os.path.join(paths.base_dir, "model_artifact.joblib")

    if args.mode in ("train", "all"):
        model, extractor = train_production_model(
            config=cfg,
            paths=paths,
            train_size=args.train_size,
            model_save_path=model_path,
        )
    else:
        print(f"Loading trained model from {model_path}...")
        model = joblib.load(model_path)
        extractor = PairFeatureExtractor()

    if args.mode in ("inference", "all"):
        run_full_inference(
            model=model,
            feature_extractor=extractor,
            config=cfg,
            paths=paths,
            threshold=args.threshold,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()

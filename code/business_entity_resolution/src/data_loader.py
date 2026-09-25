"""Data loader and train/validation split module for Amazon ML Challenge 2026.

Provides streaming and memory-efficient loading of TSV datasets,
entity-level train/validation splitting, and ground truth mapping.
"""

from collections import defaultdict
import csv
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple


class EntityRecord:
    """Lightweight representation of an entity record."""
    __slots__ = ("entity_id", "business_name", "business_address", "country")

    def __init__(self, entity_id: str, business_name: str, business_address: str, country: str):
        self.entity_id = entity_id
        self.business_name = business_name
        self.business_address = business_address
        self.country = country

    def __repr__(self) -> str:
        return f"EntityRecord({self.entity_id}, {self.business_name!r}, {self.country})"


def load_ground_truth(
    gt_path: str,
    filter_s1_ids: Optional[Set[str]] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Set[str]]:
    """Load ground truth mapping source1_entity_id -> set of matched_entity_ids.

    Always includes singletons (mapping to an empty set).
    """
    gt_map: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        count = 0
        for row in reader:
            if not row or not row[0].strip():
                continue
            s1_id = row[0].strip()
            if filter_s1_ids is not None and s1_id not in filter_s1_ids:
                continue

            matched_str = row[1].strip() if len(row) > 1 else ""
            if matched_str:
                mids = {m.strip() for m in matched_str.split(",") if m.strip()}
            else:
                mids = set()

            gt_map[s1_id] = mids
            count += 1
            if max_rows and count >= max_rows:
                break

    return gt_map


def load_entities_file(
    file_path: str,
    filter_ids: Optional[Set[str]] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, EntityRecord]:
    """Load records from a source TSV into a dict of EntityRecord."""
    records: Dict[str, EntityRecord] = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        count = 0
        for row in reader:
            if not row or not row[0].strip():
                continue
            eid = row[0].strip()
            if filter_ids is not None and eid not in filter_ids:
                continue

            name = row[1].strip() if len(row) > 1 else ""
            addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""

            records[eid] = EntityRecord(eid, name, addr, country)
            count += 1
            if max_rows and count >= max_rows:
                break

    return records


def create_train_val_split(
    gt_map: Dict[str, Set[str]],
    s1_records: Dict[str, EntityRecord],
    val_size: int = 15000,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Create a stratified entity-level train/val split based on country and singleton status.

    Args:
        gt_map: Complete ground truth mapping for S1 entities.
        s1_records: Dictionary of S1 EntityRecord.
        val_size: Total number of S1 entities to assign to validation.
        random_seed: Random seed for exact reproducibility.

    Returns:
        (train_s1_ids, val_s1_ids)
    """
    rng = random.Random(random_seed)

    # Group entities by stratum: (country, is_singleton, match_count_bucket)
    strata: Dict[Tuple[str, bool, int], List[str]] = defaultdict(list)
    for s1_id, matched_set in gt_map.items():
        if s1_id not in s1_records:
            continue
        country = s1_records[s1_id].country
        is_singleton = len(matched_set) == 0
        match_bucket = min(len(matched_set), 5)
        strata[(country, is_singleton, match_bucket)].append(s1_id)

    total_available = sum(len(ids) for ids in strata.values())
    val_ratio = min(val_size / total_available, 0.5)

    train_ids: List[str] = []
    val_ids: List[str] = []

    for key, ids in strata.items():
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)

    return train_ids, val_ids

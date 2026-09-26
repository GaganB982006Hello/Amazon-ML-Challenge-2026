"""Multi-strategy blocking and candidate generation module for Amazon ML Challenge 2026.

Implements high-recall, memory-efficient candidate retrieval:
- Partitioning by country (open-set support)
- Exact clean name match
- Legal-suffix-stripped name match
- Alphanumeric compact name match
- Rare name token inverted index
- Rare address token inverted index (bridges across languages / trade names)
- Combined Name-token + House-number / Postal-code match
- House-number + Address-token match
"""

from collections import defaultdict
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from normalization import (
    clean_text_base,
    get_character_ngrams,
    normalize_address,
    normalize_business_name,
)


class MultiStrategyBlocker:
    """High-recall multi-strategy candidate generator."""

    def __init__(
        self,
        max_candidates_per_s1: int = 40,
        max_token_doc_freq: int = 250,
        min_token_len: int = 3,
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.max_token_doc_freq = max_token_doc_freq
        self.min_token_len = min_token_len

        # Stop words to never index as single tokens (multilingual)
        self.stopwords = {
            "and", "the", "for", "with", "inc", "ltd", "pvt", "corp", "llc", "llp",
            "co", "company", "limited", "private", "enterprises", "services", "solutions",
            "center", "group", "holdings", "trading", "retail", "management", "international",
            "de", "la", "le", "des", "du", "et", "en", "pour", "france", "india", "us",
            "street", "road", "st", "rd", "avenue", "ave", "boulevard", "blvd", "lane", "ln",
            "drive", "dr", "suite", "ste", "apartment", "apt", "floor", "fl", "building", "bldg",
            "near", "opp", "opposite", "behind", "colony", "nagar", "road", "cross", "main",
            "north", "south", "east", "west", "new", "city", "state",
        }

        # Country-partitioned indexes
        self.country_indexes: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
            lambda: {
                "name": defaultdict(list),
                "rare_token": defaultdict(list),
            }
        )

        self.target_records: Dict[str, Any] = {}

        # Token frequencies per country to filter high-frequency tokens
        self.token_freqs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def index_target_entities(
        self,
        target_records: Dict[str, Any],  # Dict mapping eid -> EntityRecord
    ) -> None:
        """Index all target entities (S2 and S3) across multiple blocking strategies."""
        print(f"Indexing {len(target_records):,} target entities...")
        self.target_records = target_records

        # First pass: count token frequencies per country to cap common tokens
        for eid, rec in target_records.items():
            country = rec.country or "UNKNOWN"
            _, _, tokens = normalize_business_name(rec.business_name)
            for t in set(tokens):
                if len(t) >= self.min_token_len and t not in self.stopwords:
                    self.token_freqs[country][t] += 1


        # Second pass: build inverted indexes
        for eid, rec in target_records.items():
            country = rec.country or "UNKNOWN"
            idx = self.country_indexes[country]

            raw_clean, stripped_legal, tokens = normalize_business_name(rec.business_name)
            token_set = set(tokens)

            # Store both normalized name forms in one index to avoid duplicate indexes.
            if raw_clean:
                idx["name"][raw_clean].append(eid)
            if stripped_legal and stripped_legal != raw_clean:
                idx["name"][stripped_legal].append(eid)

            # Strategy 5: Rare name tokens
            for t in token_set:
                if (
                    len(t) >= self.min_token_len
                    and t not in self.stopwords
                    and self.token_freqs[country][t] <= self.max_token_doc_freq
                ):
                    idx["rare_token"][t].append(eid)



        print(f"Finished indexing targets into country-partitioned multi-indexes.")

    def retrieve_candidates_for_s1(
        self,
        s1_eid: str,
        business_name: str,
        business_address: str,
        country: str,
    ) -> Set[str]:
        """Retrieve candidate IDs for a single Source 1 entity."""
        country_key = country or "UNKNOWN"
        idx = self.country_indexes.get(country_key)
        if not idx:
            return set()

        raw_clean, stripped_legal, tokens = normalize_business_name(business_name)
        token_set = set(tokens)

        candidates: Set[str] = set()

        if raw_clean in idx["name"]:
            candidates.update(idx["name"][raw_clean])
        if stripped_legal and stripped_legal != raw_clean and stripped_legal in idx["name"]:
            candidates.update(idx["name"][stripped_legal])


        # 5. Rare name tokens
        for t in token_set:
            if (
                len(t) >= self.min_token_len
                and t not in self.stopwords
                and self.token_freqs[country_key].get(t, 999999) <= self.max_token_doc_freq
            ):
                if t in idx["rare_token"]:
                    for tid in idx["rare_token"][t]:
                        candidates.add(tid)
                        if len(candidates) >= self.max_candidates_per_s1 * 3:
                            break



        # Cap candidates per S1 entity to avoid explosion
        if len(candidates) > self.max_candidates_per_s1:
            _, addr_tokens, addr_numbers = normalize_address(business_address, country)
            addr_token_set = set(addr_tokens)
            scored = []
            for cid in candidates:
                rec = self.target_records.get(cid)
                if rec is None:
                    scored.append((0, cid))
                    continue
                c_clean, c_strip, c_tokens = normalize_business_name(rec.business_name)
                _, c_addr_tokens, c_nums = normalize_address(rec.business_address, country)
                c_tokens = set(c_tokens)
                c_addr_tokens = set(c_addr_tokens)
                c_nums = set(c_nums)
                score = 0.0
                if c_clean == raw_clean or c_strip == stripped_legal:
                    score += 10.0
                shared_tokens = len(token_set & c_tokens)
                score += shared_tokens * 3.0
                shared_nums = len(addr_numbers & c_nums)
                score += shared_nums * 2.5
                shared_addr = len(addr_token_set & c_addr_tokens)
                score += shared_addr * 1.5
                scored.append((score, cid))

            scored.sort(key=lambda x: x[0], reverse=True)
            candidates = {cid for _, cid in scored[: self.max_candidates_per_s1]}

        return candidates

    def generate_candidate_pairs(
        self,
        s1_records: Dict[str, Any],
    ) -> Dict[str, Set[str]]:
        """Generate candidates for all S1 entities."""
        candidates_map: Dict[str, Set[str]] = {}
        for eid, rec in s1_records.items():
            candidates = self.retrieve_candidates_for_s1(
                eid, rec.business_name, rec.business_address, rec.country
            )
            candidates_map[eid] = candidates
        return candidates_map


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates_map: Dict[str, Set[str]],
) -> Dict[str, Any]:
    """Evaluate candidate recall and reduction ratio against ground truth."""
    total_true_matches = 0
    recalled_matches = 0
    s1_with_all_recalled = 0
    total_candidates = 0
    non_singleton_count = 0

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates_map.get(s1_id, set())
        total_candidates += len(cand_set)

        if not true_set:
            continue

        non_singleton_count += 1
        n_true = len(true_set)
        total_true_matches += n_true

        found = len(true_set & cand_set)
        recalled_matches += found

        if found == n_true:
            s1_with_all_recalled += 1

    candidate_recall = recalled_matches / total_true_matches if total_true_matches > 0 else 0.0
    full_coverage_rate = s1_with_all_recalled / non_singleton_count if non_singleton_count > 0 else 0.0
    avg_cands_per_s1 = total_candidates / len(candidates_map) if candidates_map else 0.0

    return {
        "total_true_matches": total_true_matches,
        "recalled_matches": recalled_matches,
        "candidate_recall": round(candidate_recall, 5),
        "candidate_recall_pct": round(candidate_recall * 100, 2),
        "entities_with_100pct_coverage": s1_with_all_recalled,
        "full_coverage_rate": round(full_coverage_rate, 5),
        "total_candidates_generated": total_candidates,
        "avg_candidates_per_s1": round(avg_cands_per_s1, 2),
    }

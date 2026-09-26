"""Pair feature extraction module for Amazon ML Challenge 2026.

Extracts rich string, token, phonetic, numeric, interaction, and source-aware
similarity features for candidate pairs using RapidFuzz for high speed.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from rapidfuzz import fuzz

from normalization import (
    clean_text_base,
    get_character_ngrams,
    normalize_address,
    normalize_business_name,
)

FEATURE_NAMES = [
    # Business name features
    "name_exact_clean",
    "name_exact_strip",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_char3_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_prefix_4",
    "name_prefix_8",
    # Address features
    "addr_exact",
    "addr_ratio",
    "addr_token_set_ratio",
    "addr_token_sort_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_numbers_jaccard",
    "addr_numbers_shared_count",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_is_empty",
    # Country features
    "country_exact_match",
    # Cross-field interactions
    "name_sim_x_addr_sim",
    "max_sim",
    "min_sim",
    "high_name_low_addr",
    "low_name_high_addr",
    "both_high",
    # Source indicators
    "is_source2",
    "is_source3",
]


class PairFeatureExtractor:
    """Computes similarity features between Source 1 records and candidate target records."""

    def __init__(self):
        # Cache for normalized representations to avoid redundant processing
        # eid -> (raw_clean, stripped_legal, tokens_set, token_list, char3_set)
        self.name_cache: Dict[str, Tuple[str, str, Set[str], List[str], Set[str]]] = {}
        # eid -> (addr_norm, tokens_set, token_list, numeric_set, is_empty)
        self.addr_cache: Dict[str, Tuple[str, Set[str], List[str], Set[str], float]] = {}

    def clear_cache(self) -> None:
        self.name_cache.clear()
        self.addr_cache.clear()

    def get_name_rep(self, eid: str, name: str) -> Tuple[str, str, Set[str], List[str], Set[str]]:
        if eid in self.name_cache:
            return self.name_cache[eid]
        raw_clean, stripped_legal, tokens = normalize_business_name(name)
        token_set = set(tokens)
        char3 = get_character_ngrams(stripped_legal or raw_clean, 3)
        res = (raw_clean, stripped_legal, token_set, tokens, char3)
        self.name_cache[eid] = res
        return res

    def get_addr_rep(self, eid: str, addr: str, country: str) -> Tuple[str, Set[str], List[str], Set[str], float]:
        if eid in self.addr_cache:
            return self.addr_cache[eid]
        norm_str, tokens, nums = normalize_address(addr, country)
        token_set = set(tokens)
        is_empty = 1.0 if not norm_str else 0.0
        res = (norm_str, token_set, tokens, nums, is_empty)
        self.addr_cache[eid] = res
        return res

    def compute_pair_features(
        self,
        s1_eid: str,
        s1_name: str,
        s1_addr: str,
        s1_country: str,
        t_eid: str,
        t_name: str,
        t_addr: str,
        t_country: str,
    ) -> List[float]:
        """Compute the complete feature vector for a single candidate pair."""
        # Retrieve or compute normalized name representations
        s1_n_clean, s1_n_strip, s1_n_tok_set, _, s1_n_char3 = self.get_name_rep(s1_eid, s1_name)
        t_n_clean, t_n_strip, t_n_tok_set, _, t_n_char3 = self.get_name_rep(t_eid, t_name)

        # Name comparisons
        name_exact_clean = 1.0 if s1_n_clean and s1_n_clean == t_n_clean else 0.0
        name_exact_strip = 1.0 if s1_n_strip and s1_n_strip == t_n_strip else 0.0

        target_n_str = t_n_strip or t_n_clean
        s1_n_str = s1_n_strip or s1_n_clean

        if s1_n_str and target_n_str and s1_n_str == target_n_str:
            name_ratio = 1.0
            name_token_set_ratio = 1.0
            name_token_sort_ratio = 1.0
            name_partial_ratio = 1.0
            name_token_jaccard = 1.0
            name_token_overlap = float(len(s1_n_tok_set))
            name_char3_jaccard = 1.0
            name_len_diff = 0.0
            name_len_ratio = 1.0
            name_prefix_4 = 1.0
            name_prefix_8 = 1.0
        else:
            name_ratio = fuzz.ratio(s1_n_str, target_n_str) / 100.0 if s1_n_str and target_n_str else 0.0
            name_token_set_ratio = fuzz.token_set_ratio(s1_n_str, target_n_str) / 100.0 if s1_n_str and target_n_str else 0.0
            name_token_sort_ratio = fuzz.token_sort_ratio(s1_n_str, target_n_str) / 100.0 if s1_n_str and target_n_str else 0.0
            name_partial_ratio = fuzz.partial_ratio(s1_n_str, target_n_str) / 100.0 if s1_n_str and target_n_str else 0.0

            n_union = len(s1_n_tok_set | t_n_tok_set)
            n_inter = len(s1_n_tok_set & t_n_tok_set)
            name_token_jaccard = (n_inter / n_union) if n_union > 0 else 0.0
            name_token_overlap = float(n_inter)

            c_union = len(s1_n_char3 | t_n_char3)
            c_inter = len(s1_n_char3 & t_n_char3)
            name_char3_jaccard = (c_inter / c_union) if c_union > 0 else 0.0

            l1, l2 = len(s1_n_str), len(target_n_str)
            name_len_diff = float(abs(l1 - l2))
            name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

            name_prefix_4 = 1.0 if s1_n_str[:4] and s1_n_str[:4] == target_n_str[:4] else 0.0
            name_prefix_8 = 1.0 if s1_n_str[:8] and s1_n_str[:8] == target_n_str[:8] else 0.0

        # Retrieve or compute normalized address representations
        s1_a_str, s1_a_tok_set, _, s1_a_nums, _ = self.get_addr_rep(s1_eid, s1_addr, s1_country)
        t_a_str, t_a_tok_set, _, t_a_nums, addr_is_empty = self.get_addr_rep(t_eid, t_addr, t_country)

        # Address comparisons
        addr_exact = 1.0 if s1_a_str and s1_a_str == t_a_str else 0.0
        if s1_a_str and t_a_str and s1_a_str == t_a_str:
            addr_ratio = 1.0
            addr_token_set_ratio = 1.0
            addr_token_sort_ratio = 1.0
            addr_token_jaccard = 1.0
            addr_token_overlap = float(len(s1_a_tok_set))
            addr_numbers_jaccard = 1.0
            addr_numbers_shared_count = float(len(s1_a_nums))
            addr_len_diff = 0.0
            addr_len_ratio = 1.0
        else:
            addr_ratio = fuzz.ratio(s1_a_str, t_a_str) / 100.0 if s1_a_str and t_a_str else 0.0
            addr_token_set_ratio = fuzz.token_set_ratio(s1_a_str, t_a_str) / 100.0 if s1_a_str and t_a_str else 0.0
            addr_token_sort_ratio = fuzz.token_sort_ratio(s1_a_str, t_a_str) / 100.0 if s1_a_str and t_a_str else 0.0

            a_union = len(s1_a_tok_set | t_a_tok_set)
            a_inter = len(s1_a_tok_set & t_a_tok_set)
            addr_token_jaccard = (a_inter / a_union) if a_union > 0 else 0.0
            addr_token_overlap = float(a_inter)

            num_union = len(s1_a_nums | t_a_nums)
            num_inter = len(s1_a_nums & t_a_nums)
            addr_numbers_jaccard = (num_inter / num_union) if num_union > 0 else 0.0
            addr_numbers_shared_count = float(num_inter)

            al1, al2 = len(s1_a_str), len(t_a_str)
            addr_len_diff = float(abs(al1 - al2))
            addr_len_ratio = (min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 0.0

        # Country equality
        country_exact_match = 1.0 if s1_country and s1_country == t_country else 0.0

        # Cross-field interactions
        name_sim_x_addr_sim = name_token_set_ratio * addr_token_set_ratio
        max_sim = max(name_token_set_ratio, addr_token_set_ratio)
        min_sim = min(name_token_set_ratio, addr_token_set_ratio)
        high_name_low_addr = 1.0 if name_token_set_ratio > 0.85 and addr_token_set_ratio < 0.30 else 0.0
        low_name_high_addr = 1.0 if name_token_set_ratio < 0.30 and addr_token_set_ratio > 0.85 else 0.0
        both_high = 1.0 if name_token_set_ratio > 0.80 and addr_token_set_ratio > 0.80 else 0.0

        # Source indicators
        is_source2 = 1.0 if t_eid.startswith("S2-") else 0.0
        is_source3 = 1.0 if t_eid.startswith("S3-") else 0.0

        return [
            name_exact_clean,
            name_exact_strip,
            name_ratio,
            name_token_set_ratio,
            name_token_sort_ratio,
            name_partial_ratio,
            name_token_jaccard,
            name_token_overlap,
            name_char3_jaccard,
            name_len_diff,
            name_len_ratio,
            name_prefix_4,
            name_prefix_8,
            addr_exact,
            addr_ratio,
            addr_token_set_ratio,
            addr_token_sort_ratio,
            addr_token_jaccard,
            addr_token_overlap,
            addr_numbers_jaccard,
            addr_numbers_shared_count,
            addr_len_diff,
            addr_len_ratio,
            addr_is_empty,
            country_exact_match,
            name_sim_x_addr_sim,
            max_sim,
            min_sim,
            high_name_low_addr,
            low_name_high_addr,
            both_high,
            is_source2,
            is_source3,
        ]

    def extract_features_matrix(
        self,
        candidate_pairs: List[Tuple[str, str]],
        s1_records: Dict[str, Any],
        target_records: Dict[str, Any],
    ) -> np.ndarray:
        """Extract features matrix for a list of (s1_id, target_id) pairs."""
        n = len(candidate_pairs)
        n_feat = len(FEATURE_NAMES)
        X = np.empty((n, n_feat), dtype=np.float32)

        for i, (s1_id, t_id) in enumerate(candidate_pairs):
            s1_rec = s1_records.get(s1_id)
            t_rec = target_records.get(t_id)
            if not s1_rec or not t_rec:
                X[i, :] = 0.0
                continue
            feats = self.compute_pair_features(
                s1_id, s1_rec.business_name, s1_rec.business_address, s1_rec.country,
                t_id, t_rec.business_name, t_rec.business_address, t_rec.country,
            )
            X[i, :] = feats

        return X

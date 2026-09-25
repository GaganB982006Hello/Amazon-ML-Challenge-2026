"""Submission generation and validation module for Amazon ML Challenge 2026.

Formats and outputs matching_results.tsv and candidate_pairs.tsv
strictly following competition rules and runs the official validator.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Set


def write_matching_results(
    output_path: str,
    matches_dict: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> int:
    """Write matching_results.tsv with exact header and format.

    Header:
    source1_entity_id<TAB>matched_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_rows = 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            mids = matches_dict.get(s1_id, [])
            # Deduplicate while preserving order, filter out any S1 IDs
            cleaned_mids = []
            seen = set()
            for mid in mids:
                mid_s = mid.strip()
                if mid_s and mid_s not in seen and not mid_s.startswith("S1-"):
                    seen.add(mid_s)
                    cleaned_mids.append(mid_s)

            match_str = ",".join(cleaned_mids)
            f.write(f"{s1_id}\t{match_str}\n")
            n_rows += 1

    print(f"Wrote {n_rows:,} rows to {output_path}")
    return n_rows


def write_candidate_pairs(
    output_path: str,
    candidates_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    ensure_matches_present: Optional[Dict[str, List[str]]] = None,
) -> int:
    """Write candidate_pairs.tsv with exact header and format.

    Header:
    source1_entity_id<TAB>candidate_entity_ids
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_rows = 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = set(candidates_dict.get(s1_id, set()))
            # Ensure every final match is present in candidates
            if ensure_matches_present and s1_id in ensure_matches_present:
                cands.update(ensure_matches_present[s1_id])

            # Filter and sort
            valid_cands = [c for c in cands if not c.startswith("S1-")]
            cand_str = ",".join(valid_cands)
            f.write(f"{s1_id}\t{cand_str}\n")
            n_rows += 1

    print(f"Wrote {n_rows:,} rows to {output_path}")
    return n_rows


def run_official_validator(
    matching_path: str,
    candidate_path: str,
    test_dir: str,
    check_ids: bool = False,
) -> Tuple[bool, str]:
    """Run utils/validate_submission.py and return (is_pass, output_text)."""
    validator_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils", "validate_submission.py")
    )
    if not os.path.isfile(validator_script):
        # Check alternative location
        alt_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "6ab10eb3b23ba_student_resource",
                "student_resource",
                "utils",
                "validate_submission.py",
            )
        )
        if os.path.isfile(alt_path):
            validator_script = alt_path
        else:
            return False, f"Validator script not found at {validator_script}"

    cmd = [
        sys.executable,
        validator_script,
        "--matching",
        matching_path,
        "--candidate",
        candidate_path,
        "--test-dir",
        test_dir,
    ]
    if check_ids:
        cmd.append("--check-ids")

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = res.stdout + "\n" + res.stderr
    is_pass = (res.returncode == 0) and ("PASS" in output)
    return is_pass, output

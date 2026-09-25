"""Dataset profiling module for Amazon ML Challenge 2026.

Analyzes dataset row counts, missing values, country distributions,
text lengths, character sets, noise patterns, and ground truth statistics.
Streams TSV files to maintain a tiny memory footprint.
"""

from collections import Counter
import csv
import json
import math
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


LEGAL_PATTERNS = [
    r"\b(pvt|private)\s+(ltd|limited)\b",
    r"\b(ltd|limited)\b",
    r"\b(inc|incorporated)\b",
    r"\b(corp|corporation)\b",
    r"\b(llc|llp)\b",
    r"\b(co|company)\b",
    r"\b(sarl|sas|sci)\b",
    r"\b(gmbh|ag)\b",
]

NON_ASCII_SCRIPTS = {
    "Devanagari": (0x0900, 0x097F),
    "Latin_Ext_A": (0x0100, 0x017F),
    "Latin_Ext_B": (0x0180, 0x024F),
}


def detect_script(text: str) -> List[str]:
    """Detect presence of non-ASCII scripts in text."""
    scripts = set()
    for char in text:
        cp = ord(char)
        for name, (start, end) in NON_ASCII_SCRIPTS.items():
            if start <= cp <= end:
                scripts.add(name)
    return list(scripts)


def profile_ground_truth(gt_path: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Profile ground truth matching relationships."""
    total_s1 = 0
    singletons = 0
    match_counts = Counter()
    s2_matches = 0
    s3_matches = 0
    both_matches = 0
    total_matched_ids = 0

    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        for row in reader:
            if not row or not row[0].strip():
                continue
            total_s1 += 1
            s1_id = row[0].strip()
            matched_str = row[1].strip() if len(row) > 1 else ""

            if not matched_str:
                singletons += 1
                match_counts[0] += 1
            else:
                mids = [m.strip() for m in matched_str.split(",") if m.strip()]
                n = len(mids)
                total_matched_ids += n
                match_counts[n] += 1

                has_s2 = any(m.startswith("S2-") for m in mids)
                has_s3 = any(m.startswith("S3-") for m in mids)
                if has_s2:
                    s2_matches += 1
                if has_s3:
                    s3_matches += 1
                if has_s2 and has_s3:
                    both_matches += 1

            if max_rows and total_s1 >= max_rows:
                break

    non_singletons = total_s1 - singletons
    avg_matches = total_matched_ids / total_s1 if total_s1 > 0 else 0.0
    avg_matches_matched_only = total_matched_ids / non_singletons if non_singletons > 0 else 0.0

    return {
        "total_source1_entities": total_s1,
        "singletons": singletons,
        "singleton_rate": singletons / total_s1 if total_s1 > 0 else 0.0,
        "non_singletons": non_singletons,
        "total_matched_ids": total_matched_ids,
        "avg_matches_per_s1": avg_matches,
        "avg_matches_per_matched_s1": avg_matches_matched_only,
        "s2_match_count": s2_matches,
        "s2_match_rate": s2_matches / total_s1 if total_s1 > 0 else 0.0,
        "s3_match_count": s3_matches,
        "s3_match_rate": s3_matches / total_s1 if total_s1 > 0 else 0.0,
        "both_s2_s3_match_count": both_matches,
        "both_s2_s3_match_rate": both_matches / total_s1 if total_s1 > 0 else 0.0,
        "match_count_distribution": dict(sorted(match_counts.items())),
    }


def profile_source_file(file_path: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Profile an entity source TSV file."""
    total_rows = 0
    missing_names = 0
    missing_addrs = 0
    missing_country = 0
    countries = Counter()
    name_lengths = []
    addr_lengths = []
    name_token_counts = []
    addr_token_counts = []
    has_devanagari_count = 0
    has_accented_count = 0
    legal_suffix_count = Counter()
    leading_decor_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        for row in reader:
            if not row or not row[0].strip():
                continue
            total_rows += 1
            eid = row[0].strip()
            name = row[1].strip() if len(row) > 1 else ""
            addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""

            if not name:
                missing_names += 1
            else:
                n_len = len(name)
                name_lengths.append(n_len)
                n_tokens = len(name.split())
                name_token_counts.append(n_tokens)

                if any(0x0900 <= ord(c) <= 0x097F for c in name):
                    has_devanagari_count += 1
                if any(0x00C0 <= ord(c) <= 0x024F for c in name):
                    has_accented_count += 1

                if re.match(r"^[^a-zA-Z0-9\u0900-\u097F]+", name):
                    leading_decor_count += 1

                lower_name = name.lower()
                for pat in LEGAL_PATTERNS:
                    if re.search(pat, lower_name):
                        legal_suffix_count[pat] += 1

            if not addr:
                missing_addrs += 1
            else:
                addr_lengths.append(len(addr))
                addr_token_counts.append(len(addr.split()))
                if any(0x0900 <= ord(c) <= 0x097F for c in addr):
                    has_devanagari_count += 1

            if not country:
                missing_country += 1
            else:
                countries[country] += 1

            if max_rows and total_rows >= max_rows:
                break

    def calc_stats(lst: List[int]) -> Dict[str, float]:
        if not lst:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        lst.sort()
        n = len(lst)
        med = lst[n // 2] if n % 2 == 1 else (lst[n // 2 - 1] + lst[n // 2]) / 2.0
        return {
            "min": int(lst[0]),
            "max": int(lst[-1]),
            "mean": round(sum(lst) / n, 2),
            "median": round(med, 2),
        }

    return {
        "file_name": os.path.basename(file_path),
        "total_rows": total_rows,
        "missing_names": missing_names,
        "missing_names_pct": round(missing_names / total_rows * 100, 3) if total_rows > 0 else 0,
        "missing_addresses": missing_addrs,
        "missing_addresses_pct": round(missing_addrs / total_rows * 100, 3) if total_rows > 0 else 0,
        "missing_country": missing_country,
        "country_distribution": dict(countries.most_common()),
        "name_char_length_stats": calc_stats(name_lengths),
        "name_token_count_stats": calc_stats(name_token_counts),
        "address_char_length_stats": calc_stats(addr_lengths),
        "address_token_count_stats": calc_stats(addr_token_counts),
        "has_devanagari_rows": has_devanagari_count,
        "has_accented_rows": has_accented_count,
        "has_leading_decor_rows": leading_decor_count,
        "legal_suffix_matches": dict(legal_suffix_count.most_common()),
    }


def generate_markdown_report(
    gt_stats: Dict[str, Any],
    source_stats: Dict[str, Dict[str, Any]],
    output_path: str,
) -> str:
    """Generate comprehensive markdown profiling report."""
    md_lines = [
        "# Dataset Profiling & Exploratory Analysis Report",
        "",
        "## 1. Overview & Dataset Scale",
        "",
        "| Dataset / File | Total Records | Missing Names | Missing Addresses | Top Countries |",
        "| --- | --- | --- | --- | --- |",
    ]

    for fname, st in source_stats.items():
        top_c = ", ".join(f"{c}: {cnt}" for c, cnt in list(st["country_distribution"].items())[:3])
        md_lines.append(
            f"| `{fname}` | {st['total_rows']:,} | {st['missing_names']} ({st['missing_names_pct']}%) | "
            f"{st['missing_addresses']} ({st['missing_addresses_pct']}%) | {top_c} |"
        )

    md_lines.extend([
        "",
        "## 2. Ground Truth Analysis (`train_ground_truth.tsv`)",
        "",
        f"- **Total Source 1 Entities Analyzed:** {gt_stats['total_source1_entities']:,}",
        f"- **True Singletons (No Match):** {gt_stats['singletons']:,} ({gt_stats['singleton_rate'] * 100:.2f}%)",
        f"- **Entities with ≥ 1 Match:** {gt_stats['non_singletons']:,} ({(1 - gt_stats['singleton_rate']) * 100:.2f}%)",
        f"- **Total Matches (S2 + S3):** {gt_stats['total_matched_ids']:,}",
        f"- **Average Matches per S1 Entity:** {gt_stats['avg_matches_per_s1']:.3f}",
        f"- **Average Matches per Non-Singleton S1:** {gt_stats['avg_matches_per_matched_s1']:.3f}",
        f"- **S2 Match Rate:** {gt_stats['s2_match_rate'] * 100:.2f}% ({gt_stats['s2_match_count']:,} entities)",
        f"- **S3 Match Rate:** {gt_stats['s3_match_rate'] * 100:.2f}% ({gt_stats['s3_match_count']:,} entities)",
        f"- **Matches in BOTH S2 & S3:** {gt_stats['both_s2_s3_match_rate'] * 100:.2f}% ({gt_stats['both_s2_s3_match_count']:,} entities)",
        "",
        "### Match Count Distribution",
        "",
        "| Matches Count | Number of S1 Entities | Percentage |",
        "| --- | --- | --- |",
    ])

    tot_s1 = gt_stats["total_source1_entities"]
    for k, v in list(gt_stats["match_count_distribution"].items())[:15]:
        pct = (v / tot_s1 * 100) if tot_s1 > 0 else 0
        md_lines.append(f"| {k} | {v:,} | {pct:.2f}% |")

    md_lines.extend([
        "",
        "## 3. Text Characteristics & Noise Patterns",
        "",
        "| Source | Name Chars (Mean/Max) | Name Tokens (Mean/Max) | Addr Chars (Mean/Max) | Addr Tokens (Mean/Max) | Devanagari | Accented | Leading Decor |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for fname, st in source_stats.items():
        n_c = f"{st['name_char_length_stats']['mean']:.1f} / {st['name_char_length_stats']['max']}"
        n_t = f"{st['name_token_count_stats']['mean']:.1f} / {st['name_token_count_stats']['max']}"
        a_c = f"{st['address_char_length_stats']['mean']:.1f} / {st['address_char_length_stats']['max']}"
        a_t = f"{st['address_token_count_stats']['mean']:.1f} / {st['address_token_count_stats']['max']}"
        md_lines.append(
            f"| `{fname}` | {n_c} | {n_t} | {a_c} | {a_t} | "
            f"{st['has_devanagari_rows']} | {st['has_accented_rows']} | {st['has_leading_decor_rows']} |"
        )

    md_lines.extend([
        "",
        "## 4. Key Modeling Implications",
        "",
        "1. **High Singleton Frequency:** A substantial proportion of Source 1 entities have zero true matches in S2/S3. Correctly predicting empty match lists directly yields 1.0 macro F0.5 per entity.",
        "2. **Open-Set Country Distribution:** Test data contains `France` in addition to `US` and `India`. Normalizers, blockers, and feature encoders must be strictly language- and country-agnostic.",
        "3. **Multilingual Script Support:** Multiple scripts (Devanagari, Latin with French accents) are present. Unicode NFC/NFKD normalization and script-preserving tokenization are required.",
        "4. **Noisy Pre/Suffixes:** Punctuation decorations (e.g. `--`, `<<`, `**`), variable legal entity suffixes (`Pvt Ltd`, `LLC`, `Corp`, `SARL`), and address rearrangements require dual-form representations.",
        "5. **One-to-Many Match Structure:** Source 1 entities frequently match multiple entities in S2 and S3 simultaneously. Classification must score candidate pairs independently and perform entity-level calibrated aggregation rather than strict 1-to-1 bipartite matching.",
    ])

    content = "\n".join(md_lines) + "\n"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content

#!/usr/bin/env python3
"""Validate submission TSV formatting using only the standard library."""

import argparse
import os

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def source_ids(path):
    with open(path, encoding="utf-8") as handle:
        next(handle, None)
        return {line.split("\t", 1)[0].strip() for line in handle if line.strip()}


def parse(path, expected_header, required, errors):
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None
    rows = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != expected_header:
            errors.append(f"{os.path.basename(path)} has invalid header: {header}")
            return None
        for line_number, line in enumerate(handle, 2):
            source_id, separator, values = line.rstrip("\n").partition("\t")
            if not separator:
                errors.append(f"{os.path.basename(path)} line {line_number} has no TAB")
                continue
            if source_id in rows:
                errors.append(f"{os.path.basename(path)} repeats {source_id}")
            ids = [value.strip() for value in values.split(",") if value.strip()]
            if len(ids) != len(set(ids)):
                errors.append(f"{os.path.basename(path)} repeats an ID for {source_id}")
            for entity_id in ids:
                if not entity_id.startswith(("S2-", "S3-")):
                    errors.append(f"{os.path.basename(path)} has invalid ID {entity_id}")
            rows[source_id] = set(ids)
    missing = required - rows.keys()
    extra = rows.keys() - required
    if missing:
        errors.append(f"{os.path.basename(path)} is missing {len(missing)} Source-1 rows")
    if extra:
        errors.append(f"{os.path.basename(path)} has {len(extra)} unknown Source-1 rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()
    errors = []
    source1 = os.path.join(args.test_dir, "test_source1.tsv")
    required = source_ids(source1) if os.path.isfile(source1) else set()
    if not required:
        errors.append(f"Test source1 file not found or empty: {source1}")
    matching = parse(args.matching, MATCH_HEADER, required, errors)
    candidates = parse(args.candidate, CANDIDATE_HEADER, required, errors) if args.candidate else None
    if matching is not None and candidates is not None:
        omitted = {key for key, ids in matching.items() if not ids <= candidates.get(key, set())}
        if omitted:
            print(f"WARNING: {len(omitted)} rows contain matches absent from candidates")
    if args.check_ids and matching is not None:
        valid = set()
        for filename in ("test_source2.tsv", "test_source3.tsv"):
            path = os.path.join(args.test_dir, filename)
            if not os.path.isfile(path):
                errors.append(f"Missing ID source: {path}")
            else:
                valid.update(source_ids(path))
        unknown = {entity_id for ids in matching.values() for entity_id in ids if entity_id not in valid}
        if unknown:
            errors.append(f"{len(unknown)} matched IDs are absent from test Source-2/3")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS - no blocking issues found. Safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

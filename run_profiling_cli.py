"""Run profiling on train and test datasets and generate profiling report."""

import json
import os
import sys
import time

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from config import PathConfig
from profiling import profile_ground_truth, profile_source_file, generate_markdown_report


def main():
    print("=" * 60)
    print("Starting Comprehensive Dataset Profiling...")
    print("=" * 60)
    start_time = time.time()
    paths = PathConfig()

    # Ground truth profiling
    gt_file = paths.train_ground_truth
    print(f"Profiling ground truth: {gt_file}...")
    gt_stats = profile_ground_truth(gt_file, max_rows=200000)
    print(f"  Ground truth profiled in {time.time() - start_time:.1f}s")
    print(f"  Total S1 analyzed: {gt_stats['total_source1_entities']:,}")
    print(f"  Singletons: {gt_stats['singletons']:,} ({gt_stats['singleton_rate']*100:.2f}%)")
    print(f"  Non-singletons: {gt_stats['non_singletons']:,}")
    print(f"  Avg matches per S1: {gt_stats['avg_matches_per_s1']:.3f}")

    # Source files profiling
    sources = [
        paths.train_source1,
        paths.train_source2,
        paths.train_source3,
        paths.test_source1,
        paths.test_source2,
        paths.test_source3,
    ]

    source_stats = {}
    for src in sources:
        t0 = time.time()
        print(f"Profiling {src}...")
        fname = os.path.basename(src)
        stats = profile_source_file(src, max_rows=50000)
        source_stats[fname] = stats
        print(f"  {fname} done in {time.time() - t0:.1f}s (rows sampled: {stats['total_rows']:,})")
        print(f"    Countries: {stats['country_distribution']}")
        print(f"    Missing names: {stats['missing_names']} ({stats['missing_names_pct']}%)")
        print(f"    Missing addrs: {stats['missing_addresses']} ({stats['missing_addresses_pct']}%)")

    # Save JSON stats
    os.makedirs(paths.reports_dir, exist_ok=True)
    stats_json_path = os.path.join(paths.reports_dir, "profiling_stats.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump({"ground_truth": gt_stats, "sources": source_stats}, f, indent=2)
    print(f"\nSaved raw profiling stats to {stats_json_path}")

    # Generate Markdown Report
    report_md_path = paths.profiling_report
    generate_markdown_report(gt_stats, source_stats, report_md_path)
    print(f"Generated Markdown report at {report_md_path}")
    print(f"Total profiling time: {time.time() - start_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Script to package the final submission zip according to ML Challenge 2026 specifications."""

import argparse
import os
import sys
import zipfile

base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, "code", "business_entity_resolution", "src"))
from submission import run_official_validator


def create_submission_zip(team_name: str = "ML_Challenger") -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_matching = os.path.join(base_dir, "output", "matching_results.tsv")
    output_candidate = os.path.join(base_dir, "output", "candidate_pairs.tsv")
    doc_path = os.path.join(base_dir, "Documentation_template.md")
    code_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    test_dir = os.path.join(base_dir, "dataset", "test")

    # 1. Verify files exist
    assert os.path.isfile(output_matching), f"Missing {output_matching}"
    assert os.path.isfile(output_candidate), f"Missing {output_candidate}"
    assert os.path.isfile(doc_path), f"Missing {doc_path}"
    assert os.path.isdir(code_dir), f"Missing {code_dir}"

    # 2. Run official validator first
    print("Running validator before creating ZIP archive...")
    is_pass, val_out = run_official_validator(output_matching, output_candidate, test_dir)
    print(val_out)
    if not is_pass:
        print("ERROR: Output validation failed! Please fix issues before packaging.")
        sys.exit(1)

    zip_filename = f"{team_name}_submission.zip"
    zip_filepath = os.path.join(base_dir, zip_filename)

    print(f"Creating submission zip: {zip_filename}...")
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(output_matching, arcname="output/matching_results.tsv")
        zf.write(output_candidate, arcname="output/candidate_pairs.tsv")

        # Add documentation
        zf.write(doc_path, arcname="Documentation_template.md")

        # Add code files
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".py", ".md", ".txt")) and not file.startswith("."):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    zf.write(full_path, arcname=rel_path)

    print(f"\nSuccessfully generated submission ZIP: {zip_filepath}")
    print(f"ZIP size: {os.path.getsize(zip_filepath) / (1024*1024):.2f} MB")

    # Verify zip contents
    print("\nVerifying archive contents:")
    with zipfile.ZipFile(zip_filepath, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")

    return zip_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ML Challenge 2026 submission zip")
    parser.add_argument("--team-name", default="PrecisionTeam", help="Team name for zip prefix")
    args = parser.parse_args()
    create_submission_zip(args.team_name)

# Amazon ML Challenge 2026 Project Guide

## Purpose

This repository solves the Amazon ML Challenge 2026 business entity resolution task. For each Source 1 business, it predicts zero or more matching Source 2 and Source 3 records. The objective is macro-averaged $F_{0.5}$, so precision is intentionally prioritized over recall.

The implementation is offline-only. It does not call search engines, geocoders, business registries, or other external data services.

## Data Provenance and Offline Guarantee

Only these local inputs are used:

- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`
- `train_ground_truth.tsv` for supervised training and validation
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` for inference
- the locally stored `model_artifact.joblib` when inference-only mode is selected

The source code contains no HTTP client, URL reader, API-key lookup, web scraper, cloud-storage reader, geocoder, or external business-database integration. The dependency list contains ML and text-processing libraries only; installing a package is not a data source. Any model or report produced by this project is derived from the local challenge files above.

## Repository Layout

- `run_pipeline.py`: production training and test inference entry point.
- `run_validation.py`: blocking, feature, model, and threshold experiments.
- `run_profiling_cli.py`: dataset profiling report generator.
- `create_submission_zip.py`: validator-backed submission packaging.
- `code/business_entity_resolution/src/`: pipeline implementation.
- `output/`: generated matching and candidate TSV files; ignored by Git.
- `reports/`: profiling and validation reports.
- `utils/validate_submission.py`: standard-library submission validator.

## Data Placement

The challenge data is intentionally excluded from Git because it is approximately 2.4 GB. Place these seven files in `dataset/train/` and `dataset/test/`:

```text
dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
```

`PathConfig` also detects the extracted bundle layout at `6ab10eb3b23ba_student_resource/student_resource/dataset/`, which is useful for the supplied challenge archive. A clean clone should use the canonical `dataset/` layout.

## How It Works

1. **Load and profile** records with streaming TSV readers.
2. **Normalize** names and addresses while preserving Indic combining marks. Names retain cleaned and legal-suffix-stripped forms; domains and common address forms are normalized.
3. **Block** records using multiple inverted indexes: exact names, stripped names, compact names, rare name/address tokens, and numeric address keys. This reduces the comparison space before fuzzy scoring.
4. **Extract pair features** using RapidFuzz similarities, token overlap, numeric overlap, length ratios, country equality, and cross-field interactions.
5. **Train** a LightGBM pair classifier with positive ground-truth pairs and hard negatives sampled from blocked candidates.
6. **Infer by country** in chunks. Source 2 and Source 3 targets are indexed one country at a time to control memory use.
7. **Apply a precision-heavy threshold** and score-gap rule. Entities with no accepted candidates are emitted with an empty match field.
8. **Validate and package** the two required TSV outputs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/business_entity_resolution/requirements.txt
```

The pinned requirements are suitable for the project environment, but dependency installation may require a current Python version with compatible wheels.

## Reproduce the Workflow

Run commands from the repository root:

```bash
# Inspect data quality and write reports
python run_profiling_cli.py

# Run the reduced validation and ablation suite
python run_validation.py

# Train a production model and generate test predictions
python run_pipeline.py --mode all --train-size 30000 --threshold 0.84

# Or reuse model_artifact.joblib for inference only
python run_pipeline.py --mode inference --threshold 0.84

# Validate formatting and required Source 1 coverage
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

# Create the final archive after validation passes
python create_submission_zip.py --team-name PrecisionTeam
```

The full training and inference runs are large-data jobs. Runtime and memory depend on the machine; use a smaller `--train-size` and `--chunk-size` for a smoke run. The validator's `--check-ids` option performs an additional memory-intensive existence check against all Source 2/3 IDs.

## Output Contract

`matching_results.tsv` must contain exactly:

```text
source1_entity_id<TAB>matched_entity_ids
```

`candidate_pairs.tsv` must contain exactly:

```text
source1_entity_id<TAB>candidate_entity_ids
```

Every test Source 1 ID appears once in both files. Match lists are comma-separated S2/S3 IDs; singleton predictions use an empty value. The pipeline also ensures every accepted match is included in that row's candidate list.

## Main Configuration

`code/business_entity_resolution/src/config.py` contains paths and tunable values, including candidate limits, negative sampling ratio, LightGBM settings, and default thresholds. The path resolver prefers `dataset/` and falls back to the extracted archive only when the canonical data is absent.

The loader reads records with local `open()` calls and does not accept remote paths. To independently audit the restriction, search the Python source for network primitives such as `requests`, `urllib`, `urlopen`, `boto`, and `http`; none are required by this pipeline.

## Reproducibility Notes

- Randomized training and splitting use seed `42` by default.
- The checked-in `model_artifact.joblib` is a reusable trained artifact; regenerate it with `run_pipeline.py --mode train` when changing data or model settings.
- Reports under `reports/` are generated artifacts and should be refreshed after a new experiment.
- The supplied challenge data and generated prediction files are not committed because of their size and because predictions should be regenerated for the exact test data being submitted.

## Compliance

The solution uses only the supplied training/test data and local open-source Python packages. No external lookup or enrichment is part of the pipeline.

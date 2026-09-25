# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

Production-grade, end-to-end Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution**.

Designed strictly to maximize **macro-averaged $F_{0.5}$** across multilingual, open-set, noisy business entity records without external data.

---

## 1. Environment Setup

### Requirements
- Linux / macOS / Windows
- Python 3.9+ (tested up to Python 3.14)
- CPU multi-threading support (12 cores recommended)
- Pinned dependencies in `requirements.txt`

### Installation
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 2. Directory Structure

```
Amazon ML Challenge 2026/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv       # Final matches (evaluated on leaderboard)
│   └── candidate_pairs.tsv        # Exact candidate pairs fed to classifier
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py          # Central paths and pipeline hyperparameters
│       │   ├── data_loader.py     # Streaming TSV reader & stratified train/val split
│       │   ├── profiling.py       # Comprehensive dataset profiling & EDA
│       │   ├── normalization.py   # Multi-representation text & script normalizer
│       │   ├── blocking.py        # High-recall multi-strategy candidate generator
│       │   ├── features.py        # 33 pairwise string, token, numeric similarity features
│       │   ├── model.py           # Model A (Heuristic), Model B (LogReg), Model C (LightGBM)
│       │   ├── calibration.py     # Precision-heavy F0.5 threshold calibration
│       │   ├── inference.py       # Country-partitioned streaming batch inference
│       │   ├── evaluation.py      # Official macro F0.5 evaluation implementation
│       │   ├── submission.py      # Output TSV generation & validator runner
│       │   ├── pipeline.py        # End-to-end training and inference execution
│       │   └── error_analysis.py  # Error categorization and diagnostics
│       ├── README.md              # Reproduction and documentation guide
│       └── requirements.txt       # Pinned production dependencies
├── reports/
│   ├── profiling_report.md        # Dataset profile and ground truth analysis
│   ├── blocking_report.md         # Blocking recall and reduction ratio evaluation
│   ├── experiments.csv            # Empirical validation experiments log
│   ├── experiments.md             # Markdown ablation and experiment summary table
│   └── validation_report.md       # Validation metrics, country breakdowns, feature gains
├── Documentation_template.md      # Comprehensive methodology write-up
├── run_pipeline.py                # Command-line entry point for full pipeline
├── run_validation.py              # Command-line entry point for validation experiments
└── create_submission_zip.py       # Submission packaging and verification script
```

---

## 3. End-to-End Execution Guide

### Step 1: Run Validation & Ablation Experiments
```bash
python run_validation.py
```
This script:
- Creates a stratified validation split of 8,000 Source 1 entities preventing data leakage.
- Evaluates candidate blocking recall (achieving 95.05% recall with only 37.5 candidates per entity).
- Mines 315,000+ realistic hard negatives from blocking.
- Trains and evaluates Model A (Heuristic), Model B (Logistic Regression), Model C (LightGBM Name-only, Address-only, and Full Features).
- Sweeps decision thresholds and optimizes for macro $F_{0.5}$.
- Saves results to `reports/experiments.csv`, `reports/experiments.md`, and `reports/validation_report.md`.

### Step 2: Run Production Training & Test Inference
```bash
python run_pipeline.py --mode all --threshold 0.84
```
This script:
- Trains the production LightGBM classifier on the complete training set with mined hard negatives.
- Discovers open-set countries in `test_source1.tsv` (including `France`, `US`, `India`).
- Streams test entities country-by-country in chunks to maintain a light memory footprint (< 4GB RAM).
- Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Automatically executes `utils/validate_submission.py` to confirm 100% compliance with competition rules.

### Step 3: Validate Outputs
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Step 4: Create Submission Package
```bash
python create_submission_zip.py --team-name PrecisionTeam
```
This produces `<team_name>_submission.zip` containing:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `code/business_entity_resolution/`
- `Documentation_template.md`

---

## 4. Key Methodological Innovations

1. **Dual-Form Normalization:** Keeps both raw-clean and legal-suffix-stripped versions of business names across English, French, Hindi, and Tamil legal entities.
2. **Script-Preserving Accent Handling:** Strips combining diacritics strictly from Latin alphabets (e.g. French accents `é` $\rightarrow$ `e`) while preserving Indic vowel signs (matras in Hindi/Tamil/Gujarati).
3. **Domain Name Normalization:** Strips top-level domains (`.com`, `.org`, `.net`, `.in`) from website-based business records so they match pure business names directly.
4. **Rare Address Token Bridging:** Inverted indexing on rare street names and house numbers retrieves records where names were translated into non-Latin scripts or trade names (DBAs).
5. **Hard Negative Mining:** Samples realistic negatives from blocking candidates that share names or addresses rather than trivial random pairs.
6. **Precision-Weighted $F_{0.5}$ Calibration:** Sweeps decision thresholds and sets optimal cutoff at 0.84 to heavily suppress false merges while accurately identifying singletons (93.93% singleton accuracy).

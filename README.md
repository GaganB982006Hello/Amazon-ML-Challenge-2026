# Amazon ML Challenge 2026: Business Entity Resolution Solution

For the complete setup, data-placement, architecture, reproducibility, and submission guide, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

An end-to-end, production-grade Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

The implementation is strictly offline: it uses only the supplied train/test TSV files and the local model artifact. It has no API keys, network calls, web lookups, geocoding, or external business data dependencies.

This solution is designed specifically to maximize **macro-averaged $F_{0.5}$** (precision-heavy) across multi-million multilingual business entity records across independent data sources without using any external databases, web lookups, or APIs.

---

## Table of Contents
1. [Challenge Overview & Problem Statement](#1-challenge-overview--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Project Directory Structure](#3-project-directory-structure)
4. [How the Solution Works (Stage-by-Stage)](#4-how-the-solution-works-stage-by-stage)
   - [Stage 1: Data Profiling & Exploratory Analysis](#stage-1-data-profiling--exploratory-analysis)
   - [Stage 2: Script-Aware Dual-Form Text Normalization](#stage-2-script-aware-dual-form-text-normalization)
   - [Stage 3: Multi-Strategy High-Recall Blocking](#stage-3-multi-strategy-high-recall-blocking)
   - [Stage 4: 33-Dimensional Pairwise Feature Engineering](#stage-4-33-dimensional-pairwise-feature-engineering)
   - [Stage 5: Supervised Model Training & Hard Negative Mining](#stage-5-supervised-model-training--hard-negative-mining)
   - [Stage 6: Precision-Weighted $F_{0.5}$ Calibration & Singleton Detection](#stage-6-precision-weighted-f_05-calibration--singleton-detection)
   - [Stage 7: Country-Partitioned Scalable Batch Inference](#stage-7-country-partitioned-scalable-batch-inference)
   - [Stage 8: Output TSV Generation & Official Validator Verification](#stage-8-output-tsv-generation--official-validator-verification)
5. [Validation Benchmark & Ablation Study Results](#5-validation-benchmark--ablation-study-results)
6. [Environment Setup & Installation](#6-environment-setup--installation)
7. [How to Run (Step-by-Step)](#7-how-to-run-step-by-step)
   - [Step 1: Run Dataset Profiling](#step-1-run-dataset-profiling)
   - [Step 2: Run Validation & Ablation Experiments](#step-2-run-validation--ablation-experiments)
   - [Step 3: Run Production Training & Test Inference](#step-3-run-production-training--test-inference)
   - [Step 4: Validate Submissions](#step-4-validate-submissions)
   - [Step 5: Generate Final Submission ZIP](#step-5-generate-final-submission-zip)
8. [License & Fair-Play Compliance](#8-license--fair-play-compliance)

---

## 1. Challenge Overview & Problem Statement

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities.

### Key Rules & Requirements:
- **Reference Source:** Source 1 is the clean, deduplicated reference source.
- **Matching Goal:** For **every** Source 1 entity, identify zero, one, or multiple matching records from Source 2 and Source 3.
- **Singletons:** An S1 entity with no matches must output an **empty** string (`""`) for `matched_entity_ids`.
- **Open-Set Country Field:** Training covers `US` and `India`. The test set introduces `France`. The solution dynamically accommodates any country without hard-coding.
- **Evaluation Metric:** **Macro-averaged $F_{0.5}$**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Because precision is weighted $2\times$ over recall, **false positive merges are extremely costly**.
- **Absolute Rule:** **NO EXTERNAL DATA OR LOOKUPS** (no Google, Bing, OpenStreetMap, geocoding APIs, or external business databases).

---

## 2. End-to-End System Architecture

```
RAW TSV DATA (Source 1, Source 2, Source 3)
   │
   ▼
DATASET PROFILING & SCRIPT DETECTION
   │
   ▼
SCRIPT-AWARE DUAL-FORM TEXT NORMALIZATION
   ├── Raw-Clean Form (diacritics stripped for Latin only)
   ├── Legal-Suffix-Stripped Form (Corp, LLC, Pvt Ltd, SARL, etc.)
   └── Domain Name Extractor (.com, .org, .net stripped)
   │
   ▼
MULTI-STRATEGY HIGH-RECALL BLOCKING (Inverted Indexes)
   ├── Exact clean & stripped name index
   ├── Compact alphanumeric index
   ├── Compound Name-Token + House Number / PIN index
   ├── Rare name token inverted index
   └── Rare address token inverted index (bridges across scripts)
   │
   ▼
PAIRWISE CANDIDATE GENERATION (output/candidate_pairs.tsv)
   │
   ▼
33 PAIRWISE SIMILARITY FEATURES (C++ SIMD RapidFuzz)
   ├── Name similarity (Levenshtein, Token Set, Token Sort, Partial, Jaccard)
   ├── Address similarity (Token Set, Token Jaccard, Numeric PIN/House Jaccard)
   ├── Cross-field interaction terms (name × addr, min, max, asymmetric signals)
   └── Source indicators (is_source2, is_source3)
   │
   ▼
LIGHTGBM GBDT PAIR CLASSIFIER (Trained with Hard Negatives)
   │
   ▼
CALIBRATED THRESHOLDING & SCORE-GAP DECISION RULES
   ├── Optimized decision threshold (tau = 0.84)
   ├── Score-gap margin (prevents low-confidence trailing merges)
   └── Singleton detection (scores < tau -> empty match set)
   │
   ▼
FINAL MATCH RESULTS (output/matching_results.tsv)
   │
   ▼
OFFICIAL VALIDATION (utils/validate_submission.py) -> PASS
   │
   ▼
SUBMISSION ZIP CREATION (<team_name>_submission.zip)
```

---

## 3. Project Directory Structure

```
Amazon ML Challenge 2026/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv           # S1 training records (2.2M rows)
│   │   ├── train_source2.tsv           # S2 training records (5.0M rows)
│   │   ├── train_source3.tsv           # S3 training records (5.3M rows)
│   │   └── train_ground_truth.tsv      # S1 -> matched S2/S3 ground truth (2.2M rows)
│   └── test/
│       ├── test_source1.tsv            # S1 test records (1.73M rows)
│       ├── test_source2.tsv            # S2 test records (4.89M rows)
│       └── test_source3.tsv            # S3 test records (5.08M rows)
│
├── output/
│   ├── matching_results.tsv            # Final entity matches (scored on leaderboard)
│   └── candidate_pairs.tsv             # Candidate pairs fed to the classifier
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py             # Package initializer
│       │   ├── config.py               # Paths, seeds, and hyperparameters
│       │   ├── data_loader.py          # Memory-efficient TSV loader & stratified split
│       │   ├── profiling.py            # Dataset EDA, Unicode & ground truth profiling
│       │   ├── normalization.py        # Multi-representation text & script normalizer
│       │   ├── blocking.py             # Multi-strategy candidate generator
│       │   ├── features.py             # 33 pairwise feature engineering module
│       │   ├── model.py                # Heuristic, Logistic Regression, LightGBM models
│       │   ├── calibration.py          # Decision threshold & score-gap optimizer
│       │   ├── inference.py            # Country-partitioned streaming batch inference
│       │   ├── evaluation.py           # Macro F0.5 evaluation implementation
│       │   ├── submission.py           # Output writer & official validator runner
│       │   ├── pipeline.py             # Full end-to-end pipeline orchestrator
│       │   └── error_analysis.py       # Error categorization and case studies
│       ├── README.md                   # Codebase documentation
│       └── requirements.txt            # Pinned dependencies
│
├── reports/
│   ├── profiling_report.md             # Dataset profile and ground truth analysis
│   ├── blocking_report.md              # Blocking recall and reduction ratio evaluation
│   ├── experiments.csv                 # Detailed validation experiments log
│   ├── experiments.md                  # Markdown experiment summary table
│   └── validation_report.md            # Comprehensive validation results & feature gains
│
├── Documentation_template.md           # Completed official methodology write-up
├── run_pipeline.py                     # Root CLI for running the production pipeline
├── run_validation.py                   # Root CLI for running the validation suite
├── run_profiling_cli.py                # Root CLI for running dataset profiling
└── create_submission_zip.py            # Packaging script for competition zip
```

---

## 4. How the Solution Works (Stage-by-Stage)

### Stage 1: Data Profiling & Exploratory Analysis
Streams dataset TSV files line-by-line to extract structural parameters without high memory usage:
- **Ground Truth Statistics:**
  - 2,206,821 Source 1 entities analyzed.
  - **Singletons:** 123,247 entities (5.58%) have zero matches.
  - **One-to-Many Matches:** Average matches per matched S1 entity is 3.666.
  - **Cross-Source Presence:** 86.96% match in S2, 87.93% match in S3, and 80.48% match in both S2 and S3 simultaneously.
- **Multilingual Scripts:** Discovered extensive non-Latin scripts (Devanagari, Tamil, Kannada, Gujarati) alongside French accented Latin characters.
- **Missingness:** S1 address missingness is 0%, while S2 and S3 have ~3.4% missing addresses. Names are 100% complete.

### Stage 2: Script-Aware Dual-Form Text Normalization
- **Selective Diacritic Stripping:** Standard Unicode NFKD accent-stripping destroys Indic combining characters (matras). We implemented a script-aware filter that only strips combining accents from Latin characters (`0x0300` - `0x036F`), preserving Devanagari and Dravidian vowel signs intact.
- **Dual Name Representation:** Preserves both `raw_clean` (lowercase, stripped punctuation) and `stripped_legal` (legal suffixes removed across English, French, Hindi, and Tamil: `Corp`, `Inc`, `Pvt Ltd`, `LLC`, `LLP`, `SARL`, `SAS`, `SCI`, `प्राइवेट लिमिटेड`, `எல்எல்பி`).
- **Domain Name Normalization:** Strips `.com`, `.org`, `.net`, `.in`, `.co.in`, `.biz`, `.info` from business names, allowing website domains (e.g. `jayproducts.com`) to match standard business names (`Jay Products`).
- **Address Standardization:** Normalizes street types (`st` $\rightarrow$ `street`, `rd` $\rightarrow$ `road`, `ave` $\rightarrow$ `avenue`) and state abbreviations, and extracts numeric sets (house numbers, unit numbers, PIN codes).

### Stage 3: Multi-Strategy High-Recall Blocking
Comparing 1.73M test S1 records against 10M test S2+S3 records directly requires 17.3 trillion comparisons ($O(N^2)$). We developed a country-partitioned multi-index blocker:
1. **Exact Clean Name Index:** Direct match on clean business name.
2. **Stripped Legal Name Index:** Direct match ignoring corporate suffixes.
3. **Compact Alphanumeric Index:** Removes whitespace and punctuation.
4. **Name First-Word + House Number/PIN Index:** Connects entities sharing the primary business word and address number.
5. **Rare Name Token Index:** Inverted index on distinct name tokens ($\text{length} \ge 3$, document frequency $\le 250$).
6. **Rare Address Token Index:** Inverted index on locality and street tokens ($\text{length} \ge 4$) to bridge cross-script name translations (e.g., business name translated to Tamil, but street name matches).
7. **House Number + Street Token Index:** Compound address key for high-density metropolitan areas.

**Validation Performance:**
- **Candidate Recall:** **95.05%** (26,272 / 27,639 true matches captured).
- **Average Candidates per S1:** **37.5** candidates.
- **Search Space Reduction Ratio:** **> 99.998%**.

### Stage 4: 33-Dimensional Pairwise Feature Engineering
For each candidate pair, RapidFuzz (C++ SIMD accelerated) computes 33 similarity metrics:
- **Name Features (13):** Exact match booleans, Levenshtein ratio, Token Set ratio, Token Sort ratio, Partial ratio, Token Jaccard, Token Overlap count, Char 3-gram Jaccard, Length difference, Length ratio, Prefix match booleans (4 and 8 chars).
- **Address Features (11):** Exact address match boolean, Levenshtein ratio, Token Set ratio, Token Sort ratio, Token Jaccard, Token Overlap count, Numeric tokens Jaccard, Shared numeric count, Length difference, Length ratio, Target address missingness boolean.
- **Country Features (1):** Exact country match boolean.
- **Cross-Field Interactions (6):** $\text{name} \times \text{addr}$, $\max$, $\min$, strong name + weak address, weak name + strong address, both strong.
- **Source Indicators (2):** `is_source2`, `is_source3`.
- **Fast-Path Optimizations:** Exact string equality triggers immediate bypass of fuzzy calculations, providing a 4x-5x speedup.

### Stage 5: Supervised Model Training & Hard Negative Mining
- **Hard Negative Mining:** Random negative pairs are trivial to distinguish. We mine realistic hard negatives directly from the blocking stage (candidate pairs that share identical streets, common names, or postal codes but are different entities).
- **LightGBM Classifier:** 300 gradient boosted decision trees (`num_leaves=31`, `learning_rate=0.05`, `objective="binary"`).
- **Top Features by Gain:**
  1. `max_sim` (Gain: 1,399,236)
  2. `name_sim_x_addr_sim` (Gain: 1,180,100)
  3. `addr_token_set_ratio` (Gain: 393,611)
  4. `addr_token_jaccard` (Gain: 278,881)

### Stage 6: Precision-Weighted $F_{0.5}$ Calibration & Singleton Detection
- **Threshold Search:** Swept decision thresholds from 0.40 to 0.95 on held-out validation data.
- **Optimal Cutoff ($\tau = 0.84$):** Maximizes macro $F_{0.5}$ by heavily penalizing false merges while maintaining 92.03% recall.
- **Score-Gap Tolerance ($0.25$):** Allows multiple matches for an S1 entity only if secondary candidate probabilities stay within 0.25 of the top candidate.
- **Singleton Detection:** If no candidate exceeds 0.84, an empty match set (`""`) is returned, yielding **93.93% singleton accuracy**.

### Stage 7: Country-Partitioned Scalable Batch Inference
To run inference over 1.73M test entities on 14GB RAM without OOM:
1. Dynamically discovers all open-set countries in `test_source1.tsv` (`US`, `France`, `India`).
2. Processes one country at a time:
   - Loads test targets (S2 + S3) for that country.
   - Builds blocking index.
   - Streams S1 records in chunks of 50,000 entities.
   - Generates candidates, extracts features, predicts scores, and applies decision rules.
   - Streams matches directly to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
   - Cleans memory (`del`, `gc.collect()`) before advancing to the next country.

### Stage 8: Output TSV Generation & Official Validator Verification
Outputs strictly adhere to competition requirements:
- `output/matching_results.tsv`:
  `source1_entity_id<TAB>matched_entity_ids` (comma-separated, no quotes, empty string for singletons).
- `output/candidate_pairs.tsv`:
  `source1_entity_id<TAB>candidate_entity_ids` (every matched ID is guaranteed to be present in candidates).
- Automatically executes `utils/validate_submission.py` to confirm exit code 0 (`PASS`).

---

## 5. Validation Benchmark & Ablation Study Results

Evaluated on a held-out, stratified validation split of **7,999 Source 1 entities** and **299,984 candidate pairs**:

| Experiment | Model | Features Used | Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Exp 1: Baseline** | Weighted Rules (Model A) | Token Sets + Edit Distance | 0.74 | **0.9033** | 0.9508 | 0.8126 | 95.1% |
| **Exp 2: Linear Classifier** | Logistic Regression (Model B) | All 33 Pair Features | 0.94 | **0.9500** | 0.9687 | 0.9166 | 93.0% |
| **Exp 3: Name-Only Ablation** | LightGBM GBDT | 13 Name Features | 0.76 | **0.9443** | 0.9663 | 0.9042 | 89.7% |
| **Exp 4: Addr-Only Ablation** | LightGBM GBDT | 11 Address Features | 0.72 | **0.9104** | 0.9481 | 0.8452 | 91.5% |
| **Exp 5: Full Production Model** | **LightGBM GBDT (Model C)** | **All 33 Pair Features** | **0.84** | **0.95835** | **0.97749** | **0.92030** | **93.93%** |
| **Exp 6: Source-Specific Cutoff** | LightGBM GBDT | All 33 Pair Features | S2:0.85/S3:0.85 | **0.95831** | 0.97780 | 0.91930 | 94.4% |

### Validation Performance by Country:
- **United States (4,788 entities):** Macro $F_{0.5} = \mathbf{0.9748}$, Precision $= 0.9859$, Recall $= 0.9494$
- **India (3,211 entities):** Macro $F_{0.5} = \mathbf{0.9338}$, Precision $= 0.9650$, Recall $= 0.8769$

---

## 6. Environment Setup & Installation

### Requirements:
- Python 3.9+ (tested on Python 3.14)
- Virtual environment support
- 8 GB+ RAM recommended

### Setup Command:
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install pinned requirements
pip install -r code/business_entity_resolution/requirements.txt
```

The challenge dataset is intentionally not committed because it is approximately 2.4 GB. Place the seven supplied TSV files under `dataset/train/` and `dataset/test/`; the path resolver also supports the extracted `6ab10eb3b23ba_student_resource/student_resource/dataset/` layout.

---

## 7. How to Run (Step-by-Step)

### Step 1: Run Dataset Profiling
To analyze row counts, missing fields, script presence, and ground truth statistics:
```bash
python run_profiling_cli.py
```
Output saved to: `reports/profiling_report.md` and `reports/profiling_stats.json`.

### Step 2: Run Validation & Ablation Experiments
To train models on the training split and evaluate all 6 ablation experiments:
```bash
python run_validation.py
```
Outputs saved to:
- `reports/experiments.csv`
- `reports/experiments.md`
- `reports/blocking_report.md`
- `reports/validation_report.md`

### Step 3: Run Production Training & Test Inference
To train the production model on full training data and generate the final competition predictions:
```bash
python run_pipeline.py --mode all --threshold 0.84
```
This automatically produces:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

### Step 4: Validate Submissions
To verify the output files against the official competition submission validator:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
You should see:
```
PASS — no blocking issues found. Safe to submit.
```

### Step 5: Generate Final Submission ZIP
To package the final verified submission archive:
```bash
python create_submission_zip.py --team-name PrecisionTeam
```
This produces `PrecisionTeam_submission.zip` containing:
```
PrecisionTeam_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 8. License & Fair-Play Compliance

- **No External Data:** The pipeline operates 100% offline. No external APIs, commercial entity-resolution services, government business registries, or geocoding services were used.
- **Model Licensing:** All models and dependencies (`lightgbm`, `scikit-learn`, `rapidfuzz`, `scipy`, `numpy`, `pandas`) are licensed under MIT, BSD, or Apache 2.0 licenses.
- **Model Parameter Count:** The LightGBM GBDT model contains fewer than 100,000 parameters, well below the 8 billion parameter limit.

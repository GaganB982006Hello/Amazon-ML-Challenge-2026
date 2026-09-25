# Dataset Profiling & Exploratory Analysis Report

## 1. Overview & Dataset Scale

| Dataset / File | Total Records | Missing Names | Missing Addresses | Top Countries |
| --- | --- | --- | --- | --- |
| `train_source1.tsv` | 50,000 | 0 (0.0%) | 0 (0.0%) | US: 29965, India: 20035 |
| `train_source2.tsv` | 50,000 | 0 (0.0%) | 1682 (3.364%) | US: 30097, India: 19903 |
| `train_source3.tsv` | 50,000 | 0 (0.0%) | 1740 (3.48%) | US: 29790, India: 20210 |
| `test_source1.tsv` | 50,000 | 0 (0.0%) | 0 (0.0%) | India: 23300, US: 19215, France: 7485 |
| `test_source2.tsv` | 50,000 | 0 (0.0%) | 1331 (2.662%) | India: 23534, US: 19238, France: 7228 |
| `test_source3.tsv` | 50,000 | 0 (0.0%) | 1380 (2.76%) | India: 23690, US: 19249, France: 7061 |

## 2. Ground Truth Analysis (`train_ground_truth.tsv`)

- **Total Source 1 Entities Analyzed:** 200,000
- **True Singletons (No Match):** 11,194 (5.60%)
- **Entities with ≥ 1 Match:** 188,806 (94.40%)
- **Total Matches (S2 + S3):** 693,069
- **Average Matches per S1 Entity:** 3.465
- **Average Matches per Non-Singleton S1:** 3.671
- **S2 Match Rate:** 87.01% (174,027 entities)
- **S3 Match Rate:** 87.98% (175,950 entities)
- **Matches in BOTH S2 & S3:** 80.59% (161,171 entities)

### Match Count Distribution

| Matches Count | Number of S1 Entities | Percentage |
| --- | --- | --- |
| 0 | 11,194 | 5.60% |
| 1 | 10,690 | 5.34% |
| 2 | 33,695 | 16.85% |
| 3 | 48,320 | 24.16% |
| 4 | 43,899 | 21.95% |
| 5 | 29,276 | 14.64% |
| 6 | 15,054 | 7.53% |
| 7 | 5,781 | 2.89% |
| 8 | 1,616 | 0.81% |
| 9 | 420 | 0.21% |
| 10 | 51 | 0.03% |
| 11 | 4 | 0.00% |

## 3. Text Characteristics & Noise Patterns

| Source | Name Chars (Mean/Max) | Name Tokens (Mean/Max) | Addr Chars (Mean/Max) | Addr Tokens (Mean/Max) | Devanagari | Accented | Leading Decor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train_source1.tsv` | 24.1 / 71 | 3.5 / 12 | 52.0 / 201 | 8.0 / 34 | 0 | 0 | 74 |
| `train_source2.tsv` | 25.0 / 104 | 3.5 / 15 | 47.7 / 176 | 7.5 / 29 | 5260 | 2902 | 3046 |
| `train_source3.tsv` | 25.2 / 80 | 3.5 / 13 | 48.5 / 198 | 7.4 / 31 | 4154 | 3187 | 2167 |
| `test_source1.tsv` | 23.8 / 75 | 3.5 / 11 | 57.0 / 216 | 8.6 / 32 | 0 | 1189 | 324 |
| `test_source2.tsv` | 25.7 / 74 | 3.6 / 11 | 51.6 / 211 | 8.0 / 34 | 6531 | 3958 | 3571 |
| `test_source3.tsv` | 25.7 / 78 | 3.6 / 11 | 50.1 / 213 | 7.7 / 36 | 4830 | 4077 | 2447 |

## 4. Key Modeling Implications

1. **High Singleton Frequency:** A substantial proportion of Source 1 entities have zero true matches in S2/S3. Correctly predicting empty match lists directly yields 1.0 macro F0.5 per entity.
2. **Open-Set Country Distribution:** Test data contains `France` in addition to `US` and `India`. Normalizers, blockers, and feature encoders must be strictly language- and country-agnostic.
3. **Multilingual Script Support:** Multiple scripts (Devanagari, Latin with French accents) are present. Unicode NFC/NFKD normalization and script-preserving tokenization are required.
4. **Noisy Pre/Suffixes:** Punctuation decorations (e.g. `--`, `<<`, `**`), variable legal entity suffixes (`Pvt Ltd`, `LLC`, `Corp`, `SARL`), and address rearrangements require dual-form representations.
5. **One-to-Many Match Structure:** Source 1 entities frequently match multiple entities in S2 and S3 simultaneously. Classification must score candidate pairs independently and perform entity-level calibrated aggregation rather than strict 1-to-1 bipartite matching.

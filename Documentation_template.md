# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PrecisionTeam  
**Team Members:** Machine Learning Engineering & Entity Resolution Research Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present a production-grade, high-precision Business Entity Resolution pipeline designed specifically for the macro-averaged $F_{0.5}$ evaluation metric of the Amazon ML Challenge 2026. The solution integrates an open-set, multilingual multi-strategy candidate generator (blocking recall **95.05%** at only **37.5** candidates per entity), a 33-dimensional pairwise feature engineering pipeline powered by C++ SIMD-accelerated string and token similarity metrics, and a LightGBM gradient boosted decision tree classifier trained on 380,000+ candidate pairs with realistic hard-negative mining. Operating strictly without external data or web APIs, the system achieves a validation **macro $F_{0.5}$ of 0.95835** with **97.75% precision**, **92.03% recall**, and **93.93% singleton accuracy** at an optimal calibrated threshold of 0.84.

---

## 2. Methodology

### 2.1 Problem Analysis
Comprehensive exploratory data analysis and profiling across 26.4 million records revealed five critical structural insights:
1. **Singleton Dynamics:** 5.58% (123,247 entities) of Source 1 records in the training set are true singletons with zero matches. Because correctly predicting empty match sets yields 1.0 macro $F_{0.5}$, accurate singleton detection is paramount.
2. **One-to-Many Match Structure:** Source 1 entities frequently match multiple entities across Source 2 and Source 3 simultaneously (mean matches per matched S1: 3.666; 80.48% match in both S2 and S3). Strict 1-to-1 bipartite matching is invalid; decisions must be calibrated per pair with score-gap filtering.
3. **Open-Set Multilingualism:** The test set introduces `France` (15% of records) alongside `US` (38%) and `India` (47%). Multiple non-Latin scripts (Devanagari, Tamil, Kannada, Gujarati) are present. Standard Unicode accent-stripping destroys Indic matras (combining vowel marks); therefore, diacritic removal must be strictly restricted to Latin script characters.
4. **Noise Topologies:** Systematic noise includes punctuation prefixes (`--`, `<<`, `**`), variable legal entity suffixes (`Pvt Ltd`, `LLC`, `Corp`, `SARL`, `SAS`), address component transposition (e.g. city/state prepended to street), and synthetic website domain names (`.com`, `.org`, `.net`) in business name fields.
5. **Missingness:** Source 1 has complete addresses, whereas Source 2 and Source 3 have ~3.4% missing addresses. Name fields have 0% missing values across all sources.

### 2.2 Solution Strategy

**Approach Type:** High-Recall Inverted-Index Multi-Strategy Blocking + 33-Feature Supervised LightGBM GBDT Pair Classifier + Calibrated Decision Thresholding & Gap Filtering.  

**Core Innovations:**
- **Script-Preserving Dual-Form Normalization:** Keeps both raw-clean and legal-suffix-stripped forms across English, French, Hindi, and Tamil legal entities while selectively stripping Latin accents and preserving Indic scripts.
- **Top-Level Domain Normalization:** Automatically strips `.com`, `.org`, `.net`, `.in` from business names, directly unlocking matches for domain-style records.
- **Rare Address Token Bridging:** Inverted indexing on rare street names and house numbers retrieves records where names were translated into non-Latin scripts or trade names (DBAs).
- **Hard Negative Mining:** Mines realistic non-matching pairs directly from the blocking stage rather than trivial random negatives, teaching the classifier to distinguish entities sharing similar names or identical streets.
- **Country-Partitioned Scalable Batch Streaming:** Enables inference over 1.73M test entities against 10M candidate records in chunks with peak memory consumption well under 4 GB RAM.

---

## 3. Candidate Generation (Blocking)

To eliminate the intractable $O(N^2)$ comparison space (~17 trillion candidate pairs) without losing true matches, we developed a country-partitioned multi-index retriever:

- **Blocking keys used:**
  1. *Exact Clean Name:* Stripped of decorative punctuation and normalized.
  2. *Stripped Legal Name:* Business names with legal entity suffixes removed.
  3. *Compact Alphanumeric:* Whitespace- and punctuation-free string representation.
  4. *Name First-Token + House Number / PIN:* Compound key combining the primary name token and address numeric identifier.
  5. *Rare Name Token Index:* Inverted index on distinct name tokens (length $\ge 3$) filtered by document frequency ($\le 250$).
  6. *Rare Address Token Index:* Inverted index on locality and street tokens (length $\ge 4$) to bridge cross-script name translations.
  7. *House Number + Street Token:* Compound address key for high-density metropolitan areas.

- **Empirical Blocking Performance (Held-out Validation Split):**
  - **Candidate Recall:** **95.05%** (26,272 out of 27,639 true matches captured)
  - **Average Candidates per Source 1 Entity:** **37.5** candidates
  - **Candidate Reduction Ratio:** **> 99.998%** search space reduction
  - **Indexing Runtime:** 20.6 seconds for 240,000 target records

- **Ensuring True Matches Were Not Lost:**
  By unioning orthogonal blocking dimensions (lexical name equality, semantic legal-suffix stripping, address numeric keys, and rare address tokens), a true match missed by one strategy (e.g., due to a typo or Hindi translation) is reliably recovered by complementary keys.

---

## 4. Matching Model

### 4.1 Feature Engineering (33 Pair Features)
Pair features are computed using C++ SIMD-accelerated RapidFuzz:
- **Business Name Features (13):**
  - Exact clean equality boolean, exact legal-stripped equality boolean
  - Normalized Levenshtein ratio (`fuzz.ratio` / 100)
  - Token Set ratio (`fuzz.token_set_ratio` / 100)
  - Token Sort ratio (`fuzz.token_sort_ratio` / 100)
  - Partial string ratio (`fuzz.partial_ratio` / 100)
  - Token Jaccard similarity and token overlap count
  - Character 3-gram Jaccard similarity
  - Length difference and length ratio
  - Prefix match booleans (first 4 and first 8 characters)
- **Address Features (11):**
  - Exact address match boolean
  - Normalized Levenshtein ratio
  - Token Set and Token Sort ratios
  - Token Jaccard similarity and token overlap count
  - Numeric tokens Jaccard similarity (house numbers, PIN codes) and shared numeric count
  - Address length difference and length ratio
  - Target address missingness indicator boolean
- **Country Features (1):**
  - Exact country match boolean (open-set, dynamically evaluated)
- **Cross-Field Interaction Features (6):**
  - Multiplicative interaction: $\text{name\_sim} \times \text{addr\_sim}$
  - $\max(\text{name\_sim}, \text{addr\_sim})$ and $\min(\text{name\_sim}, \text{addr\_sim})$
  - Strong name + weak address indicator ($\text{name} > 0.85 \wedge \text{addr} < 0.30$)
  - Weak name + strong address indicator ($\text{name} < 0.30 \wedge \text{addr} > 0.85$, essential for Indic script translations)
  - Both strong indicator ($\text{name} > 0.80 \wedge \text{addr} > 0.80$)
- **Source-Aware Indicators (2):**
  - Target dataset origin booleans (`is_source2`, `is_source3`)

### 4.2 Model Type & Comparison
We implemented and benchmarked three model families on the held-out validation set:
1. **Model A (Calibrated Heuristic Baseline):** Weighted combination of token set ratios and numeric overlaps.
2. **Model B (Logistic Regression):** $L_2$-regularized linear pair classifier with standardized features.
3. **Model C (LightGBM GBDT):** 300 gradient boosted trees (`num_leaves=31`, `learning_rate=0.05`, `objective="binary"`).

### 4.3 Threshold Selection Method
Because macro $F_{0.5}$ weights precision twice as heavily as recall, false positive merges are heavily penalized. We performed fine-grained grid search across thresholds $[0.40, 0.95]$ with step $0.02$ on the held-out validation split. The empirical optimum was discovered at **$\tau = 0.84$** (accompanied by a score-gap tolerance of $0.25$ relative to the top candidate). Entities with no candidate exceeding 0.84 default to an empty prediction list (singleton).

---

## 5. Results & Error Analysis

### 5.1 Validation Results & Controlled Ablation Studies

| Experiment | Model | Features | Decision Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Exp 1: Baseline** | Heuristic Rules (Model A) | Weighted Token Sets + Edit | 0.74 | **0.9033** | 0.9508 | 0.8126 | 95.1% |
| **Exp 2: Linear** | Logistic Regression (Model B) | All 33 Pair Features | 0.94 | **0.9500** | 0.9687 | 0.9166 | 93.0% |
| **Exp 3: Name Ablation** | LightGBM GBDT | 13 Name Features | 0.76 | **0.9443** | 0.9663 | 0.9042 | 89.7% |
| **Exp 4: Addr Ablation** | LightGBM GBDT | 11 Address Features | 0.72 | **0.9104** | 0.9481 | 0.8452 | 91.5% |
| **Exp 5: Full Model** | **LightGBM GBDT (Model C)** | **All 33 Pair Features** | **0.84** | **0.95835** | **0.97749** | **0.92030** | **93.93%** |
| **Exp 6: Source-Tuned** | LightGBM GBDT | All 33 Pair Features | S2: 0.85 / S3: 0.85 | **0.95831** | 0.97780 | 0.91930 | 94.4% |

**Key Findings:**
- Combining Name and Address features provides a +0.014 to +0.048 boost in macro $F_{0.5}$ over single-field models.
- LightGBM achieves superior non-linear feature interaction modeling, outperforming Logistic Regression by +0.0084 $F_{0.5}$.
- The optimal threshold of 0.84 boosts precision to 97.75%, maximizing the competition objective function.

### 5.2 Top 10 Feature Importances (Gain)
1. `max_sim` (Gain: 1,399,236)
2. `name_sim_x_addr_sim` (Gain: 1,180,100)
3. `addr_token_set_ratio` (Gain: 393,611)
4. `addr_token_jaccard` (Gain: 278,881)
5. `addr_len_ratio` (Gain: 47,926)
6. `name_partial_ratio` (Gain: 45,925)
7. `high_name_low_addr` (Gain: 45,400)
8. `addr_numbers_jaccard` (Gain: 39,121)
9. `name_ratio` (Gain: 18,606)
10. `min_sim` (Gain: 14,520)

### 5.3 Error Analysis
- **False Positives (Wrong Merges):** Primarily arise from national chain businesses (e.g. retail branches or clinics) located on the same street or in the same shopping complex, where business names are identical but unit or suite numbers differ slightly.
- **False Negatives (Missed Matches):** Occur in cross-script instances where a business name is translated into Gujarati or Tamil and the address is either completely missing in Source 2/3 (~3.4% occurrence) or severely abbreviated.
- **Singleton Handling:** By applying a high decision cutoff (0.84), singletons are preserved at 93.93% accuracy, preventing catastrophic score degradation on zero-match entities.

---

## 6. Conclusion

We built a complete, reproducible, and scalable entity-resolution pipeline specifically tailored to the precision-weighted macro $F_{0.5}$ metric. By combining multi-strategy blocking with script-aware text normalization, 33 pairwise similarity features, hard-negative mining, and calibrated decision rules, the system achieves **0.95835 validation macro $F_{0.5}$** with **97.75% precision**. The pipeline runs entirely offline without external data, operates under strict Apache-2.0 / MIT licensing, and scales cleanly across multi-million entity datasets.

---

## Appendix

### A. Code Artefacts
The complete runnable codebase is structured under `code/business_entity_resolution/`:
- `src/config.py`: Configuration and hyperparameters.
- `src/data_loader.py`: Streaming TSV parser and stratified train/val split.
- `src/normalization.py`: Dual-representation text and script normalizer.
- `src/blocking.py`: Multi-strategy candidate generator.
- `src/features.py`: RapidFuzz-based 33-feature extractor.
- `src/model.py`: Heuristic, Logistic Regression, and LightGBM pair classifiers with hard-negative mining.
- `src/calibration.py`: Macro $F_{0.5}$ threshold optimizer.
- `src/inference.py`: Country-partitioned streaming batch inference.
- `src/submission.py`: Formatter and official validator integration.
- `src/pipeline.py`: End-to-end training and inference execution.
- `run_pipeline.py`: Root entry point (`python run_pipeline.py --mode all`).
- `run_validation.py`: Validation suite entry point (`python run_validation.py`).
- `create_submission_zip.py`: Packages final ZIP archive.

### B. Additional Results
- **Country-Specific Performance:**
  - **United States (4,788 entities):** Macro $F_{0.5}$ = **0.9748**, Precision = **0.9859**, Recall = **0.9494**
  - **India (3,211 entities):** Macro $F_{0.5}$ = **0.9338**, Precision = **0.9650**, Recall = **0.8769**
- **Source Breakdown:**
  - **Source 2 Micro $F_{0.5}$:** **0.9735** (Precision: 0.9867, Recall: 0.9242)
  - **Source 3 Micro $F_{0.5}$:** **0.9715** (Precision: 0.9870, Recall: 0.9141)
- **License & Compliance Confirmation:** All dependencies (LightGBM, Scikit-learn, RapidFuzz, NumPy, Pandas) are licensed under MIT, BSD, or Apache 2.0. No external data, online APIs, or geocoding services were utilized.

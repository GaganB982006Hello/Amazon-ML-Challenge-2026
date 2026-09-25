# Comprehensive Validation & Error Analysis Report

- **Total Validation S1 Entities:** 7,999
- **Best Model:** LightGBM with Full Pair Features & Hard Negatives
- **Optimal Decision Threshold:** 0.84
- **Best Validation Macro F0.5:** 0.95835
- **Macro Precision:** 0.97749
- **Macro Recall:** 0.92030
- **Singleton Accuracy:** 93.93%
- **Candidate Recall (Blocking):** 95.05%

## Country Breakdown

| Country | Entities | Macro F0.5 | Precision | Recall |
| --- | --- | --- | --- | --- |
| India | 3,211 | 0.9338 | 0.9650 | 0.8769 |
| US | 4,788 | 0.9748 | 0.9859 | 0.9494 |

## Source Performance Breakdown

- **Source 2 Micro F0.5:** 0.9735 (Precision: 0.9867, Recall: 0.9242)
- **Source 3 Micro F0.5:** 0.9715 (Precision: 0.987, Recall: 0.9141)

## Top 10 Feature Importances

| Rank | Feature | Importance (Gain) |
| --- | --- | --- |
| 1 | `max_sim` | 1399236.38 |
| 2 | `name_sim_x_addr_sim` | 1180100.55 |
| 3 | `addr_token_set_ratio` | 393611.13 |
| 4 | `addr_token_jaccard` | 278881.80 |
| 5 | `addr_len_ratio` | 47926.21 |
| 6 | `name_partial_ratio` | 45925.45 |
| 7 | `high_name_low_addr` | 45400.32 |
| 8 | `addr_numbers_jaccard` | 39121.26 |
| 9 | `name_ratio` | 18606.37 |
| 10 | `min_sim` | 14519.69 |

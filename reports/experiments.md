# Entity Resolution Experiment Suite Results

Macro F0.5 evaluation results on held-out stratified validation split.

| Experiment | Model | Features | Threshold | Macro F0.5 | Precision | Recall | Singleton Acc | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Exp 1: Heuristic Similarity Baseline** | Weighted Rules (Model A) | Token Sets + Edit Ratio + Numbers | 0.74 | **0.9033** | 0.9508 | 0.8126 | 95.1% | 2.09s |
| **Exp 2: Logistic Regression** | Logistic Regression (Model B) | All 33 Pair Features | 0.94 | **0.9500** | 0.9687 | 0.9166 | 93.0% | 4.0s |
| **Exp 3: LightGBM (Name Only)** | LightGBM GBDT | 13 Name Features | 0.76 | **0.9443** | 0.9663 | 0.9042 | 89.7% | 4.48s |
| **Exp 4: LightGBM (Address Only)** | LightGBM GBDT | 11 Address Features | 0.72 | **0.9104** | 0.9481 | 0.8452 | 91.5% | 3.47s |
| **Exp 5: LightGBM (Full Features)** | LightGBM GBDT | All 33 Features (Name+Addr+Cross+Source) | 0.84 | **0.9584** | 0.9775 | 0.9203 | 93.9% | 7.67s |
| **Exp 6: Source-Specific Calibration** | LightGBM GBDT | All 33 Features | S2:0.85/S3:0.85 | **0.9583** | 0.9778 | 0.9193 | 94.4% | 4.62s |

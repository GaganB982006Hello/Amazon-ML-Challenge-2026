# 1.How to Run

Follow these instructions to set up the environment and execute the pipeline.

## 1. Environment Setup

Create and activate a virtual environment, then install the required dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Dataset Setup

Ensure your datasets are placed inside the dataset/ directory with the following structure:

```bash
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Execution Commands

Run Full Pipeline & Test Inference
Generates the final entity matching output in output/matching_results.tsv:

```bash
python run_pipeline.py
```

Validate Submission Output
Verifies that the generated submission file conforms to the required schema:

```bash

python utils/validate_submission.py
```

Run Validation & Evaluation (Optional)Evaluates performance and macro $F_{0.5}$ metric scores on held-out validation data

```bash
python run_validation.py
```

Create Submission ZIP (Optional)
Packages the project source code and output into a final submission archive:

```bash
python create_submission_zip.py
```

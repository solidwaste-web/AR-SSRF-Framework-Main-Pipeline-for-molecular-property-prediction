# AR-SSRF Framework
[![DOI](https://zenodo.org/badge/1177781749.svg)](https://doi.org/10.5281/zenodo.18950483)
**Autoregressive Semi-Supervised Random Forest for Molecular Property Prediction**

---

## ⚠️ IMPORTANT NOTICE

This repository is currently **provided for peer review purposes only**.

The code implements the methodology described in the submitted manuscript and is shared to allow reviewers to verify the workflow and reproduce the results.

### Usage Restrictions

- This code is **NOT licensed for public reuse or redistribution**
- It is **only intended for academic peer review**
- Any reuse, modification, or redistribution requires **explicit permission from the authors**
- **Commercial use is strictly prohibited**

For licensing or collaboration inquiries, please contact the corresponding author.

---

# Overview

The **AR-SSRF (Autoregressive Semi-Supervised Random Forest)** framework is a complete machine learning pipeline for **molecular property prediction** using **semi-supervised learning**.

The framework integrates:

- Molecular representation learning
- Feature engineering
- Model selection
- Class imbalance handling
- Iterative pseudo-label expansion

The core idea is to **iteratively expand the labeled dataset using high-confidence predictions from unlabeled samples**, allowing the model to learn from both labeled and unlabeled molecular data.

The entire workflow is implemented as a modular pipeline and can be executed end-to-end via the main script.

---

# Framework Architecture

The AR-SSRF workflow consists of **five major stages**:

```
Raw Dataset
     │
     ▼
1. Data Preprocessing
     │
     ▼
2. Molecular Feature Extraction (17 feature types)
     │
     ▼
3. Feature Engineering & Filtering
     │
     ▼
4. Base Classifier Evaluation (85 combinations)
     │
     ▼
5. Autoregressive Semi-Supervised Training
     │
     ▼
Final AR-SSRF Model
```

---

# Key Features

- SMILES standardization and duplicate aggregation
- 17 molecular representation approaches
- Comprehensive descriptor + fingerprint generation
- Feature quality control and redundancy filtering
- 85 model–feature combinations evaluation
- Imbalance-aware dual-end sampling
- Autoregressive pseudo-label learning
- Early stopping with convergence monitoring

---

# System Requirements

## Operating System

- Linux (Ubuntu 18.04+ recommended)
- macOS
- Windows 10 / 11 (WSL recommended)

## Hardware Requirements

Minimum:

- 8 GB RAM
- 4 CPU cores

Recommended:

- 16+ GB RAM
- 8+ CPU cores

Storage:

- 10 GB free space

---

# Python Environment

Supported versions:

```
Python 3.8 – 3.10
```

---

# Dependencies

### Core Libraries

```
numpy==1.23.5
pandas==1.5.3
scipy==1.10.1
scikit-learn==1.2.2
```

### Molecular Processing

```
rdkit==2023.3.2
mordred==1.2.0
padelpy
```

### Machine Learning

```
xgboost==1.7.5
lightgbm==3.3.5
imbalanced-learn==0.10.1
```

### Deep Learning (Optional)

```
torch==2.0.1
torch-geometric==2.3.1
dgl==1.1.1
```

### Utilities

```
tqdm==4.65.0
joblib==1.2.0
matplotlib==3.7.1
seaborn==0.12.2
```

---

# Installation

## 1. Create Environment

Using **conda** (recommended)

```bash
conda create -n arssrf python=3.9
conda activate arssrf
```

or using **venv**

```bash
python -m venv arssrf_env
source arssrf_env/bin/activate
```

---

## 2. Install RDKit

```bash
conda install -c conda-forge rdkit=2023.3.2
```

---

## 3. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify Installation

```bash
python -c "import rdkit; print(rdkit.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

---

# Project Structure

```
AR-SSRF/
│
├── README.md
├── requirements.txt
├── main.py
│
├── data_preprocessing.py
├── comprehensive_molecular_feature_extraction.py
├── feature_engineering_module.py
├── base_classifier_evaluation.py
├── imbalance_aware_dual_end_sampling_strategy.py
├── autoregressive_iterative_learning.py

---

# Module Description

## 1. Data Preprocessing

This module performs dataset cleaning and standardization.

Main functions:

- SMILES canonicalization
- invalid structure removal
- duplicate sample aggregation
- IQR-based outlier filtering

Duplicate aggregation strategy:

| Observations | Processing |
|---|---|
| 1 value | use directly |
| 2 values | mean |
| ≥3 values | remove outliers using IQR then mean |

Output:

```
preprocessed_data.csv
```

---

# 2. Molecular Feature Extraction

The framework extracts **17 types of molecular representations** including descriptors and fingerprints.

### Molecular Descriptors

- RDKit descriptors
- Mordred descriptors
- PaDEL descriptors

### Structural Fingerprints

- MACCS
- Morgan
- RDKit
- AtomPairs
- GraphOnly
- PubChem
- Extended
- EState

### Fragment / Substructure Fingerprints

- Klekota-Roth
- Substructure
- Topological torsion
- PaDEL fingerprints

Output:

```
features/
```

---

# 3. Feature Engineering

Feature filtering pipeline includes:

- row-wise NaN filtering
- column-wise NaN filtering
- variance filtering
- correlation filtering

Default parameters:

```
row_nan_threshold = 0.30
col_nan_threshold = 0.10
variance_threshold = 0.01
correlation_threshold = 0.9
```

Output:

```
processed_features/
```

---

# 4. Base Classifier Evaluation

Five machine learning algorithms are evaluated with all feature sets.

Algorithms:

- Logistic Regression
- Random Forest
- XGBoost
- SVM
- KNN

Total combinations:

```
5 models × 17 feature sets = 85 experiments
```

Performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- MCC

The best configuration is selected automatically.

Output:

```
base_classifier_results.csv
```

---

# 5. Autoregressive Semi-Supervised Learning

The AR-SSRF training process expands the labeled dataset using **pseudo-labeled samples**.

Workflow per iteration:

```
1. Train classifier on labeled dataset
2. Predict probabilities on unlabeled pool
3. Select high-confidence samples
4. Assign pseudo labels
5. Add samples to training set
6. Repeat until convergence
```

The training loop includes:

- validation monitoring
- early stopping
- convergence detection

---

# Imbalance-Aware Dual-End Sampling

To address class imbalance, AR-SSRF uses a **dual-end sampling strategy**.

Sample selection rules:

```
Positive samples: P(y=1) ≥ threshold_high
Negative samples: P(y=1) ≤ threshold_low
```

Default thresholds:

```
threshold_high = 0.9
threshold_low = 0.1
```

---

# Running the Pipeline

## Basic Usage

```bash
python main.py --input dataset.csv
```

---

## Custom Output Directory

```bash
python main.py \
--input dataset.csv \
--output results/
```

---

## Using Custom Configuration

```bash
python main.py \
--input dataset.csv \
--config config.json
```

---

# Configuration Parameters

Example configuration:

```json
{
  "threshold_high": 0.9,
  "threshold_low": 0.1,
  "max_iterations": 50,
  "patience": 3,
  "balance_ratio": 1.0,
  "validation_split": 0.2
}
```

---

# Output Files

After execution, the pipeline generates:

```
output/
│
├── preprocessed_data.csv
├── features/
├── base_classifier_results.csv
├── ar_ssrf_model.pkl
└── training_history.csv
```

---

# Citation

If you use this framework in your research, please cite the associated manuscript (currently under review).

---

# Contact


For questions or collaboration requests, please contact the corresponding author.

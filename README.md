# CAP5610-ML-Project

Machine learning course project for multiclass text classification using the **DBPedia-14** dataset. This repository compares traditional machine learning models and deep learning models using a shared dataset, shared preprocessing workflow, and common evaluation metrics.

## Team Members

- Courtney Prater
- Matthew Rampersad
- Haida StarEagle
- Sky Zentner
- Kelly Wells

---

# Project Goal

The goal of this project is to compare multiple traditional and deep learning approaches on the same text classification task.  
Each team member is responsible for implementing **one traditional ML model and one deep learning model**.

---

# Dataset

**Dataset:** DBPedia-14  
**Task:** Multiclass text classification  
**Source:** https://huggingface.co/datasets/fancyzhx/dbpedia_14

### Dataset Summary

- 14 classes
- 560,000 training samples
- 70,000 test samples
- 630,000 total samples

Each record contains:

- `label`
- `title`
- `content`

Example:

label: 3  
title: The Rolling Stones  
content: The Rolling Stones are an English rock band formed in London...

---

# Repository Structure

```text
CAP5610-ML-Project/
│   .gitignore
│   dataset_stats.py
│   README.md
│   requirements.txt
│
├── data/
│   └── download_dataset.py
│
├── deep_models/
│   ├── cnn_text_classifier.py
│   ├── llm_finetune.py
│   ├── lstm_model.py
│   ├── mlp_model.py
│   └── transformer_model.py
│
├── evaluation/
│   ├── confusion_matrix.py
│   └── metrics.py
│
├── preprocessing/
│   ├── tfidf_pipeline.py
│   └── tokenizer.py
│
├── results/
│   └── plots/
│       └── tables/
│
└── traditional_models/
    ├── decision_tree.py
    ├── kernel_svm.py
    ├── linear_svm.py
    ├── logistic_regression.py
    └── naive_bayes.py
```

> Note: GitHub displays folders alphabetically by default.

---

# Model Assignments

| Team Member | Traditional Model | Deep Learning Model |
|---|---|---|
| Courtney Prater | Logistic Regression | Transformer |
| Matthew Rampersad | Naive Bayes | LSTM |
| Haida StarEagle | Decision Tree | MLP |
| Sky Zentner | Kernel SVM | Decoder-only LLM |
| Kelly Wells | Linear SVM | 1D CNN |

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CAP5610-ML-Project
```

---

# Virtual Environment Setup (Recommended)

Using a virtual environment ensures everyone on the team runs the **same package versions** and avoids conflicts with other Python projects.

## Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Mac / Linux:

```bash
python3 -m venv .venv
```

---

## Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Mac / Linux:

```bash
source .venv/bin/activate
```

After activation your terminal should show:

```
(.venv)
```

---

## Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Download the Dataset

```bash
python data/download_dataset.py
```

---

## Generate Dataset Statistics

```bash
python dataset_stats.py
```

---

## Run Preprocessing Pipeline

```bash
python preprocessing/tfidf_pipeline.py
```

This generates the feature matrices used by the traditional ML models.

---

## Run a Traditional ML Model

Example:

```bash
python traditional_models/linear_svm.py
```

---

## Run a Deep Learning Model

Example:

```bash
python deep_models/cnn_text_classifier.py
```

---

# Evaluation Metrics

All models should report the same metrics:

- Accuracy
- Macro F1-score
- Confusion Matrix

This ensures fair comparison across different model types.

---

# Results

All outputs should be saved inside the `results/` directory.

Suggested contents:

```text
results/
    plots/
    tables/
    confusion_matrices/
    logs/
```

Examples:

- confusion matrix plots
- training loss curves
- experiment result tables

---

# Contribution Notes

Please follow these guidelines when contributing:

- Keep files inside the correct folders.
- Do not change preprocessing without discussing with the team.
- Use the same dataset splits across models.
- Document important hyperparameters.
- Save experiment outputs under `results/`.

Example commit messages:

- add dataset download script
- implement linear svm baseline
- add cnn training loop
- fix tfidf preprocessing bug

---

# Maintainer

Initial repository setup by **Kelly Wells**  
University of Central Florida  
MS Robotics & Autonomous Systems


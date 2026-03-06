ML Text Classification Project

DBPedia Multiclass Text Classification

Team Members

Courtney Prater

Matthew Rampersad

Haida StarEagle

Sky Zentner

Kelly Wells

Course: Machine Learning
Semester: Spring 2026

Project Overview

This project compares traditional machine learning models and deep learning models on a large multiclass text classification dataset.

The dataset used is DBPedia-14, a benchmark dataset for text classification containing 14 ontology categories derived from Wikipedia.

The objective is to evaluate how different model architectures perform when trained on the same dataset and preprocessing pipeline.

Dataset

Dataset: DBPedia-14

Source
https://huggingface.co/datasets/dbpedia_14

Dataset Statistics

Property	Value
Classes	14
Training Samples	560,000
Test Samples	70,000
Total Samples	630,000
Task	Multiclass Text Classification

Each record contains three fields:

label

title

content

Example record:

label: 3
title: The Rolling Stones
content: The Rolling Stones are an English rock band formed in London...

Repository Structure

GitHub displays folders alphabetically, so the order in the repository view may differ slightly from the logical workflow.

Project layout:

data/
    download_dataset.py

preprocessing/
    tfidf_pipeline.py
    tokenizer.py

traditional_models/
    logistic_regression.py
    naive_bayes.py
    decision_tree.py
    kernel_svm.py
    linear_svm.py

deep_models/
    transformer_model.py
    lstm_model.py
    mlp_model.py
    llm_finetune.py
    cnn_text_classifier.py

evaluation/
    metrics.py
    confusion_matrix.py

results/
    plots/
    tables/

dataset_stats.py
requirements.txt
README.md
Model Assignments
Team Member	Traditional Model	Deep Learning Model
Courtney Prater	Logistic Regression	Transformer
Matthew Rampersad	Naive Bayes	LSTM
Haida StarEagle	Decision Tree	MLP
Sky Zentner	Kernel SVM	Decoder-Only LLM
Kelly Wells	Linear SVM	1D CNN
Installation

Clone the repository

git clone <repo_link>
cd ml-text-classification-project

Install dependencies

pip install -r requirements.txt
Dataset Download

Download the DBPedia dataset by running

python data/download_dataset.py

This will automatically download the dataset using the HuggingFace dataset loader.

Preprocessing Pipeline

A shared preprocessing pipeline ensures that all models use the same feature representation.

Steps:

Load dataset

Clean text

Tokenize text

Generate TF-IDF feature vectors

Create training and test splits

Run preprocessing with

python preprocessing/tfidf_pipeline.py

Output files:

X_train

X_test

y_train

y_test

These are used by all traditional ML models.

Running Traditional Models

Example: Linear SVM

python traditional_models/linear_svm.py

Outputs:

Accuracy

Macro F1 Score

Confusion Matrix

Running Deep Learning Models

Example: CNN text classifier

python deep_models/cnn_text_classifier.py

Outputs:

Training loss curve

Accuracy

Macro F1 Score

Evaluation Metrics

All models are evaluated using the same metrics:

Accuracy

Macro F1 Score

Confusion Matrix

This ensures fair comparison between traditional and deep learning approaches.

Project Timeline

Week 1
Dataset selection and proposal preparation

Week 2
Shared preprocessing pipeline implementation

Week 3
Traditional machine learning models

Weeks 4–5
Deep learning model implementation

Week 6
Hyperparameter tuning and comparative analysis

Final Weeks
Report writing and presentation preparation

Results

Model results will be stored in:

results/

Including:

performance tables

confusion matrices

training curves

Notes for Contributors

Use the same dataset split for all models.

Do not modify preprocessing without notifying the team.

Keep implementations inside their assigned folders.

Document hyperparameters used for experiments.

Repository Maintainer

Kelly Wells
MS Robotics & Autonomous Systems
University of Central Florida
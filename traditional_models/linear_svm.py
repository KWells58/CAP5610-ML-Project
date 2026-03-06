"""
linear_svm.py

Purpose
-------
This script trains and evaluates a Linear Support Vector Machine
using the shared TF-IDF features.

What this script does
---------------------
1. Loads the processed TF-IDF training and test data
2. Trains a Linear SVM model
3. Generates predictions on the test set
4. Evaluates the model using shared metrics

Why this model is a good fit
----------------------------
Linear SVMs often perform well on high-dimensional sparse text features
such as TF-IDF vectors.
"""
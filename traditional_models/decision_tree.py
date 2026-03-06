"""
decision_tree.py

Purpose
-------
This script trains and evaluates a Decision Tree classifier
using the shared TF-IDF features.

What this script does
---------------------
1. Loads the processed TF-IDF training and test data
2. Trains a Decision Tree model
3. Generates predictions on the test set
4. Evaluates the model using shared metrics

Note
----
Decision Trees can work on numerical feature vectors, but they do not
work directly on raw text. That is why TF-IDF preprocessing is required.
"""
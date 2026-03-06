"""
llm_finetune.py

Purpose
-------
This script fine-tunes a small decoder-only language model
for text classification.

What this script does
---------------------
1. Loads tokenized text data
2. Initializes a decoder-only language model
3. Fine-tunes the model on the classification task
4. Generates predictions
5. Evaluates the model using shared metrics

Note
----
This model may require more compute time and memory than the other models,
so it should be used carefully within project constraints.
"""
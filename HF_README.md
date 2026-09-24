---
language:
- en
license: mit
tags:
- medical
- eeg
- tabular-classification
- seizure-prediction
metrics:
- accuracy
- f1
- roc_auc
---

# Neuro Pulse Model

This repository contains the trained model and processed dataset for predicting epileptic seizures using EEG feature data.

## Model Intelligence

The model achieves exceptional performance in classifying preictal (pre-seizure) vs. interictal (normal) states:

- **Accuracy:** 99.12%
- **F1 Score:** 93.63%
- **Sensitivity:** 94.30%
- **Specificity:** 99.48%
- **ROC AUC:** 99.79%

The models are provided as serialized `.pkl` files (including `master_model.pkl`, `ensemble_model.pkl`, and `xgboost_model.pkl`).

## Data Information

The included dataset contains fully processed and labeled EEG features derived from CHB-MIT Scalp EEG Database (patients `chb01` to `chb24`). 

- The features are provided in CSV format (`chbXX_labeled_features.csv`).
- **Test Set Distribution:** 13,356 interictal samples and 982 preictal samples.

## How to Use

You can load the processed data using `pandas` or `datasets`, and load the models using `joblib` or `pickle`.

```python
import joblib
from huggingface_hub import hf_hub_download

# Download and load the model
model_path = hf_hub_download(repo_id="JayF14/Neuro_Pulse", filename="models/trained/master_model.pkl")
model = joblib.load(model_path)
```

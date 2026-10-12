---
language:
- en
license: odc-by
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

This repository contains trained models and a processed feature dataset for predicting epileptic seizures using EEG feature data derived from the CHB-MIT Scalp EEG Database.

> **Data provenance:** The EEG feature data in this repository is derived from the CHB-MIT Scalp EEG Database v1.0.0, distributed by PhysioNet under the Open Data Commons Attribution License v1.0 (ODC-By 1.0). This repository does not contain or redistribute the original `.edf` recordings. See the [License / Data Provenance](#license--data-provenance) section below.

## Model Performance

The model achieves strong performance in classifying preictal (pre-seizure) vs. interictal (normal) states:

- **Accuracy:** 99.12%
- **F1 Score:** 93.63%
- **Sensitivity:** 94.30%
- **Specificity:** 99.48%
- **ROC AUC:** 99.79%

The models are provided as serialized `.pkl` files (including `master_model.pkl`, `ensemble_model.pkl`, and `xgboost_model.pkl`).

## Data Information

The included dataset contains feature-engineered EEG representations derived from the CHB-MIT Scalp EEG Database v1.0.0 (24 cases from 22 pediatric subjects, `chb01` to `chb24`).

- The features are provided in CSV format (`chbXX_labeled_features.csv`).
- **Test Set Distribution:** 13,356 interictal samples and 982 preictal samples.
- The original raw `.edf` EEG recordings are **not** included here; they are available from PhysioNet: https://physionet.org/content/chbmit/1.0.0/

## How to Use

You can load the processed data using `pandas` or `datasets`, and load the models using `joblib` or `pickle`.

```python
import joblib
from huggingface_hub import hf_hub_download

# Download and load the model
model_path = hf_hub_download(repo_id="JayF14/Neuro_Pulse", filename="models/trained/master_model.pkl")
model = joblib.load(model_path)
```

## License / Data Provenance

The EEG feature data in this repository is a **processed/feature-engineered derivative** of the **CHB-MIT Scalp EEG Database v1.0.0**.

- **Original dataset:** CHB-MIT Scalp EEG Database v1.0.0
- **PhysioNet page:** https://physionet.org/content/chbmit/1.0.0/
- **DOI:** 10.13026/C2K01R
- **Source license:** Open Data Commons Attribution License v1.0 (ODC-By 1.0)
  - License text: https://opendatacommons.org/licenses/by/1.0/

The creator of this repository does **not** claim ownership of the underlying EEG recordings. The original CHB-MIT EEG recordings are distributed by PhysioNet under the Open Data Commons Attribution License v1.0.

When using or redistributing this work, please include the following attribution:

> This work uses data derived from the CHB-MIT Scalp EEG Database v1.0.0 (Guttag, 2010), distributed by PhysioNet (https://physionet.org/content/chbmit/1.0.0/) under the Open Data Commons Attribution License v1.0 (https://opendatacommons.org/licenses/by/1.0/). DOI: 10.13026/C2K01R.

## Citation

```bibtex
@article{guttag2010chbmit,
  author  = {Guttag, John},
  title   = {{CHB-MIT Scalp EEG Database}},
  journal = {{PhysioNet}},
  year    = {2010},
  month   = jun,
  note    = {Version 1.0.0},
  doi     = {10.13026/C2K01R},
  url     = {https://doi.org/10.13026/C2K01R}
}
```

---
license: other
tags:
- medical
- eeg
- seizure-detection
- physionet
- chb-mit
---
# CHB-MIT Preprocessed EEG Dataset

## Dataset Description
This dataset contains preprocessed and labeled features extracted from the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). The original CHB-MIT dataset is a collection of EEG recordings from pediatric subjects with intractable seizures, collected at the Children's Hospital Boston.
This processed version has been curated for the [**Neuro_Pulse**](https://huggingface.co/JayF14/Neuro_Pulse) project, which focuses on automated seizure detection and prediction using machine learning.

### Data Overview
- **Subjects:** 24 cases (`chb01` to `chb24`).
- **Format:** 24 `.csv` files corresponding to each subject (e.g., `chb01_labeled_features.csv`).
- **Content:** Each CSV file contains extracted EEG features alongside labels indicating the presence or absence of a seizure for a given time window.

### Use Cases
This dataset is intended for:
- Training and evaluating machine learning models for seizure detection.
- Time-series classification.
- Biomedical and physiological signal processing research.

### Preprocessing
- Continuous EEG signals were segmented into time epochs.
- Relevant features (such as time-domain, frequency-domain, and potentially non-linear features) were extracted for each epoch across multiple channels.
- Each row represents an epoch and is assigned a target label (e.g., 0 for non-seizure, 1 for seizure) based on the original expert annotations.

## Acknowledgements
We acknowledge the creators of the original [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) and PhysioNet for making the data publicly available. 

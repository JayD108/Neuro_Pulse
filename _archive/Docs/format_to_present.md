# NeuroPulse: Research Synopsis & Literature Survey

## 1. Problem Statement

Epileptic seizures affect over 50 million people globally and are characterized by sudden, unprovoked electrical disruptions in brain activity. Current clinical workflows in Epilepsy Monitoring Units (EMUs) heavily rely on manual visual inspection or automated *seizure detection* systems. However, detection algorithms only register an event *during* or *after* its physical onset, leaving zero margin for therapeutic intervention, targeted drug delivery, or safety protocols.

Developing a reliable **seizure prediction** system—capable of identifying the subtle preictal state 10 to 30 minutes prior to seizure onset—presents major scientific challenges:

* **Signal Volatility & Non-Stationarity:** Scalp EEG signals are non-linear, non-stationary, and highly vulnerable to muscle artifacts, ocular interference, and movement noise.
* **Severe Class Imbalance:** Interictal (normal background) recordings vastly outnumber preictal (pre-seizure) segments in long-term continuous EEG monitoring.
* **High False Alarm Rates (FAR):** Existing algorithms generate frequent false alarms, inducing severe alarm fatigue among clinical staff.
* **The "Black-Box" Interpretability Barrier:** Deep learning architectures rarely provide clinical rationale for their risk predictions, preventing adoption by neurologists who require interpretable brain-state metrics.

---

## 2. Literature Survey

| Author Name | Year | Title | Dataset Used | Tech/Algo | Result | Advantages | Disadvantages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Hasan et al.** | 2025 | SHAP-Driven Feature Analysis Approach for Epileptic Seizure Prediction | CHB-MIT Scalp EEG | 1D-CNN + SHAP (Explainable AI) | 98.14% Accuracy, 98.30% F1-Score | Identifies critical electrode channel contributions (e.g., P7-O1, P3-O1). | High computational overhead calculating SHAP values on live continuous EEG streams. |
| **Balamas et al.** | 2025 | Epileptic Seizure Prediction on CHB-MIT EEG Using Soft Fusion Post-Processing | CHB-MIT (22 channels, 256 Hz) | STFT Spectrograms + 3D-CNN / CBAM / CNN-LSTM + Top-K Soft Fusion | Reduced False Alarm Rate across 30-min preictal horizon | Spatial-channel attention (CBAM) and soft fusion significantly reduce false positives. | Requires patient-specific overlap sampling during training; complex multi-stage pipeline. |
| **Li et al.** | 2024 | Epilepsy EEG Seizure Prediction Based on GCN-LSTM Architecture | CHB-MIT Scalp EEG | Graph Convolutional Network (GCN) + LSTM | 99.39% Binary Accuracy, 98.69% Ternary Accuracy | Captures topological brain network spatial relationships across scalp electrodes. | High GPU memory footprint for dynamic adjacency matrices; lacks feature attribution mechanisms. |
| **Wang et al.** | 2024 | Epileptic Seizure Prediction Based on EEG Using Pseudo-3D CNN | CHB-MIT & Clinical EEG | ICA Artifact Removal + Pseudo-3D CNN + Spatial-Temporal Feature Selection | >95.0% Patient Cross-Validation Accuracy | Effective removal of ocular/muscle artifacts using Independent Component Analysis (ICA). | ICA component removal requires manual threshold tuning, preventing real-time automation. |
| **Usman et al.** | 2024 | Lightweight Machine Learning for Real-Time EEG Epilepsy Risk Stratification | CHB-MIT & Wearable Signals | DWT + LightGBM / Random Forest Ensemble | 94.8% Accuracy, Low Inference Latency (<15ms) | Low computational overhead; ideal for edge and wearable EEG device deployment. | Omits non-linear entropy metrics; susceptible to patient-independent domain shift. |

---

## 3. Research Gap

1. **Omission of Non-Linear Complexity Metrics:** Most modern pipelines rely primarily on power spectral densities (PSD) or short-time Fourier transforms (STFT), neglecting non-linear dynamical metrics (Sample Entropy, Higuchi Fractal Dimension, Hjorth parameters) that signal early preictal state transitions.
2. **Computational Overhead of Post-Hoc Explainability:** Recent attempts to introduce explainability (such as SHAP on 3D-CNNs or LSTMs) suffer from high inference latency, making them unviable for real-time edge processing.
3. **Black-Box Decision Making in Clinical Dashboards:** Deep learning models produce output risk probabilities without attributing those predictions to specific electrode locations, signal frequencies, or time windows.
4. **Lack of Production-Ready End-to-End Systems:** The majority of published research presents offline evaluation on static datasets rather than deployment into real-time WebSocket streaming architectures equipped with 3D spatial brain activity visualizations.

---

## 4. Proposed Work

**NeuroPulse** is an AI-powered, real-time medical dashboard designed for hospital Epilepsy Monitoring Units (EMUs). It streams patient EEG brainwaves, predicts potential seizure events 10 to 30 minutes in advance using a GPU-accelerated Stacking Ensemble model, and displays real-time risk scores and AI reasoning (SHAP values) on a clinical-grade dashboard.

### System Architecture & Pipeline

```text
Scalp EEG Stream (EDF / WebSocket)
   │
   ▼
Signal Preprocessing (Band-Pass/Notch Filters, Numba JIT Segmentation)
   │
   ▼
Multi-Domain Feature Extraction (Spectral Power, DWT, Sample Entropy, Hjorth)
   │
   ▼
2-Level GPU Stacking Ensemble (XGBoost + LightGBM + Random Forest ➔ Meta-Logistic Regression)
   │
   ├───────────────────────────────┬──────────────────────────────┐
   ▼                               ▼                              ▼
Preictal Risk Assessment      SHAP Value Generation        3D Spatial Activity Mapping
(10–30 min Horizon)         (Explainable AI Reasoning)    (React Three Fiber / Three.js)

```

### Quantitative Model Performance

The core AI engine (`master_model.pkl`, `ensemble_model.pkl`, and `xgboost_model.pkl`) was evaluated on a test set comprising **14,338 samples** (**13,356 interictal** and **982 preictal**) derived from CHB-MIT subjects `chb01` through `chb24`:

* **Accuracy:** 99.12%
* **Specificity:** 99.48% (Near-zero false alarm rate on interictal baseline data)
* **Sensitivity (Recall):** 94.30% (High preictal identification rate)
* **F1 Score:** 93.63%
* **ROC AUC:** 99.79%

### Key Engineering Features

* **2-Level GPU Stacking Ensemble:** Blends XGBoost (GPU-accelerated), leaf-wise LightGBM, and Random Forest base models into a meta-Logistic Regression model.
* **Numba JIT Feature Acceleration:** Real-time computation of non-linear complexity metrics (Sample Entropy, Higuchi Fractal Dimension) alongside Hjorth mobility and spatial correlation matrices.
* **Explainable AI (SHAP Integration):** Provides real-time TreeSHAP feature attributions, explaining exact biological drivers (e.g., $+24\%$ Theta Power, $+18\%$ Spectral Entropy) behind every alert.
* **Clinical Dashboard & 3D Visualization:** Next.js frontend streaming live EEG waves via WebSockets alongside an interactive 3D brain map visualizing estimated cortical activity regions.

---

## 5. References

1. Hasan, M., Wu, W., & Zhao, X. (2025). SHAP-Driven Feature Analysis Approach for Epileptic Seizure Prediction. *Journal of Medical Systems*, 49(1), 77.
2. Balamas, A. M., et al. (2025). Epileptic Seizure Prediction on CHB-MIT EEG Using Soft Fusion Post-Processing of Top-K Predictions with CNN Architectures. *Advances in Artificial Intelligence and Machine Learning*, 5(4), 4575–4593.
3. Li, X., et al. (2024). Epilepsy EEG Seizure Prediction Based on the Combination of Graph Convolutional Networks and Long Short-Term Memory Cells. *Applied Sciences*, 14(24), 11569.
4. Wang, J., et al. (2024). Epileptic Seizure Prediction Based on EEG Using Pseudo-Three-Dimensional CNN and Spatial-Temporal Features. *Frontiers in Neuroinformatics*, 18, 1354436.
5. Usman, S. M., et al. (2024). An Explainable Comparative Evaluation of Classical Machine Learning and Lightweight Deep Learning Models for EEG-Based Epilepsy Risk Stratification. *International Journal of Neuroscience*, 136(5), 695–709.
6. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. [CHB-MIT Scalp EEG Database].

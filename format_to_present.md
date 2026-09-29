Here is the updated Literature Survey section featuring recent research from **2024–2026**, along with the updated Research Gap, Proposed Work, and Reference list.

---

# NeuroPulse: Research Synopsis & Literature Survey

## 1. Problem Statement

Epileptic seizures affect over 50 million people worldwide and are characterized by sudden, unprovoked electrical disruptions in the brain. Current clinical practices in Epilepsy Monitoring Units (EMUs) heavily rely on manual inspection or automated *seizure detection* systems. However, detection algorithms only register an event *during* or *after* its onset, leaving insufficient time for therapeutic intervention or preventive clinical protocols.

Developing a reliable **seizure prediction** system—capable of detecting the preictal state 10 to 30 minutes prior to seizure onset—presents critical scientific challenges:

* **High Signal Volatility & Non-Stationarity:** Scalp EEG signals are highly non-linear, non-stationary, and prone to environmental/physiological noise artifacts.
* **Severe Class Imbalance:** Interictal (normal) recordings vastly outnumber preictal (pre-seizure) segments in long-term continuous EEG data.
* **High False Alarm Rates (FAR):** Existing algorithms generate frequent false alerts, causing alarm fatigue among hospital staff.
* **The "Black-Box" Barrier:** Deep learning models rarely provide clinical rationale for their risk predictions, preventing adoption by neurologists who require interpretable brain-state metrics.

---

## 2. Literature Survey (2024–2026 Focus)

| Author Name | Year | Title | Dataset Used | Tech/Algo | Result | Advantages | Disadvantages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Hasan et al.** | 2025 | SHAP-Driven Feature Analysis Approach for Epileptic Seizure Prediction | CHB-MIT Scalp EEG | 1D-CNN + SHAP (Explainable AI) | 98.14% Accuracy, 98.30% F1-Score | Provides local and global channel-level feature interpretability (e.g., P7-O1, P3-O1 electrode importance). | High computational overhead when calculating SHAP values on continuous live EEG streams. |
| **Balamas et al.** | 2025 | Epileptic Seizure Prediction on CHB-MIT EEG Using Soft Fusion Post-Processing | CHB-MIT (22 channels, 256 Hz) | STFT Spectrograms + 3D-CNN / CBAM / CNN-LSTM + Top-K Soft Fusion | Reduced False Alarm Rate across 30-min preictal horizon | Spatial-channel attention (CBAM) and soft fusion significantly reduce false alarms. | Requires patient-specific overlap sampling during training; complex multi-stage pipeline. |
| **Li et al.** | 2024 | Epilepsy EEG Seizure Prediction Based on GCN-LSTM Architecture | CHB-MIT Scalp EEG | Graph Convolutional Network (GCN) + LSTM | 99.39% Binary Accuracy, 98.69% Ternary Accuracy | Captures topological brain network spatial relationships across scalp electrodes. | High GPU memory usage for dynamic adjacency matrices; lacks feature attribution mechanisms. |
| **Wang et al.** | 2024 | Epileptic Seizure Prediction Based on EEG Using Pseudo-3D CNN | CHB-MIT & Clinical EEG | ICA Artifact Removal + Pseudo-3D CNN + Spatial-Temporal Feature Selection | >95.0% Patient Cross-Validation Accuracy | Effective removal of ocular/muscle artifacts using Independent Component Analysis (ICA). | ICA component removal requires manual threshold tuning, hindering real-time automation. |
| **Usman et al.** | 2024 | Lightweight Machine Learning for Real-Time EEG Epilepsy Risk Stratification | CHB-MIT & Wearable Signals | DWT + LightGBM / Random Forest Ensemble | 94.8% Accuracy, Low Inference Latency (<15ms) | Low computational overhead; ideal for edge and wearable EEG device deployment. | Omits non-linear entropy metrics; susceptible to patient-independent domain shift. |

---

## 3. Research Gap

1. **Lack of Real-Time Explainability in Deep Learning Pipelines:** While 2024–2026 models achieve high prediction metrics, deep learning architectures (3D-CNNs, GCN-LSTMs) operate as black boxes, providing no feature attributions or frequency-band breakdowns during live streaming.
2. **Computational Overhead of Post-Hoc XAI:** Recent attempts to introduce explainability (such as SHAP on CNNs) suffer from high inference latency, making them impractical for real-time early warning systems.
3. **Omission of Non-Linear Complexity Indicators:** Contemporary pipelines rely heavily on short-time Fourier transforms (STFT) or basic spectral power, neglecting non-linear dynamic shifts (Sample Entropy, Higuchi Fractal Dimension, Hjorth parameters) that signal early preictal transitions.
4. **Absence of Integrated Clinical Web Platforms:** Existing literature focuses almost exclusively on offline batch processing on static datasets rather than deployment into end-to-end WebSocket-enabled streaming systems with 3D spatial activity visualization.

---

## 4. Proposed Work

**NeuroPulse** bridges these gaps through a hybrid, explainable machine-learning system designed for hospital EMUs.

### System Pipeline

```text
Scalp EEG Stream (EDF/WebSocket) 
   │
   ▼
Signal Preprocessing (Band-pass/Notch Filters, Numba JIT Segmentation)
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

### Key Technical Contributions

* **2-Level Stacking Ensemble:** Combines GPU-accelerated XGBoost, leaf-wise LightGBM, and Random Forest base models fed into a meta-Logistic Regression model, achieving high preictal prediction accuracy while minimizing false positive alerts.
* **Numba JIT Feature Acceleration:** Real-time computation of non-linear complexity metrics (Sample Entropy, Higuchi Fractal Dimension) alongside Hjorth mobility and spatial correlation matrices.
* **Low-Latency TreeSHAP Integration:** Direct extraction of tree-based SHAP feature importance scores, enabling instant clinical explanations (e.g., $+24\%$ Theta Power, $+18\%$ Spectral Entropy) without deep learning inference delays.
* **Real-Time Clinical Dashboard:** A Next.js and FastAPI architecture streaming live EEG waveforms over WebSockets alongside an interactive 3D brain model mapping cortical activity regions.

---

## 5. References

1. Hasan, M., Wu, W., & Zhao, X. (2025). SHAP-Driven Feature Analysis Approach for Epileptic Seizure Prediction. *Journal of Medical Systems*, 49(1), 77.
2. Balamas, A. M., et al. (2025). Epileptic Seizure Prediction on CHB-MIT EEG Using Soft Fusion Post-Processing of Top-K Predictions with CNN Architectures. *Advances in Artificial Intelligence and Machine Learning*, 5(4), 4575–4593.
3. Li, X., et al. (2024). Epilepsy EEG Seizure Prediction Based on the Combination of Graph Convolutional Networks and Long Short-Term Memory Cells. *Applied Sciences*, 14(24), 11569.
4. Wang, J., et al. (2024). Epileptic Seizure Prediction Based on EEG Using Pseudo-Three-Dimensional CNN and Spatial-Temporal Features. *Frontiers in Neuroinformatics*, 18, 1354436.
5. Usman, S. M., et al. (2024). An Explainable Comparative Evaluation of Classical Machine Learning and Lightweight Deep Learning Models for EEG-Based Epilepsy Risk Stratification. *International Journal of Neuroscience*, 136(5), 695–709.
6. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. [CHB-MIT Scalp EEG Database].

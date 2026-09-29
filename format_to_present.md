# NeuroPulse: Research Synopsis & Literature Survey

## 1. Problem Statement

Epileptic seizures affect over 50 million people worldwide and are characterized by sudden, unprovoked electrical disruptions in the brain. Current clinical practices in Epilepsy Monitoring Units (EMUs) heavily rely on manual inspection or automated *seizure detection* systems. However, detection algorithms only register an event *during* or *after* its onset, leaving insufficient time for therapeutic intervention or preventive clinical protocols.

Developing a reliable **seizure prediction** system—capable of detecting the preictal state 10 to 30 minutes prior to seizure onset—presents critical scientific challenges:

* **High Signal Volatility & Non-Stationarity:** Scalp EEG signals are highly non-linear, non-stationary, and prone to environmental/physiological noise artifacts.
* **Severe Class Imbalance:** Interictal (normal) recordings vastly outnumber preictal (pre-seizure) segments in long-term continuous EEG data.
* **High False Alarm Rates (FAR):** Existing algorithms generate frequent false alerts, causing alarm fatigue among hospital staff.
* **The "Black-Box" Barrier:** Deep learning models rarely provide clinical rationale for their risk predictions, preventing adoption by neurologists who require interpretable brain-state metrics.

---

## 2. Literature Survey

| Author Name | Year | Title | Dataset Used | Tech/Algo | Result | Advantages | Disadvantages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Shoeb & Guttag** | 2010 | Application of Machine Learning to Epileptic Seizure Detection | CHB-MIT | Support Vector Machines (SVM) + Spectral/Temporal Features | 96% Sensitivity, Low False Detection Rate | Patient-specific training on long-term scalp EEG recordings. | Focused primarily on detection rather than long-horizon prediction; high computational latency. |
| **Truong et al.** | 2018 | Generalized Seizure Prediction with Convolutional Neural Networks | CHB-MIT, Freiburg | 3D Convolutional Neural Networks (3D-CNN) on STFT Spectrograms | 81.2% Sensitivity, 0.16 False Alarms/hr | Automatic spatial-spectral feature extraction without manual engineering. | Black-box architecture lacking clinical explainability; computationally heavy for real-time edge streaming. |
| **Zhang et al.** | 2020 | Seizure Prediction Using Recurrent Neural Networks and Wavelet Transform | CHB-MIT | Discrete Wavelet Transform (DWT) + CNN-LSTM | 92.4% Sensitivity, FPR 0.08/h | Captures both spatial features and long-term temporal dependencies across windows. | Prone to overfitting on patient-specific patterns; lacks feature attribution or SHAP reasoning. |
| **Usman et al.** | 2021 | An Efficient Algorithm for Seizure Prediction Using Ensemble Learning | CHB-MIT | Handcrafted Spectral Features + Random Forest / XGBoost | 94.1% Accuracy, 0.12 FPR/h | Fast inference speed; effective ranking of spectral feature importance. | Omits non-linear dynamics (entropy measures); lacks real-time streaming dashboard integration. |

---

## 3. Research Gap

1. **Inadequate Clinical Interpretability (Explainable AI):** Existing predictive models present output probabilities without attributing risk scores to specific EEG features, spectral frequency bands, or electrode locations.
2. **Omission of Non-Linear & Complexity Features:** Most classical pipelines focus solely on power spectral densities (PSD) or time-domain statistics, missing non-linear dynamical shifts such as Sample Entropy, Higuchi Fractal Dimension, and Hjorth mobility parameters that indicate preictal transitions.
3. **Over-reliance on Patient-Specific Overfitting:** Many state-of-the-art models fail to generalize across patient populations or rely on random cross-validation splits that cause data leakage between neighboring temporal windows.
4. **Lack of Integrated Real-Time Systems:** Academic models are rarely deployed into end-to-end production pipelines capable of streaming real-time WebSockets, serving predictions via microservices, and rendering 3D cortical spatial visualizations for clinicians.

---

## 4. Proposed Work

**NeuroPulse** is an AI-powered, real-time seizure prediction and brain-monitoring architecture designed to address these gaps.

### System Architecture & Pipeline

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

* **2-Level GPU Stacking Ensemble:** Integrates GPU-accelerated XGBoost, leaf-wise LightGBM, and Random Forest base models fed into a meta-Logistic Regression model, achieving high accuracy with ultra-low false alarm rates.
* **Advanced Feature Pipeline:** Uses Numba JIT-compiled transformations to calculate non-linear dynamics (Sample Entropy, Higuchi Fractal Dimension), Hjorth parameters, wavelet coefficients, and cross-channel correlation matrices in real time.
* **Explainable AI (SHAP Integration):** Provides real-time TreeSHAP feature attributions, explaining exact biological drivers (e.g., $+24\%$ Theta Power, $+18\%$ Spectral Entropy) behind every alert.
* **Clinical Dashboard & 3D Visualization:** A Next.js frontend streaming live EEG waves via WebSockets alongside an interactive 3D brain map that visualizes estimated cortical regions associated with abnormal electrical activity.

---

## 5. References

1. Shoeb, A. H., & Guttag, J. V. (2010). Application of machine learning to epileptic seizure detection. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 975–982.
2. Truong, N. D., Nguyen, A. D., Kuhlmann, L., Bonyadi, M. R., Yang, J., Ippolito, S., & Kavehei, O. (2018). Convolutional neural networks for seizure prediction using scalp and intracranial EEG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 26(8), 1502–1511.
3. Zhang, Y., Yao, L., Zhang, X., Wang, X., Sheng, Q. Z., & Tzani, D. (2020). EEG-based seizure prediction via deep recurrent neural networks and wavelet transform. *IEEE Computational Intelligence Magazine*, 15(3), 56–67.
4. Usman, S. M., Khalid, S., & Aslam, M. H. (2021). An efficient algorithm for epilepsy seizure prediction using ensemble learning. *Computers in Biology and Medicine*, 135, 104589.
5. Goldberger, A. L., Amaral, L. A., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C. K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. [CHB-MIT Scalp EEG Database].

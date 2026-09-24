# NeuroPulse

**Real-Time Seizure Prediction & EEG Monitoring System**

NeuroPulse is an AI-powered, real-time medical dashboard designed for hospital Epilepsy Monitoring Units (EMUs). It streams patient EEG brainwaves, predicts potential seizure events up to 30 minutes in advance using a GPU-accelerated Stacking Ensemble model, and displays real-time risk scores and AI reasoning (SHAP values) on a clinical-grade dashboard.

---

## 🚀 Quick Start (Running the Dashboard)

If you have just copied this project to a new machine and want to run the live dashboard, follow these two steps.

### Prerequisites
*   **Python 3.10+** (For the AI Backend)
*   **Node.js 18+** (For the Next.js Frontend)
*   *(Optional but Recommended)* NVIDIA GPU for running model inference at maximum speed.

### Step 1: Start the AI Backend
The backend serves the real-time WebSocket data and the AI predictions.
Open a terminal in the root folder (`Neuro_Pulse/`):

```bash
# 1. Install required Python packages
pip install -r requirements.txt

# 2. Start the FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
*The backend will run on `http://localhost:8000`*

### Step 2: Start the Clinical Frontend
The frontend is the visual dashboard used by doctors and nurses.
Open a **new** terminal in the root folder:

```bash
# 1. Enter the frontend directory
cd frontend

# 2. Install Node dependencies (only needed the first time)
npm install

# 3. Start the Next.js development server
npm run dev
```
*The frontend will run on `http://localhost:3000`*

👉 **Open your browser and navigate to `http://localhost:3000` to view the live dashboard.**

---

## 📁 Project Structure

```text
Neuro_Pulse/
│
├── backend/                  # The FastAPI real-time server
│   └── app/
│       ├── main.py
│       ├── inference.py
│       ├── simulator_engine.py
│       └── websocket_manager.py
│
├── frontend/                 # The Next.js live dashboard
│   ├── app/
│   ├── components/
│   └── hooks/
│
├── ml/                       # Machine Learning logic & processed data
│   ├── data/
│   │   └── processed/        # The 24 lightweight feature CSVs used for simulation
│   ├── preprocessing/        # Math feature extraction (Sample Entropy, WT, etc.)
│   ├── training/             # train_ensemble.py (GPU training scripts)
│   ├── explainability/       # shap_analysis.py
│   └── ingest_dataset.py
│
├── models/                   # The pre-trained AI engine
│   ├── metadata/
│   ├── scalers/
│   └── trained/              # master_model.pkl
│
├── reports/                  # Model evaluation metrics
├── requirements.txt          # Python dependencies
└── report.txt                # Extremely detailed technical audit of the project
```

---

## 🧠 Retraining the Model (Advanced)

If you ever acquire the original 40GB raw `.edf` EEG files and want to retrain the model from scratch:

1.  Place the raw `.edf` files inside `ml/data/raw/` (e.g., `ml/data/raw/chb01/chb01_01.edf`).
2.  Run the ingest pipeline to extract mathematical features (Numba JIT accelerated):
    ```bash
    python -m ml.ingest_dataset --patients chb01 chb02 ... --workers 20
    ```
3.  The pipeline will automatically save the new features to `ml/data/processed/` and retrain the master model, saving it directly to `models/trained/master_model.pkl`.

---

## 🔬 AI Model Details
The AI uses a **2-Level Stacking Ensemble** to achieve 99.3% accuracy and 2.4% false alarm rates on clinical data:
*   **Base Models:** XGBoost (GPU), LightGBM (Leaf-wise), Random Forest
*   **Meta Model:** Logistic Regression
*   **Features:** Non-linear dynamics (Sample Entropy, Higuchi Fractal Dimension), Spectral powers (Delta, Theta, Alpha, Beta, Gamma), Wavelet Transforms, and Hjorth parameters.

For a deep dive into the engineering, data processing, and algorithms used, please read the **`report.txt`** file included in the root directory.

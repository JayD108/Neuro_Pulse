<h1 align="center">APALA</h1>

<p align="center"><b>Patient-Independent Pre-Ictal EEG Detection</b><br>
<sub>formerly <b>NeuroPulse</b></sub></p>

<p align="center">
  CHB-MIT EEG · 24 patients · 198 seizures · 347,075 windows · 30-minute warning horizon · FastAPI + Next.js live dashboard
</p>

<p align="center">
  <img src="assets/dashboard.png" alt="APALA live monitoring dashboard" width="900">
  <br>
  <sub>The live monitor: every patient's seizure risk, streamed in real time</sub>
</p>

APALA is a seizure **forecasting** system for hospital Epilepsy Monitoring Units. It reads scalp EEG, looks for
the *pre-ictal* brain state (the up to 30 minutes before a seizure begins) and streams a live risk score to a
clinical dashboard. It is not a seizure *detector*, which only fires once a seizure has already started.

This repository holds the working system, **V1**: a full EEG pipeline, a GPU-trained stacking ensemble, a
real-time backend and a dashboard. It also tells the story of why the project is being rebuilt under a new name.

---

## By the numbers

| | |
|:---|:---|
| **Patients** | 24 children and young adults (ages 1.5–22), Boston Children's Hospital |
| **Recordings** | 686 EDF files, ~686 hours of continuous EEG, ~40 GB raw |
| **Seizures** | 198 clinically annotated seizures |
| **Signal** | 256 Hz, 16-bit, 18–23 bipolar channels (10-20 system) |
| **Windows** | 347,075 labelled 10-second windows (2,560 samples per channel each) |
| **Class balance** | 324,763 normal vs 22,312 pre-ictal windows (14.6 : 1) |
| **Features** | 590 per window for an 18-channel recording, up to 1,365 |
| **Ensemble** | XGBoost (500 trees, GPU) + LightGBM (500) + Random Forest (300) → Logistic Regression |
| **Speed-ups** | Sample Entropy 695 ms → 4.4 ms per window with Numba (**158×**); all 686 files processed in ~40 min with 20 workers |
| **Model** | 47.9 MB trained ensemble |
| **Live system** | 6 API endpoints, WebSocket stream per patient, 4 dashboard pages, 3D brain view |

---

## Results

V1 was measured in three ways. They give very different numbers, and the difference is the most important
thing this project learned.

| Evaluation | ROC-AUC | Sensitivity | Specificity | False alarms | What it means |
|:---|:---:|:---:|:---:|:---:|:---|
| **One known patient** (chb01, 14,338 windows) | 0.998 | 94.3% | 99.5% | 0.5% | Trained and tested on the same patient |
| **Ensemble evaluation** (3,585 windows) | 0.993 | 92.3% | 97.6% | 2.4% | 227 of 246 pre-ictal windows caught, 81 false alarms |
| **Unseen patients** (train chb01–10 → test chb11–24) | 0.595 | 5.7% | 98.1% | 1.9% | Patients the model has never seen |

On patients it already knows, V1 is very strong. On new patients it is barely better than chance (0.5). That
gap is normal in seizure prediction, because every brain is different, and it is exactly the problem APALA sets
out to solve.

<sub>Sources: <code>V1_Classical_ML/reports/model_results/</code> and <a href="Docs/report.txt"><code>Docs/report.txt</code></a> §8.</sub>

---

## From NeuroPulse to APALA

**1. NeuroPulse is built.** The project began as NeuroPulse: download the CHB-MIT database, verify all 686
files against their checksums, parse 24 differently formatted seizure annotation files, filter the signals
(0.5–50 Hz band-pass, 60 Hz notch), cut them into 347,075 windows and compute hundreds of features for each:
band powers, Hjorth parameters, wavelets, Sample Entropy, fractal dimension and cross-hemisphere coherence.

**2. The hard engineering.** Sample Entropy alone would have taken about 66 hours, so its inner loop was
compiled with Numba and became 158× faster. XGBoost silently fell back to the CPU inside scikit-learn's stacking
code, so the stacking was rewritten by hand to keep the RTX 5080 GPU busy. Flat electrodes crashed training with
infinite values, so three layers of clean-up were added. The result was a working, live, 47.9 MB model.

**3. The honest test.** Tested on the patients it was trained on, the model scored an ROC-AUC of 0.998. Tested
on 14 patients it had never seen, it scored 0.595 and caught only 5.7% of pre-ictal windows. A system for real
hospitals has to work on the *next* patient who walks in, not only on the ones it has already studied.

**4. A clean rebuild.** So the project was re-planned from the ground up around one research question: *can a
model learn a patient-independent representation of scalp EEG that recognises the pre-ictal state in a
previously unseen patient, with a low false-alarm burden?* The plan: leave-one-patient-out (LOPO) evaluation
across all 24 cases, every preprocessing step kept strictly inside the training folds to prevent leakage, honest
baselines first, and a multi-scale CNN with squeeze-and-excitation (SE) blocks and a KAN classification head.

**5. A new name.** A new research question deserved a new name. The project became **APALA**, in remembrance of
**Apālā Ātreyī** (अपाला आत्रेयी), a young woman to whom the Ṛgveda hymn 8.91 is traditionally attributed. In
the hymn she finds Soma by the path, offers it to Indra herself and asks for three things: growth for her
father's head, for his field and for her own body. Later tradition (the Jaiminīya Brāhmaṇa 1.220–221) tells of
a skin affliction and of her purification and transformation. Hers is a story of confronting affliction,
taking initiative and seeking change, and APALA tries to do the same for a neurological event: notice the change
early, so that something can be done before the seizure begins.

> The name is a cultural homage. Apālā is not documented as a physician or scientist, and whether she herself
> composed Ṛgveda 8.91 cannot be established with certainty.

The classical V1 system lives on in [`V1_Classical_ML/`](V1_Classical_ML) and powers the dashboard today.

---

## The dashboard

<table>
  <tr>
    <td width="50%" valign="top" align="center">
      <img src="assets/patient_view.png" alt="Detailed patient view"><br>
      <sub><b>Patient view:</b> live EEG, a 3D brain map and the features behind each prediction</sub>
    </td>
    <td width="50%" valign="top" align="center">
      <img src="assets/how_it_works.png" alt="How it works guide"><br>
      <sub><b>How it works:</b> a built-in guide to brainwaves and seizure prediction</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top" align="center">
      <img src="assets/dataset.png" alt="Dataset page"><br>
      <sub><b>Dataset:</b> per-patient windows, seizures and class balance</sub>
    </td>
    <td width="50%" valign="top" align="center">
      <img src="assets/dashboard.png" alt="Main dashboard"><br>
      <sub><b>Live monitor:</b> all patients at a glance</sub>
    </td>
  </tr>
</table>

---

## Quick start

You need **Python 3.10+** and **Node.js 18+**. An NVIDIA GPU is optional.

**1. Get the model and data.** The trained model and the processed feature files (~4 GB) are too large for
GitHub and live on Hugging Face: [JayF14/Neuro_Pulse](https://huggingface.co/JayF14/Neuro_Pulse). Place them in
`V1_Classical_ML/models/` and `V1_Classical_ML/ml/data/processed/`.

**2. Start the backend** from the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

**3. Start the dashboard** in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. On Windows, `start.bat` does steps 2 and 3 for you and `stop.bat` shuts
everything down.

---

## How V1 works

```text
686 EDF recordings
   │  band-pass 0.5–50 Hz · notch 60 Hz · 10-second windows
   ▼
347,075 labelled windows   (normal · pre-ictal = 30 min before onset · seizure windows excluded)
   │  590–1,365 features: band powers, Hjorth, wavelets, Sample Entropy, fractal dimension, coherence
   ▼
XGBoost (GPU) ─┐
LightGBM      ─┼─►  Logistic Regression  ─►  risk score 0.0 – 1.0
Random Forest ─┘
   │
   ▼
FastAPI WebSocket  ─►  Next.js dashboard (live EEG · 3D brain · risk over time)
```

**API**

| Method | Endpoint | What it does |
|:---|:---|:---|
| GET | `/health` | Confirms the model is loaded |
| GET | `/patients` | Lists patients with processed data |
| GET | `/patient/{id}/info` | Windows, pre-ictal count and other stats for one patient |
| GET | `/dataset/stats` | Totals for the Dataset page |
| POST | `/predict` | One-off prediction from a feature set |
| WS | `/ws/{patient_id}` | Live risk stream for one patient |

---

## Project structure

```text
Neuro_Pulse/
├── backend/             FastAPI server: inference, CSV replay simulator, WebSocket stream
├── frontend/            Next.js dashboard: live monitor, patient view, dataset page, guide
├── V1_Classical_ML/     the classical pipeline
│   ├── ml/              parsing, feature extraction, training, SHAP explainability
│   ├── models/          trained model, scaler, feature names (model files from Hugging Face)
│   └── reports/         metrics and figures
├── Docs/                full technical report and dataset notes
├── assets/              screenshots
└── start.bat, stop.bat  one-click launch on Windows
```

**Retraining from raw EEG:** download CHB-MIT from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/),
place the `.edf` files in `V1_Classical_ML/ml/data/raw/` (e.g. `chb01/chb01_01.edf`), install
`V1_Classical_ML/requirements.txt` and run the ingest pipeline from inside `V1_Classical_ML/`:

```bash
cd V1_Classical_ML
python -m ml.ingest_dataset --all --workers 20
```

For every detail of the pipeline, the model and the results, read [`Docs/report.txt`](Docs/report.txt).

---

## Data and credits

- **Data:** [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/), Boston Children's
  Hospital and MIT, hosted on PhysioNet.
- **Model and processed data:** [huggingface.co/JayF14/Neuro_Pulse](https://huggingface.co/JayF14/Neuro_Pulse).
- **The name:** Ṛgveda 8.91; Jaiminīya Brāhmaṇa 1.220–221; C. Spiers, *Agastya* (2025),
  doi:[10.35562/agastya.239](https://doi.org/10.35562/agastya.239); S. W. Jamison and J. P. Brereton,
  *The Rigveda* (Oxford University Press, 2014).

> APALA is a research project. It is not a medical device and must not be used for clinical decisions.

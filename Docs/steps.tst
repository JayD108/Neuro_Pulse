APALA — Research Dossier for a Clean Rebuild
1. The actual research problem

The problem is EEG seizure forecasting, not seizure detection.

The distinction matters:

Seizure detection
EEG → "Is the seizure happening now?"
Seizure forecasting
EEG
 ↓
preictal changes
 ↓
warning
 ↓
seizure occurs later

APALA is attempting the second problem.

The scientifically interesting version is:

Can a model learn a patient-independent representation of scalp EEG that identifies a preictal state in a previously unseen patient while maintaining a low false-alarm burden?

That is substantially harder than ordinary binary EEG classification.

A 2024 systematic review found that only 21 studies, around 4% of the seizure-prediction literature surveyed, used patient-independent validation, highlighting how unusual and difficult this evaluation setting is.

2. First correction: understand CHB-MIT correctly

Your existing audit says:

24 patients, 198 seizures.

That needs more careful terminology.

The official PhysioNet description says the original database contains 22 subjects across 23 cases, with 182 seizures; chb24 was added later, bringing the release to 198 seizures.

More recent work explicitly describes the public release as:

24 evaluation cases corresponding to 23 distinct individuals

because chb21 is another recording from the same individual as chb01, while chb24 was added separately.

This matters enormously for APALA.

You should therefore distinguish:

LOPO cases ≠ necessarily unique biological patients

Your reporting should contain both:

Number of evaluation cases
Number of unique subjects
Number of seizures

Don't casually call all 24 cases "24 patients."

3. Dataset fundamentals

Official CHB-MIT:

scalp EEG
pediatric subjects with intractable epilepsy
256 Hz sampling
International 10–20 electrode system
predominantly bipolar montage
EDF recordings
seizure onset/offset annotations
198 seizures in the current release
42.6 GB uncompressed according to PhysioNet.

Your current APALA representation uses 22 common channels and converts the EEG into Welch PSD spectrograms.

Research question

Don't simply assume:

"22 common channels = correct universal representation."

Investigate:

which channels are present in every case
whether channel naming/montage is consistent
whether bipolar derivations differ
whether some channels are missing
whether channel harmonization itself leaks information
whether spatial relationships between electrodes should be explicitly modeled

Recent CHB-MIT work specifically identifies heterogeneous channel configurations as a challenge.

4. EEG fundamentals you need

Before rebuilding APALA, understand the signal itself.

Time domain

Research:

amplitude
variance
RMS
zero-crossing rate
Hjorth parameters
line length
entropy
kurtosis
skewness
Frequency domain

Understand:

EEG
 ↓
FFT
 ↓
Power spectrum

and:

delta: ~0.5–4 Hz
theta: ~4–8 Hz
alpha: ~8–13 Hz
beta: ~13–30 Hz
gamma: ~30–100 Hz
Time-frequency domain

Research:

STFT
spectrogram
Welch PSD
wavelet transform
CWT
DWT
Stockwell transform
multitaper spectral estimation

This is important because existing seizure-prediction work has used very different representations.

For example, Truong et al. used STFT on 30-second EEG windows with overlapping segments before CNN classification.

A 2024 study used the Stockwell transform followed by a multi-channel Vision Transformer.

So APALA should not assume Welch PSD is automatically the optimal representation.

5. Your 4-second window deserves investigation

Current APALA:

4-second EEG window
stride = 1 second
↓
Welch PSD
↓
classification

That's computationally convenient, but the question is:

Is 4 seconds an appropriate temporal context for identifying a preictal state?

Compare:

2 s
4 s
8 s
16 s
30 s
60 s

But don't simply compare them as isolated classifiers.

A better experiment is:

same patients
same preprocessing
same model
same LOPO split
different temporal context

Then measure:

ROC-AUC
PR-AUC
seizure sensitivity
FAR/h
warning time
6. Preictal definition is a major research decision

Current APALA:

30 min before seizure
       ↓
   PREICTAL

>4 h away
       ↓
  INTERICTAL

But preictal is not a universally fixed 30-minute biological state.

The literature uses different:

preictal windows
seizure prediction horizons
seizure occurrence periods
exclusion/refractory periods.

A review describes the relationship conceptually as:

PREICTAL PERIOD
│
├── SPH
│
└── SOP

where SPH is the minimum warning interval and SOP is the period in which seizure occurrence is expected.

Therefore, APALA needs explicit definitions.

For example:

Preictal period = 30 min
SPH = ?
SOP = ?
Refractory period = ?

You cannot compare APALA's results with another paper unless these definitions are compatible.

7. This is one of the biggest weaknesses in your current design

Your current system essentially performs:

4-second window
       ↓
P(preictal)
       ↓
threshold
       ↓
alarm

The audit itself identifies this as the source of very high false-alarm rates.

But the correct solution isn't simply:

"Use a 60-second majority vote."

You need to research event-level forecasting.

The alarm system should look more like:

EEG windows
    ↓
model probabilities
    ↓
temporal aggregation
    ↓
alarm state machine
    ↓
refractory period
    ↓
SOP/SPH evaluation

Post-processing has long been studied specifically to reduce false alarms in seizure prediction.

A recent study also emphasizes that seizure sensitivity and false alarms/hour need to be evaluated at the event level, rather than treating every window independently.

8. Research the alarm state machine

This should become an actual APALA research component.

For example:

NORMAL
   │
   │ sustained P(preictal)
   ↓
WARNING
   │
   │ seizure occurs
   ↓
SUCCESS
   │
   │ no seizure
   ↓
FALSE ALARM
   │
   ↓
REFRACTORY
   │
   ↓
NORMAL

Research:

moving average
majority vote
exponential moving average
consecutive-positive windows
hysteresis
refractory period
alarm persistence
alarm merging
hidden Markov models
temporal CNN
LSTM/GRU
temporal Transformer

Don't hard-code "15/15 windows" because the existing audit suggested it. Treat it as one experimental condition.

9. The evaluation protocol is more important than the architecture

This is the biggest lesson from the literature.

Randomly splitting EEG windows can produce highly optimistic results because neighboring windows from the same patient are strongly related.

The cross-patient systematic review specifically identifies randomized splitting as a source of leakage and optimistic evaluation.

Therefore APALA should use:

OUTER LOOP
──────────────────────

Patient/case X
      ↓
FINAL TEST ONLY

Remaining cases
      ↓
TRAIN + VALIDATION

Then:

TRAIN
 ↓
fit preprocessing
 ↓
fit normalization
 ↓
fit model
 ↓
select threshold
 ↓
select smoothing parameters
 ↓
TEST unseen patient

Nothing learned from the test patient should influence model development.

That includes:

normalization
channel selection
feature selection
threshold
smoothing window
probability calibration
hyperparameters
ensemble weights
augmentation choices.

A 2026 CHB-MIT study explicitly describes this leakage-safe structure: channel harmonization, representation learning, feature refinement, model fitting and operating-point calibration were confined to the training/validation side of each outer fold, with the held-out patient used only for final evaluation.

10. You actually need nested validation

Your final architecture should conceptually be:

                 ALL CASES
                     │
          ┌──────────┴──────────┐
          │                     │
     OUTER TEST            TRAINING POOL
     unseen case                 │
                                 ↓
                           INNER VALIDATION

Outer loop:

estimates generalization.

Inner validation:

chooses the model/threshold/hyperparameters.

This prevents:

test result
 ↓
change threshold
 ↓
rerun
 ↓
report improved result

which is effectively test-set tuning.

11. Threshold selection needs redesign

Your audit currently proposes:

choose threshold where FAR ≤ 2/h

The idea is reasonable, but implementation must be leakage-safe.

A better process:

TRAIN
 ↓
learn model

VALIDATION
 ↓
select threshold + temporal parameters
 ↓
freeze everything

TEST PATIENT
 ↓
evaluate once

You can define an operating-point objective such as:

maximize seizure sensitivity
subject to FAR ≤ X/hour

but the value of X must be justified as an experimental target, not called an established universal clinical standard.

12. Don't optimize only ROC-AUC

ROC-AUC can look good while the operational alarm system is poor.

APALA should report:

Window level
ROC-AUC
PR-AUC
sensitivity
specificity
precision
F1
balanced accuracy
Event level
seizure sensitivity
false alarms/hour
warning time
detection latency
time in warning
number of alarms
missed seizures

The literature emphasizes SOP/SPH and false-alarm reporting precisely because ordinary classification metrics don't capture the forecasting problem.

13. Research baselines before NeuroKAN

This is something I strongly recommend changing in APALA v2.

Build:

Baseline 0
majority classifier

Baseline 1
logistic regression

Baseline 2
SVM / classical features

Baseline 3
small CNN

Baseline 4
ResNet

Baseline 5
ResNet + SE

Baseline 6
ResNet + MLP

Baseline 7
ResNet + KAN

Baseline 8
ResNet + KAN + temporal layer

Now you can answer:

What actually contributes to APALA?

If:

ResNet + MLP = 0.78
ResNet + KAN = 0.77

then KAN isn't helping.

If:

ResNet + MLP = 0.72
ResNet + KAN = 0.79

now KAN is scientifically interesting.

14. Research KAN itself

The original KAN paper proposes replacing fixed node activations/linear weights with learnable univariate functions parameterized using splines.

Your APALA currently uses:

CNN
 ↓
GAP
 ↓
B-spline KAN
 ↓
B-spline KAN
 ↓
prediction

But don't assume:

KAN = better EEG model.

Research:

KAN vs MLP
spline basis
grid size
spline order
regularization
computational cost
optimization stability
interpretability
KAN for biomedical signals
KAN for EEG

There is already EEG-specific KAN work, but much of it is seizure detection, which isn't automatically equivalent to seizure forecasting.

That's an important distinction.

15. Research CNN architecture for EEG

You need to understand what the dimensions mean.

Current:

22 channels × frequency × time

A conventional image CNN assumes local spatial structure.

But EEG has:

electrode relationships
+
frequency relationships
+
temporal relationships

Therefore investigate:

Option A

2D CNN

channel × time/frequency
Option B

1D temporal CNN

channel × time
Option C

spatial-temporal CNN

spatial electrode relationships
        +
temporal dynamics
Option D

graph neural network

electrodes → graph nodes
electrode connectivity → edges
Option E

Transformer

EEG patches → tokens → attention

A 2024 seizure-prediction study specifically explored a multi-channel ViT with a time-frequency representation.

16. Multi-scale representation is worth testing

Your current audit proposes two Welch resolutions.

The underlying idea is sound:

short temporal context
       ↓
higher temporal/frequency detail

long temporal context
       ↓
better low-frequency resolution

But test alternatives:

A: Welch

B: STFT

C: Wavelet

D: Welch + Wavelet

E: multi-resolution STFT

F: learned raw-signal representation

The literature already contains multi-resolution CNN approaches that explicitly separate frequency bands and report CHB-MIT results.

So this isn't just an arbitrary architectural modification.

17. Research raw EEG vs engineered representations

APALA currently does:

raw EEG
 ↓
Welch PSD
 ↓
CNN

But modern EEG models increasingly explore learning representations directly from EEG.

Compare:

Pipeline A
Raw → PSD → CNN
Pipeline B
Raw → STFT → CNN
Pipeline C
Raw → Wavelet → CNN
Pipeline D
Raw → CNN
Pipeline E
Raw → pretrained EEG encoder → forecasting head

The last direction is particularly interesting.

18. Research self-supervised EEG learning

This could become the second major research pillar of APALA.

Instead of training only on:

preictal
vs
interictal

use huge quantities of unlabeled EEG:

unlabeled EEG
      ↓
self-supervised pretraining
      ↓
EEG representation
      ↓
forecasting head
      ↓
CHB-MIT LOPO

Study:

masked signal modeling
contrastive learning
temporal contrastive learning
masked autoencoders
vector quantization
EEG transformers
transfer learning.

LaBraM is an important reference: it uses channel patches, a neural tokenizer and masked prediction, and was pretrained on roughly 2,500 hours of EEG across multiple datasets.

But there's a critical research issue:

Pretraining-data contamination.

A foundation model may already have seen recordings from a benchmark dataset.

NeuralBench explicitly warns about dataset overlap for some LaBraM evaluations.

For APALA, you need to record:

pretraining dataset
↓
does it contain CHB-MIT?
↓
which patients?
↓
which recordings?
↓
if yes → contamination?

This is exactly the kind of detail that makes a research project credible.

19. Research contrastive learning

This is particularly relevant because the central challenge is:

Patient A EEG
       ≠
Patient B EEG

while you want:

preictal(A)
       ≈
preictal(B)

interictal(A)
       ≈
interictal(B)

in representation space.

Potential experiment:

EEG encoder
      ↓
embedding
      ↓
contrastive objective
      ↓
patient-invariant representation
      ↓
classifier

Patient-independent work has already explored contrastive/multi-task strategies and reported CHB-MIT results, but reported metrics need to be interpreted in light of their exact cohorts and protocols rather than compared directly with your LOPO numbers.

20. Research domain generalization

This may actually be a stronger research direction than KAN.

The fundamental problem:

TRAIN PATIENTS
A B C D E F ...
       ↓
distribution P_train

TEST PATIENT
X
       ↓
distribution P_test

These distributions differ.

Research:

domain adaptation
domain generalization
invariant risk minimization
domain adversarial learning
CORAL
feature alignment
patient-adversarial training
patient-invariant embeddings
test-time adaptation.

But be careful:

test-time adaptation can violate your definition of "true patient-independent prediction" if it uses target-patient labeled or future information.

You must define exactly what adaptation is permitted.

21. Patient-specific vs patient-independent

This distinction is crucial.

Patient-specific
Patient X
 ↓
collect data
 ↓
train/fine-tune model
 ↓
predict X
Patient-independent
A B C D E ...
 ↓
train
 ↓
X arrives
 ↓
predict X
Patient-independent + calibration
A B C D...
 ↓
general model
 ↓
new patient X
 ↓
limited calibration data
 ↓
adapt

The latter can dramatically improve performance, but it is not the same research claim as zero-calibration LOPO.

A 2024 calibration study demonstrated that patient-independent forecasting performance can improve substantially when even limited data from the new patient is used for calibration.

Therefore APALA should have clearly separated modes:

APALA-ZERO
No target-patient calibration

APALA-CAL
Limited target-patient calibration

Don't mix them.

22. Research data augmentation

Investigate:

Time domain
jitter
scaling
time warping
cropping
permutation
Frequency domain
frequency masking
band dropout
spectral perturbation
Channel domain
channel dropout
channel permutation
channel perturbation.

Channel perturbation has already been explored for patient-independent EEG seizure models.

But every augmentation needs a biological justification.

For example:

Is randomly deleting an EEG electrode realistic?

Potentially yes.

Is arbitrarily permuting electrode channels realistic?

Usually not unless your architecture explicitly treats them as unordered.

23. Research class imbalance correctly

Your current audit uses:

Focal Loss γ = 2

But class imbalance is more complicated than:

interictal > preictal

There is also:

patients with many seizures
vs
patients with few seizures

and:

many windows from same recording
vs
few windows from another recording

If you train window-wise, patients with huge recordings can dominate the optimization.

Investigate:

class weighting
focal loss
balanced sampling
patient-balanced sampling
seizure-balanced sampling
hard-negative mining
curriculum learning.
24. Don't let the model memorize patients

This is a fascinating experiment for APALA.

Train an auxiliary classifier:

EEG embedding
      ↓
predict patient ID

If your representation is extremely patient-specific, patient ID may be easy to predict.

You could investigate:

representation
      ↓
patient classifier

versus:

representation
      ↓
patient-adversarial objective

The goal:

retain seizure-related information
remove patient-specific nuisance information

This would make the "patient-independent" claim much more scientifically meaningful.

25. Calibration

Your model outputs:

P(preictal)

But does:

0.9

actually mean roughly:

90% probability?

Not necessarily.

Research:

reliability diagrams
Expected Calibration Error
Brier score
temperature scaling
Platt scaling
isotonic regression.

The 2024 patient-independent forecasting work specifically studied calibration because raw deep-learning outputs and operational decision thresholds can behave poorly across unseen patients.

This belongs in APALA v2.

26. Statistical testing

Don't report:

APALA = 0.82

and stop.

You have multiple LOPO cases.

Report:

mean
median
standard deviation
IQR
95% confidence interval
per-patient distribution

Also consider:

bootstrap confidence intervals
paired comparisons across folds
permutation tests
Wilcoxon signed-rank test for paired model comparisons.

The patient is the independent experimental unit, not every 4-second EEG window.

This is extremely important.

You cannot treat 150,000 windows as 150,000 independent samples for statistical significance.

27. Compare against a random predictor

This is another important research concept.

A seizure prediction system should be compared with an appropriate chance/random-warning model.

Truong et al. explicitly considered whether the prediction method was statistically better than a random predictor.

For APALA, investigate:

APALA
vs
random predictor

under the same:

warning duration
SOP
alarm budget
evaluation period.

This helps answer:

Is the system actually predicting seizures rather than merely generating alarms at a favorable rate?

28. Research false-alarm economics

FAR/h isn't just a metric.

Suppose:

1 false alarm / hour

sounds good.

But over 24 hours:

~24 potential alarms/day

depending on the alarm/refractory definition.

That's why you should report:

false alarms/hour
false alarms/day
time in warning
alarm duration

alongside seizure sensitivity.

A recent systematic review specifically highlights inconsistent false-alarm reporting as a major limitation of the literature.

29. Research warning time

APALA currently targets 30 minutes.

But you should evaluate:

5 min
10 min
20 min
30 min
45 min
60 min

The key question:

Does sensitivity remain meaningful as warning time increases?

Because:

earlier warning
        ↕
prediction difficulty

is a fundamental tradeoff.

Don't simply maximize warning time.

30. External validation

CHB-MIT alone is not enough to claim generalization to all EEG patients.

The systematic review emphasizes limited external validation and substantial methodological heterogeneity in seizure-prediction studies.

Eventually APALA should be tested on another dataset if licensing/access permits.

Possible directions:

Siena
Freiburg
TUH EEG
other scalp EEG forecasting datasets
clinical institutional data if available.

The exact task must be harmonized carefully because seizure detection datasets are not automatically suitable for forecasting.

31. Important distinction: detection vs forecasting datasets

This is an easy trap.

A paper saying:

"patient-independent seizure detection"

doesn't necessarily provide evidence for:

"patient-independent seizure forecasting."

The 2026 CHB-MIT paper I found is explicitly seizure detection, not necessarily the same forecasting task APALA defines. Its strong specificity/FAR results therefore should not be directly presented as forecasting SOTA.

This distinction should be built into your literature table.

32. Literature table you should build

Your APALA research database should have:

Field	Record
Paper	
Year	
Dataset	
Unique patients	
Evaluation cases	
EEG type	scalp/iEEG
Sampling rate	
Channels	
Preprocessing	
Window length	
Window stride	
Representation	raw/STFT/Welch/wavelet/etc.
Preictal definition	
SPH	
SOP	
Interictal definition	
Model	
Patient-independent?	
LOPO?	
Calibration?	
Temporal postprocessing?	
Threshold selection	
Seizure sensitivity	
FAR/h	
AUC	
Warning time	
Limitations	
Leakage concerns	

This is more useful than collecting papers and their headline AUC.

33. Papers I would make mandatory reading
Foundational seizure prediction

Truong et al., 2018 — CNNs for seizure prediction

Important because it demonstrates a time-frequency CNN approach and discusses generalization versus patient-specific feature engineering.

Deep learning

Daoud & Bayoumi, 2019

Important historically, but patient-specific, so don't use it as direct evidence for APALA's patient-independent claim.

Patient-independent

Dissanayake et al., 2020

Study specifically focused on patient-independent seizure prediction using deep learning and CHB-MIT.

Cross-patient methodology

Shafiezadeh et al., 2024

This should probably be one of your most important papers because it directly examines patient-independent seizure prediction methodology.

Calibration

Calibrating Deep Learning Classifiers for Patient-Independent EEG Seizure Forecasting, 2024

Extremely relevant to APALA's threshold/calibration problem.

Temporal post-processing

EEG epilepsy seizure prediction: the post-processing stage as a chronology

Important for understanding how model probabilities become actual alarms.

Multi-resolution

Multiresolution CNN seizure prediction

Useful for your proposed multi-scale representation.

EEG foundation models

LaBraM, ICLR 2024

Important if you eventually want to move beyond supervised CHB-MIT training.

KAN

Liu et al., 2024 — KAN

Required before claiming NeuroKAN is theoretically motivated.

34. What I would NOT research yet

Don't waste your first month on:

huge Transformer architectures
H100 vs H200 benchmarks
CUDA optimizations
ONNX deployment
mobile deployment
UI
Flask/FastAPI
cloud architecture
fancy dashboards
model quantization
100 different loss functions.

Those are engineering details.

Your central scientific problems are:

What is preictal?
        ↓
How should EEG be represented?
        ↓
Can we generalize across patients?
        ↓
How do we prevent leakage?
        ↓
How do we convert probabilities into alarms?
        ↓
How do we evaluate forecasting honestly?
35. What APALA v2 should actually look like

I would rebuild it around this:

                    APALA v2
                       │
                       ▼
                 CHB-MIT EDF
                       │
                       ▼
              Signal harmonization
                       │
                       ▼
            Leakage-safe preprocessing
                       │
              ┌────────┴─────────┐
              │                  │
          Raw EEG          Time-frequency
              │                  │
              │          ┌───────┼────────┐
              │          │       │        │
              │         STFT    Welch   Wavelet
              │          │       │        │
              └──────────┴───────┴────────┘
                       │
                       ▼
              Patient-independent
                 representation
                       │
             ┌─────────┴─────────┐
             │                   │
          CNN/ResNet          NeuroKAN
             │                   │
             └─────────┬─────────┘
                       ▼
                 P(preictal)
                       │
                       ▼
              Probability calibration
                       │
                       ▼
             Temporal aggregation
                       │
                       ▼
                Alarm state machine
                       │
                       ▼
              SPH / SOP evaluation
                       │
                       ▼
              Patient-level metrics
                       │
                       ▼
              24-case LOPO results
36. Your experiments should be designed before coding

This is the biggest change I'd make to your workflow.

Don't start with:

"Let's build NeuroKAN."

Start with an experiment matrix.

Experiment A — representation
Welch
vs
STFT
vs
Wavelet
vs
multi-scale
Experiment B — architecture
CNN
vs
ResNet
vs
ResNet + SE
vs
ResNet + MLP
vs
ResNet + KAN
Experiment C — temporal modeling
none
vs
moving average
vs
majority vote
vs
state machine
vs
temporal neural network
Experiment D — training
standard CE
vs
weighted CE
vs
focal
Experiment E — representation learning
supervised
vs
contrastive pretraining
vs
self-supervised EEG encoder
Experiment F — generalization
ordinary LOPO
vs
patient-balanced
vs
domain-generalized representation

Then each experiment answers one scientific question.
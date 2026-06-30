# EEGNet for Motor Imagery Classification 🧠⚡

A self-study journey from neural network fundamentals to a working **EEGNet** pipeline for classifying left-hand vs. right-hand motor imagery from real EEG data (PhysioNet EEGBCI dataset). Built over Summer 2026 while learning ML for neuroscience and brain-computer interfaces (BCI).

This repo is organized chronologically as a "build-up" — starting from raw gradient descent and a single neuron, through a hand-rolled autograd engine, and finally into a full EEGNet implementation with subject-wise generalization testing (LOSO).

## Why this repo exists

I'm a first-year engineering student going deep on BCI/neurorobotics, with the goal of eventually building real EEG decoders. Rather than jumping straight into `import eegnet` from a library, I wanted to understand every layer of the stack — backpropagation by hand, autograd internals, RNN/LSTM sequence modeling, and signal processing — before applying it to real EEG signals. This repo documents that progression.

## Repository structure

```
eegnet-motor-imagery/
├── week1/                      Neural net fundamentals from scratch
│   ├── neuron.py                  Single neuron forward pass
│   ├── gradient_descent.py        Gradient descent basics
│   ├── grad_desc_q2.py            Gradient descent practice problem
│   ├── grad_desc_non_polynomial.py  Gradient descent on non-polynomial loss
│   ├── connecting_neuron_loss.py  Wiring neurons to a loss function
│   └── xor_from_scrach.py         XOR problem solved from scratch (no autograd)
│
├── week2/                      Autograd & PyTorch fundamentals
│   ├── micrograd_engine.py        Hand-built autograd engine (à la Karpathy's micrograd)
│   ├── micrograd_test1.py         Tests for the micrograd engine
│   ├── neuron.py                  Neuron class built on the micrograd engine
│   ├── neuron_test_xor.py         XOR solved using the micrograd engine
│   └── pytorch_eeg_tensor.py      First look at representing EEG data as PyTorch tensors
│
├── week3/                      Sequence models (RNN/LSTM)
│   ├── kinematics_rnn.py          RNN modeling 3-phase motor kinematics
│   ├── rlc_trancient.py           RNN modeling an RLC circuit transient response
│   └── rlc_using_lstm.py          Same RLC problem solved with an LSTM
│
├── week4/                      Applied time-series regression
│   └── kaggle_Electric_motor_temprature.py  LSTM-based regression on the Kaggle
│                                              "Electric Motor Temperature" dataset
│
├── MNE-python_projects/        Core EEG → EEGNet pipeline
│   ├── 01_physionet_ingest.py     Download & preprocess PhysioNet EEGBCI data with MNE
│   │                                (bandpass + notch filtering, epoching)
│   ├── 02_feature_extraction.py   Load epochs and extract raw data arrays
│   ├── 03_baseline.py             Classical baseline: CSP + LDA with cross-validation
│   ├── 04_pytorch_engine.py       Converting EEG epochs into PyTorch-ready tensors
│   ├── eegnet_model.py            EEGNet architecture (temporal conv → depthwise spatial
│   │                                conv → separable conv → classifier), implemented in PyTorch
│   ├── 06_train_eegnet.py         Training loop for EEGNet on a single subject
│   ├── 07_eegnet_eval.py          Evaluation / metrics for the trained model
│   ├── 08_mass_ingest.py          Multi-subject data ingestion from PhysioNet
│   ├── 09_massive_training.py     Training EEGNet on the pooled multi-subject dataset
│   └── 10_loso_pipeline.py        Leave-One-Subject-Out (LOSO) cross-subject
│                                    generalization pipeline
│
└── test_1.py                   Scratch/testing file
```

## The pipeline, end to end

1. **Ingest** — Raw EDF files for the PhysioNet EEGBCI motor imagery dataset are downloaded via `mne.datasets.eegbci`, band-pass filtered (8–30 Hz, the mu/beta rhythm band relevant to motor imagery) and notch-filtered (50 Hz) to remove line noise.
2. **Epoch** — Continuous EEG is cut into trial-aligned epochs around the `rest` / `left_fist` / `right_fist` event markers.
3. **Baseline** — A classical pipeline (Common Spatial Patterns + Linear Discriminant Analysis) is run as a sanity-check baseline using cross-validation.
4. **EEGNet** — A compact convolutional architecture purpose-built for EEG (Lawhern et al., 2018) is implemented in raw PyTorch:
   - **Block 1 (Temporal Convolution)** — acts like a learned bandpass filter
   - **Block 2 (Depthwise Spatial Convolution)** — acts like learned CSP, one filter per EEG channel group
   - **Block 3 (Separable Convolution)** — efficient feature extraction
   - **Classifier** — flattened dense layer → 2-class softmax (left vs. right hand)
5. **Scale & train** — Raw MNE data is in volts; it's rescaled to microvolts (`* 1e6`) before training, and labels are shifted to be zero-indexed.
6. **Single-subject → multi-subject → LOSO** — The pipeline is progressively scaled from training/testing on one subject, to pooling multiple subjects, to a proper **Leave-One-Subject-Out** evaluation — training on subjects 1–4 and testing on a held-out subject 5 — to honestly measure how well the model generalizes to unseen people (the real bar for any usable BCI decoder).

## Setup

```bash
git clone https://github.com/pranjalkumar0001/eegnet-motor-imagery.git
cd eegnet-motor-imagery
pip install -r requirements.txt
```

The first time you run any `MNE-python_projects` ingest script, `mne` will automatically download the relevant PhysioNet EEGBCI subject runs to its local data cache (no manual download needed, but it does require an internet connection).

## Running the EEGNet pipeline

```bash
cd MNE-python_projects

# 1. Pull and preprocess one subject's EEG data
python 01_physionet_ingest.py

# 2. Sanity-check with a classical CSP+LDA baseline
python 03_baseline.py

# 3. Train EEGNet on a single subject
python 06_train_eegnet.py
python 07_eegnet_eval.py

# 4. Scale up to multiple subjects
python 08_mass_ingest.py
python 09_massive_training.py

# 5. Leave-One-Subject-Out generalization test
python 10_loso_pipeline.py
```

## Requirements

- Python 3.9+
- See [`requirements.txt`](./requirements.txt)

## Background reading

- Lawhern, V. J., et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.* Journal of Neural Engineering.
- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [MNE-Python documentation](https://mne.tools/stable/index.html)

## License

No license has been added yet — all rights reserved by default. Open an issue if you'd like to use this code and I'm happy to add an explicit license (MIT is likely).

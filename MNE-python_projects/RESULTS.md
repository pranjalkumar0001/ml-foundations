# Results

This file documents the actual progression of experiments in this repo — including the failures, since the failures are most of what I learned from. Each phase corresponds to scripts in `MNE-python_projects/`.

> **Note:** Fill in the exact run date / commit for each result below before treating this as final. Numbers should be reproducible by re-running the corresponding script.

## Summary

| Phase | Approach | Evaluation | Accuracy | Outcome |
|---|---|---|---|---|
| 1 | CSP + LDA (classical baseline) | 5-fold cross-validation | 66.67% | Baseline established |
| 2 | Vanilla EEGNet, 1 subject | 80/20 train/test split | 44.44% | Severe overfitting |
| 3 | EEGNet, 5 subjects, raw volts | Full-batch training | 50.00% | Model collapse |
| 4 | EEGNet, 5 subjects, microvolts | Randomized 80/20 split | 95.24% | Data leakage (invalid) |
| 5 | EEGNet, microvolts, mini-batch | Leave-One-Subject-Out | 83.33% | True generalization benchmark |

The headline result is **Phase 5: 83.33% accuracy on a subject the model never saw during training** (trained on subjects 1–4, tested on subject 5). That's the only number in this table that represents an honest measure of whether the model learned something general about motor imagery, rather than something specific to the people in the training set — everything before it was either a necessary baseline or a bug.

## Phase 1 — Classical baseline (CSP + LDA)

**Script:** `03_baseline.py`

Goal: check whether the raw EEG signal contains a linearly separable motor imagery signal at all, before reaching for deep learning.

Pipeline: epoched `.edf` data → Common Spatial Patterns (CSP, 4 components) → Linear Discriminant Analysis, evaluated with 5-fold stratified cross-validation.

Result: **66.67%** mean accuracy across folds. CSP successfully picks up spatial variance between left-fist and right-fist imagery, but is capped by the linear decision boundary of LDA. This became the number every later model had to beat to justify the added complexity.

## Phase 2 — Vanilla EEGNet, single subject (overfitting)

**Script:** `06_train_eegnet.py` / `07_eegnet_eval.py`

Goal: train the PyTorch EEGNet implementation on one subject (45 trials) and beat the classical baseline.

Result: **100% training accuracy, 44.44% test accuracy.** A textbook overfitting failure — 45 trials is nowhere near enough data for a CNN's parameter count, so the network memorized training-set noise instead of a generalizable signal. Test accuracy below random guessing made it clear the model had learned nothing useful.

## Phase 3 — Scaling to 5 subjects (model collapse)

**Script:** `08_mass_ingest.py` / early `09_massive_training.py`

Goal: scale to 5 subjects (220+ trials) with weight decay added to fight overfitting.

Result: **50.00% accuracy exactly** — the model collapsed and predicted the same class for every trial. Root cause was two stacked bugs:

1. **Vanishing gradients from unit scale.** Raw MNE EEG data is in volts (~1e-6 range). Backpropagating through values that small effectively zeroed out useful gradient signal.
2. **Full-batch training.** Processing all 220+ trials as one batch averaged out the per-trial signal variance the optimizer needed to make progress, and training got stuck in a degenerate minimum.

## Phase 4 — Microvolt fix + mini-batching (data leakage)

**Script:** `09_massive_training.py`

Goal: fix the collapse by rescaling inputs to microvolts (`X * 1e6`) and switching to mini-batch gradient descent (`batch_size=16` via `DataLoader`).

Result: **95.24% accuracy** — but invalid. The train/test split was a random 80/20 split across all 5 subjects pooled together, which meant trials from the same subject ended up on both sides of the split. The model wasn't learning a general motor-imagery rule; it was partly recognizing each subject's individual EEG signature ("neural handwriting") that leaked across the split. This number is included here specifically as a cautionary example — it's the kind of result that looks great until you ask the right question about how the split was constructed.

## Phase 5 — Leave-One-Subject-Out (true benchmark)

**Script:** `10_loso_pipeline.py`

Goal: eliminate the leakage in Phase 4 by strictly separating subjects between train and test.

- **Training set:** subjects 1, 2, 3, 4 (pooled)
- **Test set:** subject 5 (never seen during training, in any form)

Result: **83.33% accuracy** on the held-out subject. The ~12-point drop from Phase 4 is the honest cost of removing subject leakage — and the fact that the model still clears 83% on a brain it has never encountered suggests it picked up on something closer to a general physiological pattern of motor imagery, rather than purely memorizing individual subjects.

## Limitations of the current results

- LOSO is currently evaluated against a single held-out subject (5). A more rigorous version would loop LOSO across all subjects and report mean ± standard deviation, since a single held-out subject is a noisy estimate of true generalization.
- No statistical significance testing has been done on the gap between CSP+LDA (66.67%) and LOSO EEGNet (83.33%).
- Training/test class balance, exact hyperparameters (learning rate, epochs, weight decay) used for each phase aren't yet logged in a reproducible config — currently they live as hardcoded values inside each script.

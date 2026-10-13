# Simplicity Bias: A Falsification Attempt

A small research study investigating whether neural networks preferentially
learn simpler features, and whether any such preference is intrinsic to
"simplicity" or is explained by correlated factors like optimization ease,
predictive strength, and label noise.

This repository contains the code, results, and experiment log for the study.
It was conducted as part of the Falsification Challenge.

---

## Research question

When a simple feature (S, linearly separable) and a complex feature (C,
k-slab, non-linearly separable) both predict the label, does a neural network
prefer S because S is *simpler*, or because S is easier to optimize, cleaner,
or more predictive?

The study does not attempt to prove that simplicity bias exists. It attempts
to find conditions under which the claim fails, weakens, or requires
qualification.

---

## Repository structure

```text
byte-ai-research-falsification-challenge-2026/
|-- notebook/
|   |-- 01.ipynb
|   `-- 02.ipynb
|-- results/
|   |-- exp0.csv
|   |-- exp1_dynamics.csv
|   |-- exp1_summary.csv
|   |-- exp2_pilot_dynamics.csv
|   |-- exp2_pilot_summary.csv
|   `-- exp3_summary.csv
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- diagnostic.py
|   |-- diagnostic_k5.py
|   |-- exp0.py
|   |-- exp1.py
|   |-- exp2.py
|   |-- exp3.py
|   |-- measures.py
|   |-- model.py
|   |-- sanity_check.py
|   `-- train.py
|-- .gitignore
|-- experiment_log.md
|-- pyproject.toml
|-- readme.md
`-- uv.lock
```

### File descriptions

**Source code (`src/`)**

| File                 | Purpose                                                                      |
| -------------------- | ---------------------------------------------------------------------------- |
| `dataset.py`       | Synthetic dataset generator (S, C, distractors) plus conflict-set generator. |
| `model.py`         | Small MLP definition.                                                        |
| `train.py`         | Training loop with optional per-epoch callback.                              |
| `measures.py`      | Reliance measures: shuffle ablation + conflict test.                         |
| `sanity_check.py`  | Logistic regression sanity check on S and C alone.                           |
| `exp0.py`          | Capacity sanity check: can the MLP learn each feature alone?                 |
| `exp1.py`          | Reproduce baseline: S vs C (k=5), both fully predictive.                     |
| `exp2.py`          | S-noise : does the network pivot when S degrades?                            |
| `exp3.py`          | Simplicity vs optimization ease: scale C's input, measure reliance.          |
| `diagnostic.py`    | Negative control: noisy S with a simple, clean C (k=1).                      |
| `diagnostic_k5.py` | Phase-transition diagnostic on C alone at k=5.                               |

**Results (`results/`)**

CSV files produced by each experiment. These are the primary data behind
every table in `experiment_log.md`.

**Root files**

| File                            | Purpose                                                                           |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `experiment_log.md`           | Full scientific log with configs, results, interpretation, and design decisions. |
| `pyproject.toml`, `uv.lock` | Python dependencies and lockfile.                                                 |
| `readme.md`                   | This file.                                                                        |

---

## Dataset design

Each sample is a 10-dimensional vector with a binary label y in {-1, +1}.

| Coordinate | Feature     | Description                                                                                              |
| ---------- | ----------- | -------------------------------------------------------------------------------------------------------- |
| 0          | S (simple)  | Linear. sign(S) predicts the label. Gaussian noise sigma_S around the class mean.                        |
| 1          | C (complex) | k-slab. Values in [0, 1) divided into k+1 equal-width alternating bands. Band parity predicts the label. |
| 2-9        | Distractors | Pure Gaussian noise, independent of the label.                                                           |

Both S and C are individually sufficient to solve the task. The generator
supports:

- `k` -- number of slabs. `k=1` is a single threshold (linear). `k=5` is
  6 alternating bands (non-linear).
- `rho_S`, `rho_C` -- label noise rates. `0.0` means perfect predictor.
- `mu_S`, `sigma_S` -- margin and noise for S, controlling optimization ease.
- `seed` -- required argument; all datasets are reproducible.

The generator also produces a **conflict dataset** where S and C imply
different classes, used to measure which feature a trained model follows.

---

## Model and training

- **Architecture:** MLP with 2 hidden layers, 64 units, ReLU, single logit
  output. `BCEWithLogitsLoss`.
- **Optimizer:** Adam, lr=1e-3, batch=128.
- **Max epochs:** 400 (motivated by Exp 0)
- **Seeds:** 3 per configuration.
- **Split:** 70/30 train/test, stratified.

---

## Reliance measures

Two independent measures of which feature the network uses:

1. **Shuffle ablation.** Shuffle one feature's values across test samples,
   measure accuracy drop. `drop_S` and `drop_C` are reported, along with
   the normalized ratio `R_S_shuffle = drop_S / (drop_S + drop_C)`.
2. **Conflict test.** Generate samples where S implies +1 and C implies -1
   (and vice versa). Measure the fraction of predictions that follow S.
   This is `R_S_conflict` in [0, 1].

Both measures are validated on an untrained model before use. `R_S_shuffle`
returns `NaN` when total drops are under 0.05, to avoid reporting ratios of
noise.

---

## Key findings

- **Exp 0 (capacity):** All features learnable. But S takes ~1 epoch, C at k=5 takes ~100. A separate diagnostic showed C eventually converges, after a phase transition around epoch 150. This set MAX_EPOCHS=400 for all later experiments.
- **Exp 1 (baseline):** With both features perfect, the network relied almost entirely on S (R_S ~ 0.99), and reliance was flat from epoch 20 to 400. Claim reproduced, but the confound was maximal — S was also cleaner and easier to optimize.
- **Exp 2 (noisy S):** Making S noisy did not push the network to C. It overfit the noisy labels instead. The extreme bias is conditional on S being near-perfect.
- **Negative control (k=1, noisy S):** When C was equally simple and cleaner than S, the network pivoted to C almost entirely. Rules out "the network cannot pivot."
- **Exp 3 (scaled C):** Scaling C's input by up to 100x did not flip the preference. Input scale alone is too weak to overcome S's head start.

---

## Overall conclusion

The simplicity bias in this setup is real but conditional. It holds strongly
when the simple feature is clean and the complex feature is genuinely harder
to represent. It weakens when the alternative is equally simple. It does not
reduce to input scale alone.

The claim is **qualified**, not falsified:

> Neural networks in this setup preferentially learn a feature when it is
> simultaneously simpler, cleaner, and easier to optimize than the
> alternative. When these factors are decoupled, the preference follows
> optimization ease and predictive strength rather than simplicity per se.

---

## How to reproduce

### Environment

Python 3.10+. This project uses [uv](https://github.com/astral-sh/uv) for
dependency management (see `pyproject.toml` and `uv.lock`).

With `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

No GPU required. All experiments run on CPU.

### Run order

Run from the `src/` directory:

```bash
# 1. Sanity checks (dataset and features)
python dataset.py
python sanity_check.py

# 2. Capacity sanity check (Exp 0)
python exp0.py

# 3. Phase-transition diagnostic (optional, ~30s)
python diagnostic_k5.py

# 4. Reproduce baseline (Exp 1)
python exp1.py

# 5. S-noise pilot (Exp 2)
python exp2.py

# 6. Negative control (noisy S, simple clean C)
python diagnostic.py

# 7. Simplicity vs optimization ease (Exp 3)
python exp3.py
```

---

## Limitations

- **Synthetic only.** All results are on k-slab synthetic data with a small
  MLP. They may not generalize to images, text, or larger architectures.
- **One definition of simplicity.** "Simple" is operationalized as "linear
  and low-dimensional." Other definitions (low frequency, short description
  length, few parameters) may produce different results.
- **Optimization ease is a proxy.** Time-to-fit and input scale are
  imperfect proxies. Matching optimization ease perfectly between two
  features is not possible.
- **Shuffle ablation assumes feature independence.** The measure is a
  proxy, not a direct readout of internal reliance.
- **Seed sensitivity.** The phase transition in C at k=5 has 3x spread
  across seeds. Three seeds is minimal; more would tighten confidence.
- **No causal claim.** The study shows correlations between feature
  properties and reliance. It does not prove that any specific factor
  causes the preference.

---

## What the study cannot establish

- That the simplicity bias exists or does not exist in general.
- That the results transfer to real datasets or larger models.
- That "simplicity" is a well-defined concept across contexts.
- That the k-slab notion of complexity is the right one.

The study is a controlled qualification of a broad claim, not a refutation.

---

## Primary record

- `experiment_log.md` -- full scientific log with configs, results,
  interpretation, and design decisions.
- `results/*.csv` -- raw numbers behind every table in the log.

Both are the primary record of the study. The code is the means of producing
them.

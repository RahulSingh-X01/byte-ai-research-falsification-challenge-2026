"""
exp0.py
Experiment 0: Capacity sanity check.

Train the MLP on S alone and on C alone (1-dim input each).
Record final accuracy and epochs-to-95% train accuracy.

Purpose: confirm the MLP can fit each feature before we test preference.
If the MLP cannot learn C at high k, then ignoring C downstream says
nothing about simplicity bias — it is a capacity failure.

Pilot-first design:
  Start with SEEDS=[0], K_VALUES_C=[1,5], MAX_EPOCHS=100.
  Measure runtime. Expand only if the pilot is fast enough.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from dataset import generate_dataset
from train import train_model

# --- Speed controls -------------------------------------------------------
torch.set_num_threads(4)   # avoid thread thrash on tiny models

# --- Experiment configuration --------------------------------------------
MAX_EPOCHS = 400         
K_VALUES_C = [1, 5]      
SEEDS = [0, 1, 2]        
N_SAMPLES = 5000     
RESULTS_DIR = "../results"


def run_one(X_feature, y, seed, label=""):
    t0 = time.time()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feature, y, test_size=0.3, random_state=seed, stratify=y
    )
    _, hist = train_model(
        X_tr, y_tr, X_te, y_te,
        input_dim=1, seed=seed, max_epochs=MAX_EPOCHS,
    )
    elapsed = time.time() - t0
    e95 = hist["epochs_to_95"] if hist["epochs_to_95"] is not None else MAX_EPOCHS
    print(
        f"    {label:<8} seed={seed}  {elapsed:5.1f}s  "
        f"train_acc={hist['train_acc'][-1]:.3f}  "
        f"test_acc={hist['test_acc'][-1]:.3f}  "
        f"epochs_to_95={e95}",
        flush=True,
    )
    return {
        "train_acc": hist["train_acc"][-1],
        "test_acc": hist["test_acc"][-1],
        "epochs_to_95": e95,
        "runtime_s": elapsed,
    }


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    t_start = time.time()

    print("=== Experiment 0: Capacity Sanity Check ===")
    print(
        f"n_samples={N_SAMPLES}, seeds={SEEDS}, "
        f"k_values={K_VALUES_C}, max_epochs={MAX_EPOCHS}"
    )
    print()

    # --- S alone ---
    print("Feature S (linear):")
    for seed in SEEDS:
        X, y, _ = generate_dataset(
            seed=seed, n_samples=N_SAMPLES, k=1,
            rho_S=0.0, rho_C=0.0,
        )
        r = run_one(X[:, 0:1], y, seed, label="S")
        r.update({"feature": "S", "k": None, "seed": seed})
        rows.append(r)

    # --- C alone at each k ---
    for k in K_VALUES_C:
        print(f"\nFeature C (k={k} slabs):")
        for seed in SEEDS:
            X, y, _ = generate_dataset(
                seed=seed, n_samples=N_SAMPLES, k=k,
                rho_S=0.0, rho_C=0.0,
            )
            r = run_one(X[:, 1:2], y, seed, label=f"C k={k}")
            r.update({"feature": "C", "k": k, "seed": seed})
            rows.append(r)

    # --- Save CSV ---
    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "exp0.csv")
    df.to_csv(out_path, index=False)

    total = time.time() - t_start
    print()
    print("=" * 74)
    print(f"Total runtime: {total:.1f}s for {len(df)} runs "
          f"({total / len(df):.1f}s per run)")
    print(f"Saved to {out_path}")

    # --- Summary table ---
    print()
    print(f"{'Feature':<8}{'k':<6}{'Train Acc':<20}{'Test Acc':<20}{'Epochs to 95%':<16}")
    print("-" * 74)
    for (feat, k), group in df.groupby(["feature", "k"], dropna=False):
        tr = group["train_acc"]
        te = group["test_acc"]
        ep = group["epochs_to_95"]
        k_str = "-" if pd.isna(k) else str(int(k)) # type: ignore
        print(
            f"{feat:<8}{k_str:<6}"
            f"{tr.mean():.3f} ± {tr.std():.3f}   "
            f"{te.mean():.3f} ± {te.std():.3f}   "
            f"{ep.mean():6.1f} ± {ep.std():5.1f}"
        )


if __name__ == "__main__":
    run()
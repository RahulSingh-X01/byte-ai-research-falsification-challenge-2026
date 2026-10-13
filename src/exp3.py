"""
exp3.py
Experiment 3: Simplicity vs optimization ease.

Hold both features clean and equally predictive. Vary their input scales
to control optimization ease. Track reliance.

If R_S depends on the scale ratio, the "simplicity bias" in Exp 1 is partly
optimization ease. If R_S is invariant, the claim survives a serious test.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from dataset import generate_dataset
from train import train_model
from measures import compute_reliance

torch.set_num_threads(4)

# --- Config ---
SEEDS = [0, 1, 2]
N_SAMPLES = 5000
K = 5
RHO_S = 0.0
RHO_C = 0.0
ALPHA_S = 1.0                   # S scale fixed
ALPHA_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_EPOCHS = 400
CALLBACK_EVERY = 100
RESULTS_DIR = "../results"

def run_one(alpha_C, seed):
    X, y, _ = generate_dataset(
        seed=seed, n_samples=N_SAMPLES, k=K,
        rho_S=RHO_S, rho_C=RHO_C, mu_S=1.0, sigma_S=0.5,
    )
    X = X.copy()
    X[:, 0] *= ALPHA_S
    X[:, 1] *= alpha_C

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )

    log = {"epoch": [], "R_S_conflict": [], "R_S_shuffle": [],
           "baseline_acc": [], "drop_S": [], "drop_C": []}

    def cb(model, epoch):
        r = compute_reliance(
            model, X_te, y_te, k=K, seed=seed,
            alpha_S=ALPHA_S, alpha_C=alpha_C,      # <-- CHANGED
        )
        log["epoch"].append(epoch)
        log["R_S_shuffle"].append(r["R_S_shuffle"])
        log["R_S_conflict"].append(r["R_S_conflict"])
        log["baseline_acc"].append(r["baseline_acc"])
        log["drop_S"].append(r["drop_S"])
        log["drop_C"].append(r["drop_C"])

    t0 = time.time()
    model, hist = train_model(
        X_tr, y_tr, X_te, y_te,
        input_dim=X_tr.shape[1], seed=seed, max_epochs=MAX_EPOCHS,
        epoch_callback=cb, callback_every=CALLBACK_EVERY,
    )
    elapsed = time.time() - t0

    return {
        "alpha_C": alpha_C,
        "alpha_S": ALPHA_S,
        "ratio": alpha_C / ALPHA_S,
        "seed": seed,
        "runtime_s": elapsed,
        "final_train_acc": hist["train_acc"][-1],
        "final_test_acc": hist["test_acc"][-1],
        "epochs_to_95": hist["epochs_to_95"],
        "final_R_S_conflict": log["R_S_conflict"][-1],
        "final_R_S_shuffle": log["R_S_shuffle"][-1],
        "final_drop_S": log["drop_S"][-1],
        "final_drop_C": log["drop_C"][-1],
    }


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    print("=== Experiment 3: Simplicity vs optimization ease ===")
    print(f"k={K}, rho_S=rho_C=0, alpha_S={ALPHA_S}, "
          f"alpha_C in {ALPHA_C_VALUES}, max_epochs={MAX_EPOCHS}\n")

    rows = []
    for alpha_C in ALPHA_C_VALUES:
        print(f"--- alpha_C = {alpha_C} (ratio C:S = {alpha_C/ALPHA_S}) ---")
        for seed in SEEDS:
            r = run_one(alpha_C, seed)
            print(
                f"  seed={seed}  test_acc={r['final_test_acc']:.3f}  "
                f"R_S_conflict={r['final_R_S_conflict']:.3f}  "
                f"R_S_shuffle={r['final_R_S_shuffle']:.3f}  "
                f"drop_S={r['final_drop_S']:.3f}  "
                f"drop_C={r['final_drop_C']:.3f}  "
                f"epochs_to_95={r['epochs_to_95']}",
                flush=True,
            )
            rows.append(r)
        print()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp3_summary.csv"), index=False)

    total = time.time() - t_start
    print("=" * 74)
    print(f"Total runtime: {total:.1f}s for {len(df)} runs")
    print()
    print("Mean reliance by scale ratio:")
    print(f"{'ratio':<10}{'test_acc':<12}{'R_S_conflict':<16}{'R_S_shuffle':<14}{'drop_S':<10}{'drop_C':<10}")
    print("-" * 72)
    for ratio, group in df.groupby("ratio"):
        print(
            f"{ratio:<10.2f}"
            f"{group['final_test_acc'].mean():<12.3f}"
            f"{group['final_R_S_conflict'].mean():<16.3f}"
            f"{group['final_R_S_shuffle'].mean():<14.3f}"
            f"{group['final_drop_S'].mean():<10.3f}"
            f"{group['final_drop_C'].mean():<10.3f}"
        )
    print(f"\nSaved to {RESULTS_DIR}/exp3_summary.csv")


if __name__ == "__main__":
    run()
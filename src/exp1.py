"""
exp1.py
Experiment 1: Reproduce the baseline.

Present both S and C (k=5), both fully predictive.
Train the MLP for 400 epochs.
Track reliance over training.

Question: does the network rely primarily on S?

Expected: yes, it relies on S. This reproduces the original claim
in our controlled setup. The interesting part is *when* it happens.
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
MAX_EPOCHS = 400
CALLBACK_EVERY = 20
RESULTS_DIR = "../results"


def run_one(seed):
    X, y, _ = generate_dataset(
        seed=seed, n_samples=N_SAMPLES, k=K,
        rho_S=0.0, rho_C=0.0, mu_S=1.0, sigma_S=0.5,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )

    # Callback closure
    log = {"epoch": [], "R_S_shuffle": [], "R_S_conflict": [],
           "baseline_acc": [], "drop_S": [], "drop_C": []}

    def cb(model, epoch):
        r = compute_reliance(model, X_te, y_te, k=K, seed=seed)
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

    # Final reliance (already logged at the last callback)
    return {
        "seed": seed,
        "runtime_s": elapsed,
        "final_train_acc": hist["train_acc"][-1],
        "final_test_acc": hist["test_acc"][-1],
        "epochs_to_95": hist["epochs_to_95"],
        "final_R_S_shuffle": log["R_S_shuffle"][-1],
        "final_R_S_conflict": log["R_S_conflict"][-1],
        "log": log,
    }


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    print("=== Experiment 1: Reproduce Baseline (S vs C k=5) ===")
    print(f"n_samples={N_SAMPLES}, seeds={SEEDS}, k={K}, max_epochs={MAX_EPOCHS}")
    print(f"callback every {CALLBACK_EVERY} epochs\n")

    summary_rows = []
    dyn_rows = []

    for seed in SEEDS:
        r = run_one(seed)
        print(
            f"seed={seed}  {r['runtime_s']:5.1f}s  "
            f"test_acc={r['final_test_acc']:.3f}  "
            f"R_S_shuffle={r['final_R_S_shuffle']:.3f}  "
            f"R_S_conflict={r['final_R_S_conflict']:.3f}  "
            f"epochs_to_95={r['epochs_to_95']}",
            flush=True,
        )
        summary_rows.append({k: v for k, v in r.items() if k != "log"})
        for i, ep in enumerate(r["log"]["epoch"]):
            dyn_rows.append({
                "seed": seed,
                "epoch": ep,
                "R_S_shuffle": r["log"]["R_S_shuffle"][i],
                "R_S_conflict": r["log"]["R_S_conflict"][i],
                "baseline_acc": r["log"]["baseline_acc"][i],
                "drop_S": r["log"]["drop_S"][i],
                "drop_C": r["log"]["drop_C"][i],
            })

    df_summary = pd.DataFrame(summary_rows)
    df_dyn = pd.DataFrame(dyn_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "exp1_summary.csv"), index=False)
    df_dyn.to_csv(os.path.join(RESULTS_DIR, "exp1_dynamics.csv"), index=False)

    total = time.time() - t_start
    print()
    print("=" * 74)
    print(f"Total runtime: {total:.1f}s for {len(df_summary)} runs")

    print("\nSummary:")
    print(f"{'seed':<6}{'test_acc':<12}{'R_S_shuffle':<14}{'R_S_conflict':<14}{'epochs_to_95':<12}")
    print("-" * 60)
    for _, row in df_summary.iterrows():
        print(
            f"{int(row['seed']):<6}"
            f"{row['final_test_acc']:<12.3f}"
            f"{row['final_R_S_shuffle']:<14.3f}"
            f"{row['final_R_S_conflict']:<14.3f}"
            f"{row['epochs_to_95']:<12}"
        )

    print("\nMean across seeds:")
    print(f"  R_S_shuffle  = {df_summary['final_R_S_shuffle'].mean():.3f} "
          f"± {df_summary['final_R_S_shuffle'].std():.3f}")
    print(f"  R_S_conflict = {df_summary['final_R_S_conflict'].mean():.3f} "
          f"± {df_summary['final_R_S_conflict'].std():.3f}")

    print(f"\nSaved: {RESULTS_DIR}/exp1_summary.csv, {RESULTS_DIR}/exp1_dynamics.csv")


if __name__ == "__main__":
    run()
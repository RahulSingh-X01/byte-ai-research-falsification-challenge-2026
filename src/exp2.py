"""
exp2.py
Experiment 2 (pilot): Decouple complexity from predictive strength.

Exp 1 showed the MLP relies almost entirely on S when S is a perfect
predictor. Now we ask: does that break when S becomes noisy?

Pilot grid: k=5 fixed, rho_C=0 fixed, rho_S varies in {0, 0.1, 0.2, 0.3}.
3 seeds per config. 12 runs total.

If the pilot shows a flip (R_S drops toward 0.5 or below at some rho_S),
expand to a 2D grid over (rho_S, rho_C).
If no flip occurs even at rho_S=0.3, that is itself a finding: noisy S is
still preferred, so something other than S's perfect predictiveness is
driving the lock-in.
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

# --- Pilot config ---
SEEDS = [0, 1, 2]
N_SAMPLES = 5000
K = 5
RHO_S_VALUES = [0.0, 0.1, 0.2, 0.3]
RHO_C_FIXED = 0.0
MAX_EPOCHS = 400
CALLBACK_EVERY = 50
RESULTS_DIR = "../results"


def run_one(rho_S, rho_C, seed):
    X, y, _ = generate_dataset(
        seed=seed, n_samples=N_SAMPLES, k=K,
        rho_S=rho_S, rho_C=rho_C, mu_S=1.0, sigma_S=0.5,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )

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

    # Final reliance = last callback point
    final = {
        "rho_S": rho_S,
        "rho_C": rho_C,
        "seed": seed,
        "runtime_s": elapsed,
        "final_train_acc": hist["train_acc"][-1],
        "final_test_acc": hist["test_acc"][-1],
        "epochs_to_95": hist["epochs_to_95"],
        "final_R_S_shuffle": log["R_S_shuffle"][-1],
        "final_R_S_conflict": log["R_S_conflict"][-1],
        "final_drop_S": log["drop_S"][-1],
        "final_drop_C": log["drop_C"][-1],
    }
    return final, log


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    print("=== Experiment 2 (pilot): S noise vs reliance ===")
    print(f"n_samples={N_SAMPLES}, seeds={SEEDS}, k={K}, "
          f"rho_C={RHO_C_FIXED}, rho_S in {RHO_S_VALUES}, "
          f"max_epochs={MAX_EPOCHS}")
    print()

    summary_rows = []
    dyn_rows = []

    for rho_S in RHO_S_VALUES:
        print(f"--- rho_S = {rho_S} ---")
        for seed in SEEDS:
            final, log = run_one(rho_S, RHO_C_FIXED, seed)
            print(
                f"  seed={seed}  {final['runtime_s']:5.1f}s  "
                f"test_acc={final['final_test_acc']:.3f}  "
                f"R_S_shuffle={final['final_R_S_shuffle']:.3f}  "
                f"R_S_conflict={final['final_R_S_conflict']:.3f}  "
                f"drop_S={final['final_drop_S']:.3f}  "
                f"drop_C={final['final_drop_C']:.3f}",
                flush=True,
            )
            summary_rows.append(final)
            for i, ep in enumerate(log["epoch"]):
                dyn_rows.append({
                    "rho_S": rho_S,
                    "rho_C": RHO_C_FIXED,
                    "seed": seed,
                    "epoch": ep,
                    "R_S_shuffle": log["R_S_shuffle"][i],
                    "R_S_conflict": log["R_S_conflict"][i],
                    "baseline_acc": log["baseline_acc"][i],
                    "drop_S": log["drop_S"][i],
                    "drop_C": log["drop_C"][i],
                })
        print()

    df_summary = pd.DataFrame(summary_rows)
    df_dyn = pd.DataFrame(dyn_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "exp2_pilot_summary.csv"), index=False)
    df_dyn.to_csv(os.path.join(RESULTS_DIR, "exp2_pilot_dynamics.csv"), index=False)

    total = time.time() - t_start
    print("=" * 74)
    print(f"Total runtime: {total:.1f}s for {len(df_summary)} runs")

    print("\nMean reliance by rho_S:")
    print(f"{'rho_S':<8}{'test_acc':<12}{'R_S_shuffle':<14}{'R_S_conflict':<14}{'drop_S':<10}{'drop_C':<10}")
    print("-" * 68)
    for rho_S, group in df_summary.groupby("rho_S"):
        print(
            f"{rho_S:<8.1f}"
            f"{group['final_test_acc'].mean():<12.3f}"
            f"{group['final_R_S_shuffle'].mean():<14.3f}"
            f"{group['final_R_S_conflict'].mean():<14.3f}"
            f"{group['final_drop_S'].mean():<10.3f}"
            f"{group['final_drop_C'].mean():<10.3f}"
        )

    print(f"\nSaved: {RESULTS_DIR}/exp2_pilot_summary.csv")
    print(f"       {RESULTS_DIR}/exp2_pilot_dynamics.csv")


if __name__ == "__main__":
    run()
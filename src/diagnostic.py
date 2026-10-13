"""
diagnostic_k1.py
Negative control: if C is easy to learn (k=1) and S is noisy (rho_S=0.2),
does the network switch to C?

If yes: the network can pivot when the alternative is equally easy.
If no: the network's lock-in is deeper than just optimization difficulty.
"""

import time
import torch
from sklearn.model_selection import train_test_split

from dataset import generate_dataset
from train import train_model
from measures import compute_reliance

torch.set_num_threads(4)

SEEDS = [0, 1, 2]
N_SAMPLES = 5000
RHO_S = 0.2
RHO_C = 0.0
K = 1                # C is now linear — equally "simple" as S
MAX_EPOCHS = 400


def run_one(seed):
    X, y, _ = generate_dataset(
        seed=seed, n_samples=N_SAMPLES, k=K,
        rho_S=RHO_S, rho_C=RHO_C, mu_S=1.0, sigma_S=0.5,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )
    t0 = time.time()
    model, hist = train_model(
        X_tr, y_tr, X_te, y_te,
        input_dim=X_tr.shape[1], seed=seed, max_epochs=MAX_EPOCHS,
    )
    r = compute_reliance(model, X_te, y_te, k=K, seed=seed)
    return {
        "seed": seed,
        "runtime_s": time.time() - t0,
        "train_acc": hist["train_acc"][-1],
        "test_acc": hist["test_acc"][-1],
        "R_S_conflict": r["R_S_conflict"],
        "drop_S": r["drop_S"],
        "drop_C": r["drop_C"],
    }


if __name__ == "__main__":
    print(f"Diagnostic: k={K} (both features linear), rho_S={RHO_S}, rho_C={RHO_C}")
    print(f"S-Bayes accuracy = 0.977 - 0.954 * rho_S = {0.977 - 0.954 * RHO_S:.3f}")
    print(f"C-only accuracy  = 1.000")
    print()
    for seed in SEEDS:
        r = run_one(seed)
        print(
            f"seed={seed}  {r['runtime_s']:5.1f}s  "
            f"train={r['train_acc']:.3f}  test={r['test_acc']:.3f}  "
            f"R_S_conflict={r['R_S_conflict']:.3f}  "
            f"drop_S={r['drop_S']:.3f}  drop_C={r['drop_C']:.3f}",
            flush=True,
        )
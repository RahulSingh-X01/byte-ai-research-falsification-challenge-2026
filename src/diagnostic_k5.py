"""
diagnostic_k5.py
Is the MLP failing at k=5 because of time, or capacity?

Train on C alone at k=5 for many epochs.
Print train/test accuracy every N epochs.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from dataset import generate_dataset
from train import train_model

torch.set_num_threads(4)

SEED = 0
N_SAMPLES = 5000
K = 5
MAX_EPOCHS = 1000
LOG_EVERY = 50


def run():
    X, y, _ = generate_dataset(
        seed=SEED, n_samples=N_SAMPLES, k=K,
        rho_S=0.0, rho_C=0.0,
    )
    X_C = X[:, 1:2]  # C alone, 1-dim input

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_C, y, test_size=0.3, random_state=SEED, stratify=y
    )

    _, hist = train_model(
        X_tr, y_tr, X_te, y_te,
        input_dim=1, seed=SEED, max_epochs=MAX_EPOCHS,
    )

    print(f"\nC at k={K}, {MAX_EPOCHS} epochs")
    print(f"{'epoch':<8}{'train_acc':<12}{'test_acc':<12}")
    print("-" * 32)
    for e in range(0, MAX_EPOCHS, LOG_EVERY):
        print(
            f"{e+1:<8}{hist['train_acc'][e]:<12.3f}{hist['test_acc'][e]:<12.3f}"
        )

    print(f"\nFinal train_acc: {hist['train_acc'][-1]:.3f}")
    print(f"Final test_acc:  {hist['test_acc'][-1]:.3f}")
    print(f"epochs_to_95:    {hist['epochs_to_95']}")


if __name__ == "__main__":
    run()
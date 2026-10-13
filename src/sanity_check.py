"""
sanity_check.py
Independent check that S is linear and C is linear only when k=1.

Trains logistic regression on S alone and C alone.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dataset import generate_dataset


def lr_accuracy(X_feature, y, seed):
    """Train LR on a single feature column and return test accuracy."""
    X = X_feature.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


def run(k, seed=42, n_samples=20000):
    X, y, meta = generate_dataset(
        seed=seed, n_samples=n_samples, k=k,
        rho_S=0.0, rho_C=0.0, mu_S=1.0, sigma_S=0.5,
    )
    acc_S = lr_accuracy(X[:, 0], y, seed)
    acc_C = lr_accuracy(X[:, 1], y, seed)
    print(f"k={k:2d}  |  LR on S: {acc_S:.3f}  |  LR on C: {acc_C:.3f}")


if __name__ == "__main__":
    print("Logistic regression sanity check")
    print("-" * 50)
    for k in [1, 5]:
        run(k)
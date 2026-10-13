"""
measures.py
Reliance measures for determining which feature a trained MLP uses.

Measures:
  A) Shuffle ablation: shuffle one feature's values across samples,
     measure the accuracy drop.
  B) Conflict test: samples where S and C imply different classes,
     measure which one the model follows.

The primary scalar summary is R_S in [0, 1]:
  R_S = 1  -> model relies entirely on S
  R_S = 0  -> model relies entirely on C
  R_S = 0.5 -> balanced
"""

import numpy as np
import torch

from dataset import generate_conflict_dataset


def _predict_logits(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        return model(X_t).cpu().numpy()


def _accuracy(model, X, y, device="cpu"):
    logits = _predict_logits(model, X, device)
    preds = np.where(logits > 0, 1, -1)
    return float((preds == y).mean())


def shuffle_reliance(model, X, y, feature_idx, seed=0, n_repeats=3, device="cpu"):
    """
    Shuffle one feature column and measure accuracy drop.

    Returns:
      baseline_acc : accuracy on untouched test set
      shuffled_acc : mean accuracy over n_repeats shuffles
      drop         : baseline_acc - shuffled_acc
    """
    rng = np.random.default_rng(seed)
    baseline_acc = _accuracy(model, X, y, device)

    drops = []
    for _ in range(n_repeats):
        X_shuf = X.copy()
        perm = rng.permutation(len(X_shuf))
        X_shuf[:, feature_idx] = X_shuf[perm, feature_idx]
        acc = _accuracy(model, X_shuf, y, device)
        drops.append(baseline_acc - acc)

    return {
        "baseline_acc": baseline_acc,
        "shuffled_acc": baseline_acc - float(np.mean(drops)),
        "drop": float(np.mean(drops)),
    }


def conflict_reliance(
    model,
    n_samples,
    k,
    mu_S=1.0,
    sigma_S=0.5,
    alpha_S=1.0,
    alpha_C=1.0,
    seed=0,
    device="cpu",
):
    """
    Fraction of conflict samples where the model follows S.

    Returns R_S in [0, 1]:
      1 -> always follows S
      0 -> always follows C

    alpha_S and alpha_C scale the S and C columns after generation, so the
    conflict set matches the input scale the model was trained on.
    """
    X, y_S, _ = generate_conflict_dataset(
        n_samples=n_samples, k=k, mu_S=mu_S, sigma_S=sigma_S, seed=seed,
    )
    X = X.copy()
    X[:, 0] *= alpha_S
    X[:, 1] *= alpha_C
    logits = _predict_logits(model, X, device)
    preds = np.where(logits > 0, 1, -1)
    return float((preds == y_S).mean())


def compute_reliance(
    model, X_test, y_test, k, seed=0,
    alpha_S=1.0, alpha_C=1.0, device="cpu",
):
    """
    Compute both reliance measures and combine into a normalized R_S.

    R_S_shuffle = drop_S / (drop_S + drop_C), clipped to [0, 1].
    R_S_conflict = fraction following S on conflict samples.
    """
    r_S = shuffle_reliance(model, X_test, y_test, feature_idx=0, seed=seed, device=device)
    r_C = shuffle_reliance(model, X_test, y_test, feature_idx=1, seed=seed + 1000, device=device)

    denom = r_S["drop"] + r_C["drop"]
    if denom <= 0.05:
        r_S_shuffle = float("nan")
    else:
        r_S_shuffle = r_S["drop"] / denom

    r_S_conflict = conflict_reliance(
        model, n_samples=2000, k=k,
        alpha_S=alpha_S, alpha_C=alpha_C,
        seed=seed + 2000, device=device,
    )

    return {
        "baseline_acc": r_S["baseline_acc"],
        "drop_S": r_S["drop"],
        "drop_C": r_C["drop"],
        "R_S_shuffle": r_S_shuffle,
        "R_S_conflict": r_S_conflict,
    }
    
if __name__ == "__main__":
    import numpy as np
    from model import MLP
    from dataset import generate_dataset

    # Build an untrained model
    torch.manual_seed(0)
    model = MLP(input_dim=10, hidden_dim=64, n_hidden=2)

    # Generate a small test set with both features
    X, y, _ = generate_dataset(
        seed=0, n_samples=2000, k=5, rho_S=0.0, rho_C=0.0,
    )

    print("Untrained model — expected: baseline_acc ~0.5, drops ~0, R_S_conflict ~0.5")
    print()

    r = compute_reliance(model, X, y, k=5, seed=0)
    for key, val in r.items():
        print(f"  {key:15s} = {val}")
"""
dataset.py
Synthetic dataset generator for investigating simplicity bias.

Each sample:
  - Coordinate 0: simple feature S (linear, sign predicts class)
  - Coordinate 1: complex feature C (k-slab, band parity predicts class)
  - Coordinates 2-9: pure Gaussian distractors
"""

import numpy as np
from scipy.stats import norm


def generate_dataset(
    seed,
    n_samples=10000,
    k=5,
    rho_S=0.0,
    rho_C=0.0,
    mu_S=1.0,
    sigma_S=0.5,
    n_distractors=8,
):
    """
    Generate a synthetic binary classification dataset.

    Returns
    -------
    X : np.ndarray (n_samples, 2 + n_distractors)
    y : np.ndarray (n_samples,) in {-1, +1}
    meta : dict with generation parameters
    """
    rng = np.random.default_rng(seed)

    # 1. True labels
    y = rng.choice([-1, 1], size=n_samples)

    # 2. Simple feature S
    #    With prob (1 - rho_S), S agrees with y; with prob rho_S, it disagrees.
    flip_S = rng.random(n_samples) < rho_S
    implied_S = np.where(flip_S, -y, y)
    S = implied_S * mu_S + rng.normal(0, sigma_S, size=n_samples)

    # 3. Complex feature C (k-slab)
    #    Bands 0..k. Band i has parity +1 if i even, -1 if i odd.
    n_bands = k + 1
    band_width = 1.0 / n_bands
    band_parities = np.array([1 if i % 2 == 0 else -1 for i in range(n_bands)])

    flip_C = rng.random(n_samples) < rho_C
    implied_C = np.where(flip_C, -y, y)

    valid_bands = {
        1: np.where(band_parities == 1)[0],
        -1: np.where(band_parities == -1)[0],
    }

    C = np.empty(n_samples)
    for label in (-1, 1):
        mask = implied_C == label
        n_mask = int(mask.sum())
        if n_mask == 0:
            continue
        chosen = rng.choice(valid_bands[label], size=n_mask)
        C[mask] = (chosen + rng.random(n_mask)) * band_width

    # 4. Distractors
    distractors = rng.normal(0, 1, size=(n_samples, n_distractors))

    # 5. Assemble
    X = np.column_stack([S, C, distractors])

    meta = {
        "k": k,
        "rho_S": rho_S,
        "rho_C": rho_C,
        "mu_S": mu_S,
        "sigma_S": sigma_S,
        "n_distractors": n_distractors,
        "n_samples": n_samples,
        "seed": seed,
    }
    return X, y, meta


def bayes_accuracy_simple(mu_S, sigma_S):
    """Bayes accuracy from S alone (ignoring rho_S)."""
    return norm.cdf(mu_S / sigma_S)


# --------------------------------------------------------------------------- #
# Conflict dataset helpers
# --------------------------------------------------------------------------- #

def sample_C_values(n_samples, k, target_class, rng):
    """
    Sample C values whose band parity matches target_class (+1 or -1).
    Used for building conflict-test samples.
    """
    n_bands = k + 1
    band_width = 1.0 / n_bands
    band_parities = np.array([1 if i % 2 == 0 else -1 for i in range(n_bands)])
    valid_bands = np.where(band_parities == target_class)[0]
    chosen = rng.choice(valid_bands, size=n_samples)
    return (chosen + rng.random(n_samples)) * band_width


def generate_conflict_dataset(
    n_samples, k, mu_S=1.0, sigma_S=0.5, n_distractors=8, seed=None,
):
    """
    Generate samples where S and C imply *different* classes.

    Half the samples: S implies +1, C implies -1.
    Half the samples: S implies -1, C implies +1.

    Samples where either feature is ambiguous (sign(S) does not match its
    intended class, or C's parity does not match its intended class) are
    filtered out, so both features are unambiguous by construction.

    Returns:
      X   : (n_samples_kept, 2 + n_distractors)
      y_S : class implied by S (what sign(S) says)
      y_C : class implied by C (what C's band parity says)
    """
    rng = np.random.default_rng(seed)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # Case 1: S implies +1, C implies -1
    S1 = mu_S + rng.normal(0, sigma_S, size=n_pos)      # S > 0 on average
    C1 = sample_C_values(n_pos, k, target_class=-1, rng=rng)

    # Case 2: S implies -1, C implies +1
    S2 = -mu_S + rng.normal(0, sigma_S, size=n_neg)     # S < 0 on average
    C2 = sample_C_values(n_neg, k, target_class=+1, rng=rng)

    S = np.concatenate([S1, S2])
    C = np.concatenate([C1, C2])
    y_S = np.array([+1] * n_pos + [-1] * n_neg)
    y_C = np.array([-1] * n_pos + [+1] * n_neg)

    distractors = rng.normal(0, 1, size=(n_samples, n_distractors))
    X = np.column_stack([S, C, distractors])

    # --- Filter out ambiguous samples ---
    # Keep only samples where sign(S) matches y_S AND C's parity matches y_C.
    n_bands = k + 1
    band_width = 1.0 / n_bands
    bands = np.floor(C / band_width).astype(int)
    band_parities = np.where(bands % 2 == 0, 1, -1)

    s_sign = np.where(S > 0, 1, -1)
    keep = (s_sign == y_S) & (band_parities == y_C)

    X = X[keep]
    y_S = y_S[keep]
    y_C = y_C[keep]
    return X, y_S, y_C


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # --- Main dataset sanity check ---
    X, y, meta = generate_dataset(
        seed=42, n_samples=20000, k=5, rho_S=0.0, rho_C=0.0,
    )
    print("Main dataset sanity check")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)

    acc_S = ((X[:, 0] > 0).astype(int) * 2 - 1 == y).mean()
    print(f"  Accuracy from S alone: {acc_S:.3f}")

    k = meta["k"]
    bands = np.floor(X[:, 1] * (k + 1)).astype(int)
    parities = np.where(bands % 2 == 0, 1, -1)
    acc_C = (parities == y).mean()
    print(f"  Accuracy from C alone: {acc_C:.3f}")

    # --- Conflict dataset sanity check ---
    Xc, yS, yC = generate_conflict_dataset(
        n_samples=5000, k=5, seed=42,
    )
    n_bands = 5 + 1
    band_width = 1.0 / n_bands
    bands = np.floor(Xc[:, 1] / band_width).astype(int)
    parities = np.where(bands % 2 == 0, 1, -1)
    s_sign = np.where(Xc[:, 0] > 0, 1, -1)

    print()
    print("Conflict dataset sanity check")
    print("  n_samples returned:", len(Xc))
    print("  sign(S) matches yS:", (s_sign == yS).mean())
    print("  C parity matches yC:", (parities == yC).mean())
    print("  yS == -yC:", (yS == -yC).all())
    print("Conflict X shape:", Xc.shape)
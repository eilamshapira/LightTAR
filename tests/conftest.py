import numpy as np
import pytest


def make_layers(n=1500, d=96, n_layers=2, seed=0, latent=12):
    """Synthetic 'encoder states': shared latent factors + noise, rows L2-normalized per layer."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, latent))
    mats = []
    for _ in range(n_layers):
        A = rng.standard_normal((latent, d))
        X = Z @ A + 0.7 * rng.standard_normal((n, d)) + 0.3  # nonzero mean, like real embeddings
        mats.append(X / np.linalg.norm(X, axis=1, keepdims=True))
    return np.stack(mats, 1).astype(np.float32), Z


def targets(Z, seed=0):
    rng = np.random.default_rng(seed + 1)
    n = len(Z)
    return {
        "regression": (Z[:, 0] + 0.5 * Z[:, 1] ** 2 + 0.3 * rng.standard_normal(n), False),
        "binary": (((Z[:, 1] + 0.3 * rng.standard_normal(n)) > 0).astype(int), True),
        "multiclass": (np.digitize(Z[:, 2] + 0.3 * rng.standard_normal(n), [-1, 0, 1]), True),
    }


@pytest.fixture(scope="session")
def data():
    X, Z = make_layers()
    return X, targets(Z)


def align(R, P):
    """Max |R - P| after per-column sign alignment."""
    R = np.asarray(R, np.float64)
    P = np.asarray(P, np.float64)
    s = np.sign((R * P).sum(0))
    s[s == 0] = 1
    return float(np.abs(R - P * s).max())

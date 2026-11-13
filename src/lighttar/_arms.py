"""The two arms of ``GatedLightTAR``: the frozen pipeline (raw last layer -> PCA-30) and LightTAR from
moment sums shared across the inner folds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .core import _moment_sums, _split_layers, raw_target

BASELINE = "frozen"
CANDIDATE = "lighttar"
FROZEN_DIM = 30  # the frozen baseline's PCA dimension
FROZEN_SEED = 42  # the frozen baseline's PCA seed (the SEED of MulTaBench's frozen e5 + PCA-30 pipeline)


def _first_layer32(H) -> np.ndarray:
    """The first layer of ``(n, d)``, ``(n, L, d)`` or a list of layers, as a float32 numpy array."""
    return np.asarray(_split_layers(H, as_numpy=True)[0][0], dtype=np.float32)


class FrozenPCA(BaseEstimator, TransformerMixin):
    """The standard frozen pipeline on one column: scikit-learn PCA-30 of the raw first layer (the encoder's
    last layer by default), fitted and applied on float32 states exactly as the upstream pipeline does
    (a float64 fit rotates near-degenerate trailing components)."""

    def fit(self, H, y=None) -> FrozenPCA:
        from sklearn.decomposition import PCA

        X = _first_layer32(H)
        self.n_components_ = max(1, min(FROZEN_DIM, X.shape[1], len(X) - 1))
        self.pca_ = PCA(n_components=self.n_components_, random_state=FROZEN_SEED).fit(X)
        return self

    def transform(self, H) -> np.ndarray:
        return self.pca_.transform(_first_layer32(H)).astype(np.float32, copy=False)


@dataclass
class ColumnStates:
    """One column's float64 layers, its float32 first layer, and the moment sums of all its rows (shared by
    the inner folds)."""

    mats: list
    first32: np.ndarray
    Y: np.ndarray
    classes: np.ndarray | None
    total: np.ndarray

    def rows(self, idx) -> list:
        return [m[idx] for m in self.mats]


def _frozen_arm(cs: ColumnStates, tr, va, head) -> tuple[np.ndarray, np.ndarray]:
    fz = FrozenPCA().fit(cs.first32[tr])
    return fz.transform(cs.first32[tr]), fz.transform(cs.first32[va])


def _lighttar_arm(cs: ColumnStates, tr, va, head) -> tuple[np.ndarray, np.ndarray]:
    """The projection of the inner-train rows from shared sums: all rows minus the inner-val part."""
    A = cs.total - _moment_sums(cs.rows(va), cs.Y[va])
    proj = head()._fit_from_sums(A, [m.shape[1] for m in cs.mats], cs.classes)
    return proj.transform(cs.rows(tr)), proj.transform(cs.rows(va))


ARMS: dict[str, Callable] = {BASELINE: _frozen_arm, CANDIDATE: _lighttar_arm}


def prepare_column(H, y, classification: bool) -> ColumnStates:
    """The per-column work shared by all inner folds: float64 layers, float32 first layer, total sums."""
    raw = _split_layers(H, as_numpy=True)[0]
    mats = [np.asarray(m, dtype=np.float64) for m in raw]
    Y, classes = raw_target(y, classification)
    if not classification:
        Y = Y - Y.mean()  # the sums are shift-invariant in y; centering keeps them well conditioned
    return ColumnStates(mats, np.asarray(raw[0], dtype=np.float32), Y, classes, _moment_sums(mats, Y))


def fold_arms(cs: ColumnStates, tr, va, head: Callable) -> dict:
    """``{arm: (Z_inner_train, Z_inner_val)}`` of one column on one inner fold; ``head()`` makes an unfitted
    ``LightTARProjection``. The inner-train rows must be the complement of the inner-val rows."""
    return {arm: fn(cs, tr, va, head) for arm, fn in ARMS.items()}


def column_arms(H, y, splits, classification: bool, head: Callable) -> list[dict]:
    """``fold_arms`` of one column for every inner fold, sequentially."""
    if not splits:
        return []
    cs = prepare_column(H, y, classification)
    return [fold_arms(cs, tr, va, head) for tr, va in splits]

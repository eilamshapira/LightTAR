"""LightTAR and GatedLightTAR: the DataFrame steps (scikit-learn API)."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ._columns import column_texts, detect_text_columns
from ._gate import (
    BASELINE,
    CANDIDATE,
    FrozenPCA,
    GateResult,
    check_gate_params,
    inner_splits,
    make_gate_model,
    prepare_column,
    run_gate,
)
from .core import LightTARProjection, _is_tensor, is_classification_target
from .encoder import LayerEncoder
from .image import ImageLayerEncoder


def _folds(n: int, k: int, y, classification: bool, seed: int) -> np.ndarray:
    """Shuffled k-fold labels; stratified (round-robin within class) for classification."""
    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=int)
    if classification:
        _, inv = np.unique(np.asarray(y).astype(str), return_inverse=True)
        start = 0
        for c in rng.permutation(inv.max() + 1):
            idx = rng.permutation(np.flatnonzero(inv == c))
            fold[idx] = (start + np.arange(len(idx))) % k
            start += len(idx)
    else:
        fold[rng.permutation(n)] = np.arange(n) % k
    return fold


def _rows(H, mask: np.ndarray):
    """Boolean row selection for numpy arrays and torch tensors alike."""
    if _is_tensor(H):
        import torch

        return H[torch.from_numpy(np.flatnonzero(mask)).to(H.device)]
    return H[mask]


def _make_projection(n_anchor: int, n_supervised: int, m: float, classification: bool) -> LightTARProjection:
    """The per-column head of ``LightTAR``."""
    task = "classification" if classification else "regression"
    return LightTARProjection(n_anchor=n_anchor, n_supervised=n_supervised, m=m, task=task)


class LightTAR(BaseEstimator, TransformerMixin):
    """Replace the text columns of a DataFrame by LightTAR features from a frozen encoder.

    Each selected column becomes up to ``n_anchor + n_supervised`` float32 columns ``{col}__lt_{i}`` in place;
    the other columns pass through unchanged. Missing values are encoded as "Unknown Value", the task
    (classification or regression) is detected from y, and with several columns the projection of one
    column is fitted while the next is encoded.

    Parameters
    ----------
    columns : "auto" or list of str, default "auto"
        The columns to replace; "auto" uses ``detect_text_columns``. Image columns are passed
        explicitly, e.g. ``columns=detect_image_columns(df)`` with an ``ImageLayerEncoder``.
    encoder : str, LayerEncoder, ImageLayerEncoder, object or None, default "intfloat/e5-small-v2"
        A text model name, a configured encoder (any object with ``encode(values, column)`` returning
        ``(n, n_layers, dim)`` states), or ``None`` to pass ``layer_states`` to fit and transform.
    layers : sequence of int, float or str, default (1.0, 0.5)
        Encoder layers when ``encoder`` is a model name (see ``LayerEncoder``).
    n_anchor : int, default 30
        Unsupervised anchor dimensions per column.
    n_supervised : int, default 5
        Supervised dimensions per column.
    m : float, default 60.0
        Shrinkage toward the anchor (see ``LightTARProjection``).
    cv : int or None, default None
        None fits on the training rows and transforms them with that fit. With ``k``, ``fit_transform``
        returns out-of-fold features for the training rows (k folds, stratified for classification);
        ``transform`` always uses the fit on all training rows.
    device : str or None, default None
        Encoder device when ``encoder`` is a model name.
    batch_size : int, default 32
        Encoder batch size when ``encoder`` is a model name.
    random_state : int, default 0
        Seed of the ``cv`` folds.
    verbose : bool, default False
        Print a line per encoded column.
    """

    def __init__(
        self,
        columns: str | Sequence[str] = "auto",
        encoder: Any = "intfloat/e5-small-v2",
        layers: Sequence = (1.0, 0.5),
        n_anchor: int = 30,
        n_supervised: int = 5,
        m: float = 60.0,
        cv: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
        random_state: int = 0,
        verbose: bool = False,
    ):
        self.columns = columns
        self.encoder = encoder
        self.layers = layers
        self.n_anchor = n_anchor
        self.n_supervised = n_supervised
        self.m = m
        self.cv = cv
        self.device = device
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X, dtype=object)
        names = getattr(self, "feature_names_in_", None)
        cols = list(names) if names is not None else [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)

    def _get_encoder(self):
        if getattr(self, "_encoder", None) is None:
            if isinstance(self.encoder, str):
                self._encoder = LayerEncoder(
                    self.encoder, layers=self.layers, batch_size=self.batch_size, device=self.device
                )
            elif hasattr(self.encoder, "encode"):
                self._encoder = self.encoder
            else:
                raise ValueError("encoder=None: pass layer_states for every column")
        return self._encoder

    def _states(self, df, col, layer_states, prepared=None):
        """Encoder states of one column: from ``layer_states``, or encoded (kept on the GPU if the encoder
        runs there)."""
        if layer_states is not None and col in layer_states:
            H = layer_states[col]
            H = H if _is_tensor(H) else np.asarray(H)
            if len(H) != len(df):
                raise ValueError(f"layer_states[{col!r}] has {len(H)} rows, X has {len(df)}")
            return H
        if self.verbose:
            print(f"[lighttar] encoding column {col!r} ({len(df)} rows)", flush=True)
        enc = self._get_encoder()
        if isinstance(enc, ImageLayerEncoder):
            enc._ready()
            out = "torch" if str(enc.device_).startswith("cuda") else "numpy"
            return enc.encode(df[col].tolist(), column=str(col), output=out)
        texts = column_texts(df[col])
        if isinstance(enc, LayerEncoder):
            enc._ready()
            out = "torch" if str(enc.device_).startswith("cuda") else "numpy"
            return enc.encode(texts, column=str(col), output=out, prepared=prepared)
        return enc.encode(texts, column=str(col))

    def _iter_states(self, df, cols, layer_states):
        """Yield (col, states) in order; with a LayerEncoder, column j+1 is tokenized in a thread while
        column j runs on the device."""
        todo = [c for c in cols if layer_states is None or c not in layer_states]
        enc = self._get_encoder() if len(todo) > 1 else None
        if not isinstance(enc, LayerEncoder):
            for col in cols:
                yield col, self._states(df, col, layer_states)
            return
        enc._ready()
        with ThreadPoolExecutor(max_workers=1) as pool:
            futs = {}
            nxt = iter(todo)
            first = next(nxt, None)
            futs[first] = pool.submit(enc.prepare, column_texts(df[first]), str(first))
            for col in cols:
                if col not in futs:
                    yield col, self._states(df, col, layer_states)
                    continue
                c2 = next(nxt, None)
                if c2 is not None:
                    futs[c2] = pool.submit(enc.prepare, column_texts(df[c2]), str(c2))
                yield col, self._states(df, col, layer_states, prepared=futs.pop(col).result())

    def _assemble(self, df, blocks: Mapping[str, np.ndarray]) -> pd.DataFrame:
        parts = []
        for col in df.columns:
            if col in blocks:
                Z = blocks[col]
                parts.append(pd.DataFrame(Z, index=df.index, columns=self._col_names(col, Z.shape[1])))
            else:
                parts.append(df[[col]])
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    def _setup(self, X, y):
        df = self._frame(X)
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_features_in_ = df.shape[1]
        if isinstance(self.columns, str):
            if self.columns != "auto":
                raise ValueError("columns must be 'auto' or a list of column names")
            cols = detect_text_columns(df)
        else:
            cols = list(self.columns)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"columns not in X: {missing}")
        self.text_columns_ = cols
        y = np.asarray(y)
        if len(y) != len(df):
            raise ValueError(f"X has {len(df)} rows but y has {len(y)}")
        self.is_classification_ = is_classification_target(y)
        self._encoder = None
        return df, y

    def fit(self, X, y, layer_states: Mapping[str, Any] | None = None) -> LightTAR:
        """Fit one projection per selected column. ``layer_states`` maps a column to precomputed
        ``(n, n_layers, dim)`` encoder states, which skips encoding it."""
        self._fit(X, y, layer_states, return_train=False)
        return self

    def fit_transform(self, X, y=None, layer_states: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Fit and return the training features (out-of-fold with ``cv``)."""
        if y is None:
            raise ValueError("LightTAR is supervised: y is required")
        return self._fit(X, y, layer_states, return_train=True)

    def _fit(self, X, y, layer_states, return_train):
        df, y = self._setup(X, y)
        cls = self.is_classification_
        fold = _folds(len(df), int(self.cv), y, cls, self.random_state) if return_train and self.cv else None
        cols = list(self.text_columns_)
        with ThreadPoolExecutor(max_workers=1) as pool:  # one fit at a time; the main thread encodes
            futs = [
                (col, pool.submit(self._fit_column, H, y, fold, return_train))
                for col, H in self._iter_states(df, cols, layer_states)
            ]
            results = {col: f.result() for col, f in futs}
        self.models_ = {col: r[0] for col, r in results.items()}
        return self._assemble(df, {col: r[1] for col, r in results.items()}) if return_train else None

    def _head(self) -> LightTARProjection:
        return _make_projection(self.n_anchor, self.n_supervised, self.m, self.is_classification_)

    def _fit_column(self, H, y, fold, return_train) -> tuple:
        """The fit (and the training features) of one column."""
        model = self._head().fit(H, y)
        if not return_train:
            return model, None
        if fold is None:
            return model, model.transform(H)
        a = model.n_anchor_
        Z = np.zeros((len(y), a + model.n_supervised_), dtype=np.float32)
        for f in range(int(self.cv)):
            te = fold == f
            fm = self._head().fit(_rows(H, ~te), y[~te])
            Zf, af = fm.transform(_rows(H, te)), min(fm.n_anchor_, a)  # fewer rows: blocks may be narrower
            Z[np.ix_(te, range(af))] = Zf[:, :af]
            sf = min(fm.n_supervised_, model.n_supervised_)
            Z[np.ix_(te, range(a, a + sf))] = Zf[:, fm.n_anchor_ : fm.n_anchor_ + sf]
        return model, Z

    def _width(self, col) -> int:
        """Output columns of a replaced column."""
        model = getattr(self, "models_", {}).get(col)
        return model.components_.shape[1] if model is not None else self.n_anchor + self.n_supervised

    def _col_names(self, col, k: int) -> list[str]:
        return [f"{col}__lt_{i}" for i in range(k)]

    def transform(self, X, layer_states: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Replace the fitted columns by their features."""
        if not hasattr(self, "models_"):
            raise RuntimeError("LightTAR is not fitted; call fit first")
        df = self._frame(X)
        blocks = {
            col: self.models_[col].transform(H)
            for col, H in self._iter_states(df, list(self.text_columns_), layer_states)
        }
        return self._assemble(df, blocks)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = []
        for col in self.feature_names_in_:
            names += self._col_names(col, self._width(col)) if col in self.text_columns_ else [col]
        return np.asarray(names, dtype=object)

    def __getstate__(self):
        state = dict(super().__getstate__())
        state.pop("_encoder", None)  # weights are reloaded on the next transform
        return state


class GatedLightTAR(LightTAR):
    """LightTAR or the standard frozen pipeline, chosen once per fit by a majority vote over inner folds.

    The frozen pipeline replaces a column by a scikit-learn PCA-30 of the raw first layer (the encoder's
    last layer by default). On each inner fold both arms are fitted on inner-train and ``gate_model`` is
    scored on inner-val on each arm's table; LightTAR is kept iff it scores strictly higher on a strict
    majority of the valid folds (or no fold is valid, with a warning). The decision covers all columns and
    is ``gate_decision_`` (``"lighttar"`` or ``"frozen"``), next to ``gate_wins_``, ``gate_n_valid_``,
    ``gate_k_eff_``, ``gate_scores_`` (``(K, 2)``, frozen then LightTAR, NaN for an invalid fold) and
    ``gate_note_``. With ``"frozen"`` each column becomes ``{col}__frozen_0..29``, fitted on all training
    rows (also with ``cv``: the PCA does not see y); with ``"lighttar"`` the output is that of ``LightTAR``.
    The gate computes in float64 and shares the LightTAR moments across the inner folds.

    Parameters
    ----------
    gate_model : "lightgbm" or estimator, default "lightgbm"
        The proxy learner: a small fixed LightGBM (needs ``lighttar[gate]``), or any scikit-learn
        estimator, cloned for every fold and arm.
    gate_cv : int, default 5
        Inner folds (stratified for classification, seeded by ``random_state``).
    gate_scoring : callable or None, default None
        ``scoring(y_true, y_pred) -> float``, higher is better; ``y_pred`` is ``predict_proba`` for
        classification and ``predict`` for regression. None: AUC (binary, macro one-vs-rest for
        multiclass) or R².
    gate_features : {"all", "text"}, default "all"
        "all": the proxy also sees the other columns, first, encoded per inner fold on inner-train
        (numeric as float, others as categories with a missing-value category; LightGBM handles them
        natively, other estimators get the codes, -1 for unseen); "text": only the text blocks of each arm.
    n_jobs : int or None, default None
        CPU threads of the gate (None: all cores). Its folds and arms run in parallel; ``n_jobs=1`` runs
        everything sequentially on one thread (the paper's protocol); every value gives the same result.
    **LightTAR parameters
        ``columns``, ``encoder``, ``layers``, ``n_anchor``, ``n_supervised``, ``m``, ``cv``, ``device``,
        ``batch_size``, ``random_state``, ``verbose``, as in ``LightTAR``.
    """

    def __init__(
        self,
        columns: str | Sequence[str] = "auto",
        encoder: Any = "intfloat/e5-small-v2",
        layers: Sequence = (1.0, 0.5),
        n_anchor: int = 30,
        n_supervised: int = 5,
        m: float = 60.0,
        cv: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
        random_state: int = 0,
        verbose: bool = False,
        *,
        gate_model: Any = "lightgbm",
        gate_cv: int = 5,
        gate_scoring: Callable | None = None,
        gate_features: str = "all",
        n_jobs: int | None = None,
    ):
        lighttar = (columns, encoder, layers, n_anchor, n_supervised, m, cv, device, batch_size, random_state)
        super().__init__(*lighttar, verbose)
        self.gate_model = gate_model
        self.gate_cv = gate_cv
        self.gate_scoring = gate_scoring
        self.gate_features = gate_features
        self.n_jobs = n_jobs

    def _setup(self, X, y):
        check_gate_params(self.gate_features, self.gate_cv, self.n_jobs)
        df, y = super()._setup(X, y)
        self._gate_proto = make_gate_model(self.gate_model, self.is_classification_)
        self._gate_splits = inner_splits(y, int(self.gate_cv), self.is_classification_, self.random_state)
        return df, y

    def _fit(self, X, y, layer_states, return_train):
        df, y = self._setup(X, y)
        cols = list(self.text_columns_)
        splits, k_eff, note = self._gate_splits if cols else ([], 0, "no columns to replace")
        rest = [c for c in df.columns if c not in cols] if self.gate_features == "all" else []
        n_jobs = int(self.n_jobs or os.cpu_count() or 1)
        parallel = max(1, min(n_jobs, 2 * len(splits)))  # proxy fits run side by side, n_jobs // parallel threads each
        with ThreadPoolExecutor(max_workers=parallel) as pool:  # columns are prepared while the next is encoded
            states, preps = {}, {}
            for col, H in self._iter_states(df, cols, layer_states):
                states[col] = H
                preps[col] = pool.submit(prepare_column, H, y, self.is_classification_) if splits else None
            gate = (self._gate_proto, self.is_classification_, self.gate_scoring, self._head, k_eff, note)
            frame = df[[c for c in df.columns if c in cols or c in rest]]
            if cols:
                res = run_gate(pool, max(1, n_jobs // parallel), preps, frame, y, splits, *gate)
            else:  # nothing to replace: both arms are the same table, and ties go to the frozen arm
                res = GateResult(BASELINE, 0, 0, 0, note, np.empty((0, 2)))
        self.gate_decision_, self.gate_wins_, self.gate_n_valid_ = res.decision, res.wins, res.n_valid
        self.gate_k_eff_, self.gate_scores_, self.gate_note_ = res.k_eff, res.scores, res.note
        del self._gate_proto, self._gate_splits, preps
        if res.decision == CANDIDATE:
            cv = int(self.cv) if return_train and self.cv else 0
            fold = _folds(len(df), cv, y, self.is_classification_, self.random_state) if cv else None
            results = {c: self._fit_column(states[c], y, fold, return_train) for c in cols}
        else:
            fitted = {c: FrozenPCA().fit(states[c]) for c in cols}
            results = {c: (fz, fz.transform(states[c]) if return_train else None) for c, fz in fitted.items()}
        self.models_ = {c: r[0] for c, r in results.items()}
        return self._assemble(df, {c: r[1] for c, r in results.items()}) if return_train else None

    def _width(self, col) -> int:
        model = getattr(self, "models_", {}).get(col)
        return model.n_components_ if isinstance(model, FrozenPCA) else super()._width(col)

    def _col_names(self, col, k: int) -> list[str]:
        frozen = getattr(self, "gate_decision_", CANDIDATE) != CANDIDATE
        return [f"{col}__frozen_{i}" for i in range(k)] if frozen else super()._col_names(col, k)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_gate_proto", None)
        state.pop("_gate_splits", None)
        return state

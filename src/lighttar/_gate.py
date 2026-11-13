"""The machinery of ``GatedLightTAR``: inner splits, the two arms, the gate model, the proxy tables and the
majority vote."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import clone

from ._arms import (  # noqa: F401  (re-exported)
    ARMS,
    BASELINE,
    CANDIDATE,
    FROZEN_DIM,
    FROZEN_SEED,
    FrozenPCA,
    column_arms,
    fold_arms,
    prepare_column,
)
from ._columns import MISSING_TEXT, column_texts, is_numerical, is_text

MISSING = MISSING_TEXT  # the category of a missing or unseen value in a categorical proxy column
LIGHTGBM_PARAMS = dict(
    n_estimators=100,
    num_leaves=7,
    max_depth=3,
    learning_rate=0.1,
    min_child_samples=20,
    n_jobs=1,  # replaced per fit by the gate's thread budget
    random_state=0,
    verbosity=-1,
    force_col_wise=True,  # skips LightGBM's row/col-wise timing test; the same trees, any thread count
)
_NO_LIGHTGBM = "GatedLightTAR needs lightgbm: pip install 'lighttar[gate]' (or pass your own estimator as gate_model)"

Splits = list[tuple[np.ndarray, np.ndarray]]


@dataclass
class GateResult:
    """Outcome of one gate run; ``scores`` is ``(k_eff, 2)``, baseline then candidate, NaN for invalid folds."""

    decision: str
    wins: int
    n_valid: int
    k_eff: int
    note: str
    scores: np.ndarray


def check_gate_params(gate_features, gate_cv, n_jobs) -> None:
    """Validate the gate's own parameters of ``GatedLightTAR``."""
    if gate_features not in ("all", "text"):
        raise ValueError("gate_features must be 'all' or 'text'")
    if isinstance(gate_cv, bool) or not isinstance(gate_cv, Integral) or gate_cv < 2:
        raise ValueError("gate_cv must be an integer >= 2")
    if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1):
        raise ValueError("n_jobs must be None or an integer >= 1")


def inner_splits(y, k: int, classification: bool, random_state: int) -> tuple[Splits, int, str]:
    """Shuffled inner K-fold splits as positional ``(train, val)`` indices; returns ``(splits, k_eff, note)``.

    Classification uses ``StratifiedKFold``; rows of a class with a single row always stay on the
    train side. A binary minority smaller than ``k`` reduces ``k`` to the minority count, and so does the
    smallest class when no class has ``k`` rows (stratification would be impossible). Regression
    uses ``KFold`` (``k`` reduced to the number of rows if needed). When fewer than two folds are
    possible, ``splits`` is empty and ``k_eff`` is 0.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(y)
    pos = np.arange(n)
    if not classification:
        k_eff = min(int(k), n)
        if k_eff < 2:
            return [], 0, "too few rows"
        kf = KFold(n_splits=k_eff, shuffle=True, random_state=random_state)
        return list(kf.split(pos)), k_eff, "ok" if k_eff == k else f"gate_cv reduced to {k_eff} (too few rows)"
    ys = np.asarray(y).astype(str)
    vals, counts = np.unique(ys, return_counts=True)
    single = np.isin(ys, vals[counts < 2])
    elig, ye = pos[~single], ys[~single]
    ev_counts = np.unique(ye, return_counts=True)[1]
    if len(ev_counts) < 2:
        return [], 0, "single class"
    k_eff, notes = min(int(k), len(elig)), []
    if len(ev_counts) == 2 and ev_counts.min() < k_eff:
        k_eff = int(ev_counts.min())
        notes.append(f"binary minority {k_eff} < {k}: gate_cv reduced to {k_eff}")
    elif ev_counts.max() < k_eff:
        k_eff = int(ev_counts.min())
        notes.append(f"every class has fewer than {k} rows: gate_cv reduced to {k_eff}")
    if k_eff < 2:
        return [], 0, "too few rows"
    skf = StratifiedKFold(n_splits=k_eff, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():  # a rare class with fewer rows than folds is expected here
        warnings.simplefilter("ignore", UserWarning)
        raw = list(skf.split(elig, ye))
    splits = [(np.sort(np.concatenate([elig[tr], pos[single]])), np.sort(elig[va])) for tr, va in raw]
    if single.any():
        notes.append(f"{int(single.sum())} single-row-class rows kept on the train side")
    return splits, k_eff, "; ".join(notes) or "ok"


def _macro_auc(y_true, proba, classes) -> float:
    """Mean one-vs-rest AUC over the classes present in ``y_true`` (with both outcomes)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    aucs = []
    for j, c in enumerate(classes):
        pos = y_true == c
        if 0 < pos.sum() < len(pos):
            aucs.append(roc_auc_score(pos.astype(int), proba[:, j]))
    if not aucs:
        raise ValueError("no class with both outcomes in the validation rows")
    return float(np.mean(aucs))


def default_scorer(classification: bool, y) -> Callable:
    """The benchmark metric: AUC (binary: positive-class probability; multiclass: macro one-vs-rest),
    or R² for regression. Probability columns follow the sorted classes of ``y``."""
    from sklearn.metrics import r2_score

    if not classification:
        return lambda y_true, pred: float(r2_score(y_true, pred))
    classes = np.unique(np.asarray(y))
    if len(classes) == 2:

        def binary_auc(y_true, proba):
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score((np.asarray(y_true) == classes[1]).astype(int), np.asarray(proba)[:, 1]))

        return binary_auc
    return lambda y_true, proba: _macro_auc(y_true, np.asarray(proba), classes)


def make_gate_model(spec, classification: bool):
    """An unfitted gate model: ``"lightgbm"`` (LightGBM-small, fixed parameters) or a clone of the given
    scikit-learn estimator."""
    if isinstance(spec, str):
        if spec != "lightgbm":
            raise ValueError(f"gate_model must be 'lightgbm' or an estimator, got {spec!r}")
        try:
            import lightgbm
        except ImportError as err:
            raise ImportError(_NO_LIGHTGBM) from err
        cls = lightgbm.LGBMClassifier if classification else lightgbm.LGBMRegressor
        return cls(**LIGHTGBM_PARAMS)
    if not hasattr(spec, "fit"):
        raise ValueError(f"gate_model must be 'lightgbm' or an estimator with fit, got {type(spec).__name__}")
    return clone(spec)


def _as_datetime(s: pd.Series) -> pd.Series:
    """As the upstream date parsing: values through ``pd.to_datetime(errors="coerce")`` (dates outside the
    nanosecond range become missing), in local time."""
    s = pd.to_datetime(s.apply(lambda v: v.replace('"', "") if isinstance(v, str) else v), errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.where(s.between(pd.Timestamp.min, pd.Timestamp.max))


def _date_features(train: pd.Series, val: pd.Series) -> dict:
    """A datetime column as year, month, day (hour unless every train value is a date), seconds since the
    epoch and ISO weekday, float32 values as float64, NaN where missing (skrub's ``DatetimeEncoder`` with
    ``add_weekday=True``, as the upstream date preprocessing)."""
    a, b = _as_datetime(train), _as_datetime(val)
    known = a.dropna()
    feats = ["year", "month", "day"] + ([] if (known.dt.normalize() == known).all() else ["hour"])
    out = {}
    for f in [*feats, "total_seconds", "weekday"]:
        pair = []
        for s in (a, b):
            if f == "total_seconds":
                v = (s - pd.Timestamp(1970, 1, 1)) / pd.Timedelta("1s")
            elif f == "weekday":
                v = s.dt.day_of_week + 1
            else:
                v = getattr(s.dt, f)
            pair.append(v.astype("float32").astype("float64"))
        out[f"{train.name}_{f}"] = tuple(pair)
    return out


def encode_remainder(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The non-replaced columns, encoded on ``train_df`` only, as the upstream (MulTaBench) preprocessing:
    datetime columns are expanded by ``_date_features`` and placed last; then ``is_numerical`` columns become
    float64 (non-numeric values become NaN) and any other column a pandas ``category`` over its sorted string
    values in ``train_df`` plus ``MISSING``, which also takes missing and unseen values."""
    from pandas.api.types import is_datetime64_any_dtype

    cols, dates = {}, {}
    for col in train_df.columns:
        a, b = train_df[col], val_df[col]
        if is_datetime64_any_dtype(a.dtype):
            dates.update(_date_features(a, b))
        else:
            cols[col] = (a, b)
    tr, va = {}, {}
    for col, (a, b) in {**cols, **dates}.items():
        if is_numerical(a):
            tr[col], va[col] = (pd.to_numeric(v, errors="coerce").astype("float64") for v in (a, b))
        else:
            cats = sorted(set(column_texts(a)) | {MISSING})
            for v, out in ((a, tr), (b, va)):
                t = pd.Series(column_texts(v), index=v.index)
                out[col] = pd.Categorical(t.where(t.isin(cats), MISSING), categories=cats)
    return pd.DataFrame(tr, index=train_df.index), pd.DataFrame(va, index=val_df.index)


def fold_tables(arms: Mapping[str, Mapping], frame: pd.DataFrame, tr, va) -> dict:
    """``{arm: (X_inner_train, X_inner_val)}`` of one fold as DataFrames in the research gate's column order.
    A replaced column counts as text on this fold only if its inner-train values are text (``is_text``, not
    ``is_numerical``); otherwise it is encoded with the other columns. The other columns of ``frame``
    (encoded on inner-train, in frame order, dates last) come first, then the arm's block of every text column
    in sorted column-name order. ``arms[c]`` is ``fold_arms(...)`` of column ``c``. Columns are named
    ``f0, f1, ...`` (LightGBM rejects some characters in feature names)."""
    sub = frame.iloc[tr]
    order = sorted((c for c in arms if not is_numerical(sub[c]) and is_text(sub[c])), key=str)
    rest = [c for c in frame.columns if c not in order]
    extra = encode_remainder(sub[rest], frame.iloc[va][rest])
    fold = {}
    for arm in ARMS:
        sides = []
        for side in (0, 1):
            blk = [pd.DataFrame(np.asarray(arms[c][arm][side], dtype=np.float64)) for c in order]
            rem = [extra[side].reset_index(drop=True)] if extra[side].shape[1] else []
            X = pd.concat(rem + blk, axis=1) if rem + blk else pd.DataFrame(index=range(len(extra[side])))
            X.columns = [f"f{i}" for i in range(X.shape[1])]
            sides.append(X)
        fold[arm] = tuple(sides)
    return fold


def proxy_tables(blocks: Mapping[str, Sequence[Mapping]], frame: pd.DataFrame, splits: Splits) -> list[dict]:
    """``fold_tables`` for every fold; ``blocks[c][j]`` is ``column_arms(...)[j]`` of column ``c``."""
    return [fold_tables({c: b[j] for c, b in blocks.items()}, frame, tr, va) for j, (tr, va) in enumerate(splits)]


def _is_lightgbm(model) -> bool:
    """True for a LightGBM scikit-learn model (or a subclass); never imports lightgbm."""
    lgb = sys.modules.get("lightgbm")
    return lgb is not None and isinstance(model, lgb.LGBMModel)


def as_float(X: pd.DataFrame) -> np.ndarray:
    """A proxy table for estimators without categorical support: category columns as their codes."""
    cols = [X[c].cat.codes.astype("float64") if isinstance(X[c].dtype, pd.CategoricalDtype) else X[c] for c in X]
    return np.column_stack(cols).astype(np.float64) if cols else np.empty((len(X), 0))


def _predict(model, X, classification: bool, classes: np.ndarray):
    """``predict_proba`` with columns aligned to ``classes`` (classification), else ``predict``."""
    if not classification:
        return model.predict(X)
    P = np.asarray(model.predict_proba(X))
    seen = list(getattr(model, "classes_", classes))
    if len(seen) == len(classes) and np.array_equal(np.asarray(seen), classes):
        return P
    out = np.zeros((len(P), len(classes)))
    idx = {c: j for j, c in enumerate(classes.tolist())}
    for j, c in enumerate(seen):
        out[:, idx[c]] = P[:, j]
    return out


def _fit_score(model, threads: int, tables, y, tr, va, classification: bool, classes, score) -> float:
    """One proxy fit (a fresh clone with ``threads`` threads when it has ``n_jobs``) and its inner-val score.
    Estimators other than LightGBM get float tables, and joblib's thread backend (no worker processes)."""
    fitted = clone(model)
    if "n_jobs" in fitted.get_params(deep=False):
        fitted.set_params(n_jobs=threads)
    if _is_lightgbm(model):
        fitted.fit(*tables[:1], y[tr])
        return float(score(y[va], _predict(fitted, tables[1], classification, classes)))
    from joblib import parallel_config

    Xtr, Xva = as_float(tables[0]), as_float(tables[1])
    with parallel_config(backend="threading"):  # thread-local: an estimator's own joblib work stays in threads
        fitted.fit(Xtr, y[tr])
        return float(score(y[va], _predict(fitted, Xva, classification, classes)))


def run_gate(pool, threads: int, preps: Mapping, frame: pd.DataFrame, y, splits: Splits, model, classification: bool,
             scoring, head: Callable, k_eff: int = 0, note: str = "ok") -> GateResult:  # fmt: skip
    """Run the gate on ``pool`` and take the majority vote.

    ``preps[c]`` is a future of ``prepare_column`` for text column ``c``. Per fold, the arms of every
    column, then the fold's tables, then its two proxy fits are submitted as their inputs complete, so
    folds overlap; each proxy fit gets ``threads`` threads. A fold whose proxy fit
    or scorer raises is invalid. The candidate wins a fold when it scores strictly higher; it is chosen
    iff ``2 * wins > n_valid``, and also when no fold is valid.
    """
    y = np.asarray(y)
    classes = np.unique(y) if classification else None
    score = scoring or default_scorer(classification, y)
    scores = np.full((len(splits), 2), np.nan)
    errors: dict[int, list] = {}
    arms: dict[int, dict] = {j: {} for j in range(len(splits))}
    pending = {f: ("prep", c) for c, f in preps.items()} if splits else {}
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for f in sorted(done, key=lambda f: repr(pending[f])):  # a reproducible submission order
            kind, key = pending.pop(f)
            if kind == "fit":
                try:
                    scores[key] = f.result()
                except Exception as err:  # any failure of the gate model or the scorer invalidates the fold
                    errors.setdefault(key[0], []).append((key[1], f"{type(err).__name__}: {err}"))
                continue
            out = f.result()  # representation and table errors propagate
            if kind == "prep":
                for j, (tr, va) in enumerate(splits):
                    pending[pool.submit(fold_arms, out, tr, va, head)] = ("arm", (key, j))
            elif kind == "arm":
                (c, j), (tr, va) = key, splits[key[1]]
                arms[j][c] = out
                if len(arms[j]) == len(preps):
                    pending[pool.submit(fold_tables, arms.pop(j), frame, tr, va)] = ("table", j)
            else:
                tr, va = splits[key]
                for a, arm in enumerate(ARMS):
                    task = (model, threads, out[arm], y, tr, va, classification, classes, score)
                    pending[pool.submit(_fit_score, *task)] = ("fit", (key, a))
    return vote(scores, [min(errors[j])[1] for j in sorted(errors)], k_eff, note)


def vote(scores: np.ndarray, errors: list, k_eff: int, note: str) -> GateResult:
    """The majority vote over the valid folds (``errors``: the first error of each invalid fold)."""
    for j in range(len(scores)):
        if np.isnan(scores[j]).any():
            scores[j] = np.nan
    valid = ~np.isnan(scores).any(1)
    n_valid = int(valid.sum())
    wins = int((scores[valid, 1] > scores[valid, 0]).sum())
    notes = [] if note == "ok" else [note]
    if errors:
        notes.append(f"gate model failed on {len(errors)} fold{'s' if len(errors) > 1 else ''}")
        warnings.warn(f"GatedLightTAR: {notes[-1]} (first error: {errors[0]})", stacklevel=4)
    if n_valid == 0:
        warnings.warn(
            f"GatedLightTAR: the gate could not run ({'; '.join(notes) or 'no folds'}); keeping LightTAR", stacklevel=4
        )
    decision = CANDIDATE if n_valid == 0 or 2 * wins > n_valid else BASELINE
    return GateResult(decision, wins, n_valid, int(k_eff), "; ".join(notes) or "ok", scores)

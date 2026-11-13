"""Text-column detection and the cell texts handed to an encoder."""

from __future__ import annotations

from numbers import Number

import numpy as np
import pandas as pd

MIN_TEXT_UNIQUE = 100
MIN_TEXT_UNIQUE_RATIO = 0.8
MAX_NUMERIC_FOR_CATEGORICAL = 50
MISSING_TEXT = "Unknown Value"


def _valid(values) -> list:
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(v)
        elif v is None or _isna(v) or (isinstance(v, Number) and not np.isfinite(v)):
            continue
        else:
            out.append(v)
    return out


def _isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _is_numeric_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.isdigit()
    if isinstance(v, (int, float)):
        return True
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def is_numerical(s) -> bool:
    """Numerical column: it has valid values and either a numeric (or bool) dtype or more than 50 distinct
    valid values of which at most one is non-numeric."""
    from pandas.api.types import is_numeric_dtype

    if is_numeric_dtype(s.dtype):
        return bool(np.isfinite(s.to_numpy(dtype=float, na_value=np.nan)).any())
    vals = _valid(s.tolist())
    if not vals:
        return False
    uniq = set(vals)
    return len(uniq) > MAX_NUMERIC_FOR_CATEGORICAL and sum(not _is_numeric_value(v) for v in uniq) <= 1


def is_text(s) -> bool:
    """A non-numerical column is text if its values as strings (missing values as "Unknown Value") have at
    least 100 distinct values or a distinct ratio of at least 0.8; otherwise it is categorical."""
    vals = column_texts(s)
    n = len(set(vals))
    return bool(vals) and (n >= MIN_TEXT_UNIQUE or n / len(vals) >= MIN_TEXT_UNIQUE_RATIO)


def detect_text_columns(df) -> list:
    """The text columns of a DataFrame: neither datetime nor ``is_numerical``, and ``is_text``."""
    from pandas.api.types import is_datetime64_any_dtype

    return [
        c for c in df.columns if not is_datetime64_any_dtype(df[c].dtype) and not is_numerical(df[c]) and is_text(df[c])
    ]


def column_texts(s) -> list:
    """Values as strings, with missing values as "Unknown Value"."""
    return s.astype(object).fillna(MISSING_TEXT).astype(str).tolist()

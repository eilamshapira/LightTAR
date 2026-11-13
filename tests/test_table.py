import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from conftest import make_layers, targets
from lighttar import LightTAR, LightTARProjection, detect_text_columns
from lighttar.table import column_texts

N = 600


@pytest.fixture(scope="module")
def frame():
    X1, Z = make_layers(n=N, d=48, seed=1)
    X2, _ = make_layers(n=N, d=40, seed=2)
    y = targets(Z)["binary"][0]
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num": np.arange(N, dtype=float),
            "title": [f"title {i}" for i in range(N)],  # text (all distinct)
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], N)),  # categorical
            "desc": [f"description number {i % 300}" for i in range(N)],  # text (300 distinct >= 100)
            "flag": rng.integers(0, 2, N).astype(bool),
        }
    )
    return df, y, {"title": X1, "desc": X2}


def kw():
    return dict(encoder=None, n_anchor=6, n_supervised=2)


class FakeEncoder:
    """Deterministic stand-in for LayerEncoder: (n, 2, 16) states from a text hash and length."""

    def encode(self, texts, column=None):
        out = np.empty((len(texts), 2, 16), dtype=np.float32)
        for r, t in enumerate(texts):
            g = np.random.default_rng(abs(hash((column, t))) % (2**32))
            out[r] = g.standard_normal((2, 16))
            out[r, :, 0] += len(t) / 10.0
        return out


def test_auto_detection_rule():
    n = 400
    df = pd.DataFrame(
        {
            "ints": np.arange(n),
            "floats": np.linspace(0, 1, n),
            "numeric_strings": [str(i) for i in range(n)],  # mostly numerical: not text
            "numeric_with_sentinel": [str(i) if i % 7 else "?" for i in range(n)],
            "few_levels": ["x", "y"] * (n // 2),  # categorical
            "ratio_text": [f"w{i % 90}" for i in range(n)],  # 90 distinct / 400: categorical
            "many_levels": [f"w{i % 150}" for i in range(n)],  # 150 distinct: text
            "short_unique": [f"u{i}" if i < 30 else None for i in range(n)],  # 31/400 with "Unknown Value": categorical
            "all_missing": [None] * n,
            "dates": pd.date_range("2020-01-01", periods=n),
        }
    )
    assert detect_text_columns(df) == ["many_levels"]


def test_output_layout_and_dtypes(frame):
    df, y, states = frame
    lt = LightTAR(**kw())
    out = lt.fit_transform(df, y, layer_states=states)
    assert lt.text_columns_ == ["title", "desc"] and lt.is_classification_
    expect = ["num"] + [f"title__lt_{i}" for i in range(8)] + ["cat"] + [f"desc__lt_{i}" for i in range(8)] + ["flag"]
    assert list(out.columns) == expect == list(lt.get_feature_names_out())
    assert out["cat"].dtype == df["cat"].dtype and out["flag"].dtype == bool and out["num"].dtype == float
    assert all(out[f"title__lt_{i}"].dtype == np.float32 for i in range(8))
    for col in ("title", "desc"):  # the per-column fits, run in a background thread, equal direct fits
        direct = LightTARProjection(n_anchor=6, n_supervised=2).fit(states[col], y).transform(states[col])
        np.testing.assert_array_equal(out[[f"{col}__lt_{i}" for i in range(8)]].to_numpy(), direct)
    pd.testing.assert_frame_equal(out, lt.transform(df, layer_states=states))


def test_explicit_columns_and_errors(frame):
    df, y, states = frame
    out = LightTAR(columns=["desc"], **kw()).fit_transform(df, y, layer_states=states)
    assert "title" in out.columns and "desc__lt_0" in out.columns
    with pytest.raises(ValueError):
        LightTAR(columns=["nope"], **kw()).fit(df, y, layer_states=states)
    with pytest.raises(ValueError):
        LightTAR(columns="text", **kw()).fit(df, y, layer_states=states)
    with pytest.raises(ValueError):
        LightTAR(**kw()).fit(df, y)  # encoder=None without states
    with pytest.raises(ValueError):
        LightTAR(**kw()).fit(df, y[:-1], layer_states=states)
    with pytest.raises(ValueError):
        LightTAR(**kw()).fit_transform(df, layer_states=states)
    with pytest.raises(RuntimeError):
        LightTAR(**kw()).transform(df)


def test_regression_target(frame):
    df, _, states = frame
    y = np.linspace(0, 1, N) ** 2
    lt = LightTAR(**kw())
    out = lt.fit_transform(df, y, layer_states=states)
    assert not lt.is_classification_ and np.isfinite(out.filter(like="__lt_").to_numpy()).all()


def test_cross_fitting(frame):
    df, y, states = frame
    lt = LightTAR(cv=5, **kw())
    oof = lt.fit_transform(df, y, layer_states=states)
    full = LightTAR(**kw()).fit_transform(df, y, layer_states=states)
    cols = [f"title__lt_{i}" for i in range(8)]
    assert not np.allclose(oof[cols].to_numpy(), full[cols].to_numpy())
    pd.testing.assert_frame_equal(lt.transform(df, layer_states=states), full)  # transform = full fit
    pd.testing.assert_frame_equal(oof, LightTAR(cv=5, **kw()).fit_transform(df, y, layer_states=states))


def test_small_training_set_width():
    X, Z = make_layers(n=10, d=48, seed=3)
    y = targets(Z)["binary"][0]
    df = pd.DataFrame({"t": [f"text {i}" for i in range(10)]})
    lt = LightTAR(columns=["t"], encoder=None, cv=5)  # inner folds of 8 rows: narrower than the full fit
    oof = lt.fit_transform(df, y, layer_states={"t": X})
    width = lt.models_["t"].n_anchor_ + lt.models_["t"].n_supervised_
    assert lt.models_["t"].n_anchor_ == 9 and oof.shape == (10, width) and np.isfinite(oof.to_numpy()).all()
    assert list(oof.columns) == list(lt.get_feature_names_out())
    assert lt.transform(df, layer_states={"t": X}).shape == oof.shape


def test_clone_params_pickle(frame):
    df, y, states = frame
    lt = LightTAR(n_anchor=4, m=10.0, encoder=None)
    assert clone(lt).get_params() == lt.get_params()
    lt.fit(df, y, layer_states=states)
    lt2 = pickle.loads(pickle.dumps(lt))
    pd.testing.assert_frame_equal(lt.transform(df, layer_states=states), lt2.transform(df, layer_states=states))


def test_pipeline(frame):
    df, y, states = frame
    X = df[["num", "title", "desc"]]
    pipe = make_pipeline(LightTAR(encoder=FakeEncoder(), n_anchor=6, n_supervised=2), LogisticRegression(max_iter=3000))
    pipe.fit(X, y)
    assert pipe.named_steps["lighttar"].text_columns_ == ["title", "desc"]
    assert pipe.predict_proba(X).shape == (N, 2)
    pipe2 = make_pipeline(LightTAR(**kw()), LogisticRegression(max_iter=3000))
    pipe2.fit(X, y, lighttar__layer_states=states)  # fit params are routed
    assert clone(pipe).get_params()["lighttar__n_anchor"] == 6
    pickle.loads(pickle.dumps(pipe)).predict(X)


def test_column_transformer(frame):
    df, y, _ = frame

    def ct():
        lt = LightTAR(encoder=FakeEncoder(), n_anchor=6, n_supervised=2)
        return ColumnTransformer([("text", lt, ["title", "desc"])], remainder="passthrough")

    c = ct().set_output(transform="pandas")
    out = c.fit_transform(df, y)
    assert out.shape == (N, 16 + 3)
    assert "text__title__lt_0" in out.columns and "remainder__cat" in out.columns
    assert c.named_transformers_["text"].text_columns_ == ["title", "desc"]  # auto on the given columns
    arr = ct().fit_transform(df[["num", "title", "desc"]], y)  # default numpy output
    assert arr.shape == (N, 17)
    np.testing.assert_allclose(arr[:, :16].astype(float), out.iloc[:, :16].to_numpy(), atol=1e-6)


def test_column_texts_missing_values():
    s = pd.Series(["a", None, np.nan, 3, 2.5])
    assert column_texts(s) == ["a", "Unknown Value", "Unknown Value", "3", "2.5"]


@pytest.mark.parametrize("module,cls", [("tabicl", "TabICLClassifier"), ("tabpfn", "TabPFNClassifier")])
def test_pipeline_with_foundation_model(module, cls):
    mod = pytest.importorskip(module)
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 2, 400)
    df = pd.DataFrame(
        {
            "review": [("good " if v else "bad ") + f"item {i}" for i, v in enumerate(lab)],
            "price": rng.random(400),
            "brand": pd.Categorical(rng.choice(["a", "b"], 400)),
        }
    )
    pipe = make_pipeline(LightTAR(encoder=FakeEncoder(), n_anchor=8, n_supervised=2), getattr(mod, cls)())
    pipe.fit(df.iloc[:300], lab[:300])
    assert pipe.predict_proba(df.iloc[300:]).shape == (100, 2)

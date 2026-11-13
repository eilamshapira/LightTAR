"""PrecomputedEncoder: encoder states computed elsewhere, looked up by cell text."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from conftest import make_layers, targets
from lighttar import LightTAR, LightTARProjection, PrecomputedEncoder

N = 300


@pytest.fixture(scope="module")
def table():
    H, Z = make_layers(n=N, d=24, seed=7)
    y = targets(Z, 7)["binary"][0]
    texts = [f"review {i}" for i in range(N - 1)] + ["Unknown Value"]  # the last cell is missing
    df = pd.DataFrame({"review": texts[:-1] + [None], "price": np.arange(N, dtype=float)})
    return df, y, H, dict(zip(texts, H, strict=True))


def test_single_and_multi_layer_vectors():
    one = PrecomputedEncoder({"a": np.ones(4), "b": np.zeros(4)})
    out = one.encode(["b", "a", "b"], "col")
    assert out.shape == (3, 1, 4) and out.dtype == np.float32 and out[1].sum() == 4
    two = PrecomputedEncoder({"a": np.ones((2, 4)), "b": np.zeros((2, 4))})
    assert two.encode(["a", "b"], "col").shape == (2, 2, 4)


def test_per_column_tables():
    enc = PrecomputedEncoder({"x": {"a": np.ones(3)}, "y": {"a": np.zeros(3)}, 5: {"a": np.full(3, 2.0)}})
    assert enc.encode(["a"], "x").sum() == 3 and enc.encode(["a"], "y").sum() == 0
    assert enc.encode(["a"], "5").sum() == 6  # LightTAR passes column names as strings
    with pytest.raises(KeyError, match="'z'"):
        enc.encode(["a"], "z")


def test_miss_raises_with_the_text():
    enc = PrecomputedEncoder({"a": np.ones(3)})
    with pytest.raises(KeyError, match="'not there'") as err:
        enc.encode(["a", "not there"], "review")
    assert "'review'" in str(err.value)


class Constant:
    """A tiny fallback encoder: (n, 1, 3) states equal to the text length."""

    def __init__(self):
        self.seen = []

    def encode(self, texts, column):
        self.seen += list(texts)
        return np.array([[[len(t)] * 3] for t in texts], dtype=np.float32)


def test_fallback_encodes_misses_in_order():
    fb = Constant()
    enc = PrecomputedEncoder({"a": np.zeros(3)}, fallback=fb)
    out = enc.encode(["xyz", "a", "hello"], "c")
    assert fb.seen == ["xyz", "hello"]
    np.testing.assert_array_equal(out[:, 0, 0], [3, 0, 5])


def test_lighttar_matches_layer_states(table):
    df, y, H, vectors = table
    a = LightTAR(encoder=PrecomputedEncoder(vectors), n_anchor=6, n_supervised=2).fit_transform(df, y)
    b = LightTAR(encoder=None, n_anchor=6, n_supervised=2).fit_transform(df, y, layer_states={"review": H})
    pd.testing.assert_frame_equal(a, b)


def test_pipeline_fit_predict(table):
    df, y, _, vectors = table
    pipe = make_pipeline(
        LightTAR(encoder=PrecomputedEncoder(vectors), n_anchor=6, n_supervised=2), LogisticRegression(max_iter=3000)
    )
    pipe.fit(df.iloc[:200], y[:200])
    assert pipe.predict(df.iloc[200:]).shape == (100,)
    assert pipe.score(df.iloc[200:], y[200:]) > 0.7


def test_projection_on_one_layer(table):
    _, y, H, _ = table
    Z = LightTARProjection(n_anchor=6, n_supervised=2).fit(H[:, 0], y).transform(H[:, 0])
    assert Z.shape == (N, 8) and Z.dtype == np.float32
    np.testing.assert_array_equal(Z, LightTARProjection(n_anchor=6, n_supervised=2).fit_transform(H[:, :1], y))


def test_no_torch_import():
    code = (
        "import sys, numpy as np, pandas as pd; from lighttar import LightTAR, PrecomputedEncoder; "
        "rng = np.random.default_rng(0); t = [f'r{i}' for i in range(120)]; "
        "v = {s: rng.standard_normal((2, 8)) for s in t}; "
        "LightTAR(encoder=PrecomputedEncoder(v), n_anchor=4, n_supervised=1)"
        ".fit_transform(pd.DataFrame({'r': t}), rng.integers(0, 2, 120)); "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'PIL')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"

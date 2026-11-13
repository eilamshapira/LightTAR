"""Public surface: exports, version, import cost, scikit-learn behaviour, layer specs."""

import importlib.metadata as md
import pickle
import re
import subprocess
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import lighttar
from lighttar import LightTAR, LightTARProjection
from lighttar.encoder import resolve_layers


def test_public_names():
    assert sorted(lighttar.__all__) == sorted(
        [
            "LightTAR",
            "GatedLightTAR",
            "LightTARProjection",
            "LayerEncoder",
            "PrecomputedEncoder",
            "ImageLayerEncoder",
            "detect_text_columns",
            "detect_image_columns",
            "__version__",
        ]
    )


def test_version_is_pep440_and_matches_metadata():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", lighttar.__version__)
    try:
        installed = md.version("lighttar")
    except md.PackageNotFoundError:  # source tree on PYTHONPATH, not installed
        return
    if "site-packages" in lighttar.__file__:
        assert installed == lighttar.__version__


def test_import_pulls_no_heavy_modules():
    code = (
        "import sys, lighttar; "
        "print(','.join(m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'lightgbm', 'PIL')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"`import lighttar` imported: {out}"


def test_constructor_parameters():
    assert list(LightTAR().get_params()) == sorted(
        ["columns", "encoder", "layers", "n_anchor", "n_supervised", "m", "cv", "device", "batch_size"]
        + ["random_state", "verbose"]
    )
    assert LightTARProjection().get_params() == {"n_anchor": 30, "n_supervised": 5, "m": 60.0, "task": "auto"}


def test_get_set_params_and_clone():
    est = LightTARProjection(n_anchor=10, m=20.0, task="regression")
    p = est.get_params()
    c = clone(est)
    assert c.get_params() == p and c is not est
    est.set_params(n_supervised=3)
    assert est.n_supervised == 3


def test_pipeline_and_pickle(data):
    X, T = data
    y, _ = T["binary"]
    pipe = make_pipeline(LightTARProjection(), LogisticRegression(max_iter=500))
    pipe.fit(X[:1000], y[:1000])
    assert pipe.score(X[1000:], y[1000:]) > 0.7
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(pipe)).predict(X), pipe.predict(X))


def test_fit_transform_and_feature_names(data):
    X, T = data
    y, _ = T["regression"]
    est = LightTARProjection()
    np.testing.assert_array_equal(est.fit_transform(X, y), LightTARProjection().fit(X, y).transform(X))
    assert list(est.get_feature_names_out()[:2]) == ["lt0", "lt1"]


def test_relative_layers_e5_small_and_large():
    # hidden_states has L+1 entries (index 0 = embeddings); negative index = idx - (L+1)
    for L, expect in [(12, (-1, -7)), (24, (-1, -13))]:
        idx = resolve_layers((1.0, 0.5), L)
        assert tuple(i - (L + 1) for i in idx) == expect
        assert resolve_layers(("last", "middle"), L) == idx
        assert resolve_layers(list(expect), L) == idx


def test_layer_edge_cases():
    assert resolve_layers([0, 3, -1, -13], 12) == [0, 3, 12, 0]
    assert resolve_layers([0.0, "embeddings", "first"], 6) == [0, 0, 0]
    assert resolve_layers([0.25], 6) == [2]  # floor(1.5 + 0.5) = 2 (round half up)
    assert resolve_layers([1 / 3], 12) == [4]
    for bad in ([13], [-14], [1.5], ["deep"], [True]):
        with pytest.raises((ValueError, TypeError)):
            resolve_layers(bad, 12)

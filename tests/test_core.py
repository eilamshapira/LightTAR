import pickle

import numpy as np
import pytest

from _reference import ReferenceLightTAR
from conftest import align, make_layers, targets
from lighttar import LightTARProjection
from lighttar.core import (
    _REF_SEED,
    _host_moments,
    _moment_sums,
    _split_layers,
    _top_eigvec_lanczos,
    is_classification_target,
    target_matrix,
)

TASKS = ["regression", "binary", "multiclass"]


def _rel(R, P):
    return align(R, P) / np.abs(R).max()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("m", [60.0, 20.0, 0.0])
def test_matches_reference_up_to_sign(data, task, m):
    X, T = data
    y, is_cls = T[task]
    R = ReferenceLightTAR(y, is_cls, m=m).fit(X).transform(X)
    P = LightTARProjection(m=m).fit(X, y).transform(X)
    assert P.shape == (len(X), 35) and P.dtype == np.float32
    assert _rel(R, P) <= 1e-5


def test_reference_signs(data):
    """Every component has a positive inner product with the fixed reference vector."""
    X, T = data
    est = LightTARProjection().fit(X, T["binary"][0])
    d = est.layer_dims_[0]
    ref = np.random.default_rng(_REF_SEED).standard_normal(est.n_features_in_)
    assert (ref[:d] @ est.anchor_components_[:d] > 0).all()
    assert (ref @ est.x_weights_ > 0).all()


def test_out_of_sample(data):
    X, T = data
    y, _ = T["binary"]
    tr, te = slice(0, 1000), slice(1000, None)
    R = ReferenceLightTAR(y[tr], True).fit(X[tr]).transform(X[te])
    Z = LightTARProjection().fit(X[tr], y[tr]).transform(X[te])
    assert Z.dtype == np.float32 and Z.shape == (500, 35)
    assert _rel(R, Z) <= 1e-5


def test_input_formats_equivalent(data):
    X, T = data
    y, _ = T["multiclass"]
    a = LightTARProjection().fit(X, y).transform(X)
    b = LightTARProjection().fit([X[:, 0], X[:, 1]], y).transform([X[:, 0], X[:, 1]])
    np.testing.assert_array_equal(a, b)
    c = LightTARProjection().fit(X.astype(np.float64), y).transform(X.astype(np.float64))
    assert np.abs(a - c).max() <= 1e-5 * np.abs(a).max()


def test_single_layer_and_unequal_dims(data):
    X, T = data
    y, _ = T["regression"]
    assert LightTARProjection().fit(X[:, 0], y).transform(X[:, 0]).shape == (len(X), 35)
    L = [X[:, 0], X[:, 1, :40]]
    assert LightTARProjection(n_anchor=10, n_supervised=3).fit(L, y).transform(L).shape == (len(X), 13)


def test_constant_and_duplicate_columns():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((400, 2, 20))
    X[:, 0, 5] = 0.25  # constant column: left unscaled
    X[:, 1, 7] = X[:, 1, 6]  # duplicate column
    y = rng.integers(0, 2, 400)
    R = ReferenceLightTAR(y, True, k_anchor=8).fit(X).transform(X)
    P = LightTARProjection(n_anchor=8).fit(X, y).transform(X)
    assert np.isfinite(P).all() and _rel(R, P) <= 1e-5


def test_string_labels_equal_integer_labels(data):
    X, T = data
    y, _ = T["multiclass"]
    a = LightTARProjection().fit(X, y).transform(X)
    b = LightTARProjection().fit(X, np.array([f"c{v}" for v in y], dtype=object)).transform(X)
    np.testing.assert_array_equal(a, b)


def test_task_auto_rule():
    assert is_classification_target(np.array(["a", "b"]))
    assert is_classification_target(np.array([0, 1, 1, 0]))
    assert is_classification_target(np.array([0.0, 1.0, 2.0]))
    assert is_classification_target(np.array([True, False]))
    assert not is_classification_target(np.arange(100))  # many integer values
    assert not is_classification_target(np.array([0.5, 1.0, 2.0]))


def test_explicit_task(data):
    X, T = data
    y, _ = T["multiclass"]
    est = LightTARProjection(task="regression").fit(X, y)
    assert not est.is_classification_ and est.classes_ is None
    assert LightTARProjection().fit(X, y).classes_.tolist() == [0, 1, 2, 3]


def test_lanczos_top_eigvec():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 300))
    M = A @ A.T
    w, ok = _top_eigvec_lanczos(M.__matmul__, rng.standard_normal(300), tol=1e-14)
    v = np.linalg.eigh(M)[1][:, -1]
    assert ok and min(np.abs(w - v).max(), np.abs(w + v).max()) < 1e-9


def test_determinism_and_errors(data):
    X, T = data
    y, _ = T["binary"]
    np.testing.assert_array_equal(
        LightTARProjection().fit(X, y).transform(X), LightTARProjection().fit(X, y).transform(X)
    )
    with pytest.raises(ValueError):
        LightTARProjection(task="ranking").fit(X, y)
    with pytest.raises(ValueError):
        LightTARProjection(n_anchor=10_000).fit(X, y)
    with pytest.raises(ValueError):
        LightTARProjection().fit(X, y[:-1])
    with pytest.raises(ValueError):
        LightTARProjection().fit(X[0], y)
    with pytest.raises(RuntimeError):
        LightTARProjection().transform(X)
    est = LightTARProjection().fit(X, y)
    with pytest.raises(ValueError):
        est.transform(X[:, :1])


def test_pickle(data):
    X, T = data
    y, _ = T["regression"]
    est = LightTARProjection().fit(X, y)
    a = est.transform(X)
    est2 = pickle.loads(pickle.dumps(est))
    assert "_dev_cache" not in est.__getstate__()
    np.testing.assert_array_equal(est2.transform(X), a)


def test_torch_input_equals_numpy(data):
    torch = pytest.importorskip("torch")
    X, T = data
    y, _ = T["binary"]
    a = LightTARProjection().fit(X, y).transform(X)
    Xt = torch.from_numpy(X)
    est = LightTARProjection().fit(Xt, y)
    b = est.transform(Xt)
    assert isinstance(b, np.ndarray) and np.abs(a - b).max() <= 1e-5
    c = LightTARProjection().fit([Xt[:, 0], Xt[:, 1]], y).transform([Xt[:, 0], Xt[:, 1]])
    assert np.abs(a - c).max() <= 1e-5


def test_device_kernels_match_host(data):
    """The torch kernels (used for tensors on a GPU) agree with the host path; run here on CPU tensors."""
    torch = pytest.importorskip("torch")
    from lighttar.core import _torch_moments

    X, T = data
    y, is_cls = T["multiclass"]
    Y, _ = target_matrix(y, is_cls)
    mats, _ = _split_layers(X, as_numpy=True)
    host = _host_moments(mats, Y)
    dev = _torch_moments([torch.from_numpy(np.ascontiguousarray(m)) for m in mats], Y)
    for h, d in zip(host, dev, strict=True):
        assert np.abs(h - d).max() <= 1e-5 * max(1.0, np.abs(h).max())
    est = LightTARProjection().fit(X, y)
    Zt = est._transform_torch([torch.from_numpy(np.ascontiguousarray(m)) for m in mats])
    assert np.abs(Zt - est.transform(X)).max() <= 1e-5 * np.abs(Zt).max()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("m", [60.0, 0.0])
def test_default_path_on_200_rows_matches_dense_reference(task, m):
    """With n > n_anchor the rank cap and the degeneracy stop leave the closed form unchanged: the fit on
    200 rows equals the dense numpy reference (explicit standardization, eigh PCA anchor, deflated eigh
    steps of Sxy Sxy' + (m/n) Sxx) on the training and new rows, up to column signs."""
    X, Z = make_layers(n=260, seed=5)
    y, is_cls = targets(Z, seed=5)[task]
    est = LightTARProjection(m=m).fit(X[:200], y[:200])
    P = est.transform(X)
    R = ReferenceLightTAR(y[:200], is_cls, m=m).fit(X[:200]).transform(X)
    assert (est.n_anchor_, est.n_supervised_) == (30, 5) and P.shape == (260, 35)
    assert _rel(R, P) <= 1e-5


def test_ranks_capped_by_training_rows():
    X, Z = make_layers(n=40, seed=2)
    y = targets(Z, seed=2)["multiclass"][0]
    est = LightTARProjection().fit(X[:8], y[:8])
    out = est.transform(X)
    assert est.n_anchor_ == 7 and est.n_supervised_ <= 5 and out.shape == (40, 7 + est.n_supervised_)
    assert len(est.get_feature_names_out()) == out.shape[1]
    assert np.isfinite(out).all() and (out[:8].std(0) > 1e-6).all()  # no column constant on the train rows


@pytest.mark.parametrize("m", [60.0, 0.0])
@pytest.mark.parametrize("distinct", [1, 2])
def test_degenerate_column_fits_with_fewer_supervised_columns(m, distinct):
    X, _ = make_layers(n=20, seed=4)
    Xd = X[np.arange(8) % distinct]  # 8 rows with 1 or 2 distinct state vectors (identical texts)
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    est = LightTARProjection(m=m).fit(Xd, y)
    out = est.transform(X)
    assert est.n_anchor_ == 7 and est.n_supervised_ == distinct - 1
    assert out.shape == (20, 7 + est.n_supervised_) and np.isfinite(out).all()
    A = _moment_sums(_split_layers(Xd, as_numpy=True)[0], np.eye(2)[y])
    via_sums = LightTARProjection(m=m)._fit_from_sums(A, [96, 96], np.array([0, 1]))
    assert via_sums.n_supervised_ == est.n_supervised_

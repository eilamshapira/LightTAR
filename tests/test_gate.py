"""GatedLightTAR: the majority-vote switch between the frozen pipeline (raw last layer -> PCA-30) and LightTAR."""

import hashlib
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

from conftest import make_layers, targets
from lighttar import GatedLightTAR, LightTAR, LightTARProjection
from lighttar._gate import FROZEN_SEED, MISSING, as_float, column_arms, encode_remainder, fold_tables, inner_splits
from lighttar.core import _moment_sums, raw_target

N = 300


def hidden(n=N, d=64, seed=0, task="binary"):
    """Two layers whose top 30 principal directions are nuisance; y lives in a lower-variance direction,
    so the frozen PCA-30 misses it and the supervised LightTAR directions find it."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    F = rng.standard_normal((n, 30)) * np.linspace(3, 2, 30)
    s = rng.standard_normal(n)
    mats = [
        F @ Q[:, :30].T + 1.2 * s[:, None] * Q[:, 30 + j] + 0.05 * rng.standard_normal((n, d)) + 0.3 for j in (0, 1)
    ]
    t = s + 0.2 * rng.standard_normal(n)
    y = {"binary": (t > 0).astype(int), "regression": t, "multiclass": np.digitize(t, [-0.7, 0.7])}[task]
    return np.stack(mats, 1).astype(np.float32), y


def frozen_out(X):
    """The standard frozen pipeline on all rows: PCA-30 fitted and applied on the raw float32 last layer."""
    X = np.asarray(X, np.float32)
    return PCA(n_components=30, random_state=FROZEN_SEED).fit(X).transform(X)


def majority(g):
    s = g.gate_scores_[~np.isnan(g.gate_scores_).any(1)]
    return "lighttar" if len(s) == 0 or 2 * (s[:, 1] > s[:, 0]).sum() > len(s) else "frozen"


def logreg():
    return LogisticRegression(max_iter=2000)


@pytest.fixture(scope="module")
def informative():
    """Text states that carry the binary target off the top principal directions, plus numeric, bool and
    category columns."""
    H, y = hidden()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num": rng.standard_normal(N),
            "t": [f"x{i}" for i in range(N)],
            "flag": rng.integers(0, 2, N).astype(bool),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], N)),
        }
    )
    return df, y, {"t": H}


@pytest.fixture(scope="module")
def shuffled():
    """Labels shuffled at a fixed seed: the text states carry nothing about y, a numeric column does, and
    the supervised directions (many, unshrunk, on wide states) only fit noise."""
    n = 200
    H, Z = make_layers(n=n, d=512, seed=0)
    rng = np.random.default_rng(0)
    y = rng.permutation(targets(Z, 0)["binary"][0])
    df = pd.DataFrame({"t": [f"x{i}" for i in range(n)], "num": y + 0.8 * rng.standard_normal(n)})
    return df, y, {"t": H}


def kw(**extra):
    return dict(encoder=None, n_anchor=6, n_supervised=2, **extra)


def noisy_kw(**extra):
    return dict(encoder=None, n_anchor=4, n_supervised=8, m=0.0, **extra)


def lt_cols(col, k):
    return [f"{col}__lt_{i}" for i in range(k)]


def frozen_cols(col, k=30):
    return [f"{col}__frozen_{i}" for i in range(k)]


@pytest.mark.parametrize("model", ["lightgbm", "logreg"])
def test_informative_keeps_lighttar(informative, model):
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
    df, y, states = informative
    g = GatedLightTAR(gate_model=logreg() if model == "logreg" else model, **kw())
    out = g.fit_transform(df, y, layer_states=states)
    assert g.gate_decision_ == "lighttar" and g.gate_wins_ == g.gate_k_eff_ == g.gate_n_valid_ == 5
    assert g.gate_note_ == "ok" and g.gate_scores_.shape == (5, 2)
    assert (g.gate_scores_[:, 1] > g.gate_scores_[:, 0]).all()
    pd.testing.assert_frame_equal(out, LightTAR(**kw()).fit_transform(df, y, layer_states=states))
    pd.testing.assert_frame_equal(g.transform(df, layer_states=states), out)


@pytest.mark.parametrize("model", ["lightgbm", "logreg"])
def test_shuffled_labels_choose_frozen(shuffled, model):
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
    df, y, states = shuffled
    g = GatedLightTAR(gate_model=logreg() if model == "logreg" else model, **noisy_kw())
    out = g.fit_transform(df, y, layer_states=states)
    assert g.gate_decision_ == "frozen" and 2 * g.gate_wins_ <= g.gate_n_valid_ == 5 and majority(g) == "frozen"
    assert list(out.columns) == frozen_cols("t") + ["num"] == list(g.get_feature_names_out())
    assert out.shape[1] == 31 and not isinstance(g.models_["t"], LightTARProjection)  # no LightTAR head fitted
    np.testing.assert_array_equal(out[frozen_cols("t")].to_numpy(), frozen_out(states["t"][:, 0, :]))
    assert (out[frozen_cols("t")].dtypes == np.float32).all()
    pd.testing.assert_frame_equal(out[["num"]], df[["num"]])
    pd.testing.assert_frame_equal(g.transform(df, layer_states=states), out)


def test_estimator_gate_without_lightgbm(informative, monkeypatch):
    df, y, states = informative
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # `import lightgbm` now raises ImportError
    with pytest.raises(ImportError, match=r"pip install 'lighttar\[gate\]'"):
        GatedLightTAR(**kw()).fit(df, y, layer_states=states)
    assert GatedLightTAR(gate_model=logreg(), **kw()).fit(df, y, layer_states=states).gate_decision_ == "lighttar"
    y_reg = df["num"].to_numpy() + np.linspace(0, 1, N)
    g = GatedLightTAR(gate_model=Ridge(), **kw()).fit(df, y_reg, layer_states=states)
    assert not g.is_classification_ and g.gate_n_valid_ == 5 and g.gate_decision_ == majority(g)


def test_gate_cv_and_minority(informative):
    df, y, states = informative
    g = GatedLightTAR(gate_model=logreg(), gate_cv=3, **kw()).fit(df, y, layer_states=states)
    assert g.gate_k_eff_ == 3 and g.gate_scores_.shape == (3, 2) and g.gate_n_valid_ == 3
    y_rare = np.zeros(N, int)
    y_rare[:3] = 1
    g = GatedLightTAR(gate_model=logreg(), **kw()).fit(df, y_rare, layer_states=states)
    assert g.gate_k_eff_ == 3 and "reduced to 3" in g.gate_note_
    with pytest.raises(ValueError):
        GatedLightTAR(gate_cv=1, **kw()).fit(df, y, layer_states=states)
    with pytest.raises(ValueError):
        GatedLightTAR(gate_features="none", **kw()).fit(df, y, layer_states=states)


def test_infeasible_gate_keeps_lighttar_and_warns(informative):
    df, _, states = informative
    y_one = np.zeros(N, int)
    y_one[0] = 1  # one class with a single row: one usable class left
    with pytest.warns(UserWarning, match="could not run"):
        g = GatedLightTAR(gate_model=logreg(), **kw()).fit(df, y_one, layer_states=states)
    assert g.gate_decision_ == "lighttar" and g.gate_n_valid_ == 0 and g.gate_k_eff_ == 0
    assert g.gate_note_ == "single class" and g.gate_scores_.shape == (0, 2)
    splits, k_eff, note = inner_splits(np.array([0] * 20 + [1]), 5, True, 0)
    assert (splits, k_eff) == ([], 0) and note == "single class"
    splits, k_eff, note = inner_splits(np.array([0] * 20 + [1] * 10 + [2]), 5, True, 0)
    assert k_eff == 5 and all(30 in tr for tr, _ in splits) and "kept on the train side" in note


def test_frozen_arm_is_pca_of_raw_last_layer(informative):
    _, y, states = informative
    H = states["t"]
    splits = inner_splits(y, 5, True, 0)[0]
    arms = column_arms(H, y, splits, True, lambda: LightTARProjection(6, 2, task="classification"))
    last = H[:, 0, :]
    assert last.dtype == np.float32  # the upstream call: float32 states, no upcast
    for (tr, va), fold in zip(splits, arms, strict=True):
        p = PCA(n_components=30, random_state=FROZEN_SEED).fit(last[tr])
        Ztr, Zva = fold["frozen"]
        assert Ztr.shape == (len(tr), 30) and Ztr.dtype == np.float32
        np.testing.assert_array_equal(Ztr, p.transform(last[tr]))  # bit-identical to the upstream fit + transform
        np.testing.assert_array_equal(Zva, p.transform(last[va]))
        ft = PCA(n_components=30, random_state=FROZEN_SEED).fit_transform(last[tr])
        np.testing.assert_allclose(Ztr, ft, atol=1e-4)  # fit_transform differs only by float32 rounding, same signs


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_shared_moments_match_direct_fit(task):
    H, y = hidden(task=task, seed=2)
    y = y + 1000.0 if task == "regression" else y  # an offset target: the sums stay well conditioned
    cls = task != "regression"
    kind = "classification" if cls else "regression"
    splits = inner_splits(y, 5, cls, 0)[0]
    arms = column_arms(H, y, splits, cls, lambda: LightTARProjection(6, 2, task=kind))
    H64 = H.astype(np.float64)
    for (tr, va), fold in zip(splits, arms, strict=True):
        direct = LightTARProjection(6, 2, task=kind).fit(H64[tr], y[tr])
        np.testing.assert_allclose(fold["lighttar"][0], direct.transform(H64[tr]), atol=1e-6)
        np.testing.assert_allclose(fold["lighttar"][1], direct.transform(H64[va]), atol=1e-6)


def test_fit_from_sums_drops_absent_class():
    H, y = hidden(task="multiclass", seed=4)
    rows = y != 1
    Y, classes = raw_target(y, True)
    mats = [H[rows, j, :].astype(np.float64) for j in (0, 1)]
    sums = LightTARProjection(6, 2)._fit_from_sums(_moment_sums(mats, Y[rows]), [64, 64], classes)
    direct = LightTARProjection(6, 2, task="classification").fit(mats, y[rows])
    assert list(sums.classes_) == list(direct.classes_) == [0, 2]
    np.testing.assert_allclose(sums.transform(mats), direct.transform(mats), atol=1e-6)


@pytest.mark.parametrize("data,gate_cv", [("informative", 5), ("informative", 3), ("shuffled", 5), ("shuffled", 4)])
def test_decision_follows_majority(request, data, gate_cv):
    df, y, states = request.getfixturevalue(data)
    g = GatedLightTAR(gate_model=logreg(), gate_cv=gate_cv, **kw()).fit(df, y, layer_states=states)
    s = g.gate_scores_
    assert g.gate_wins_ == int((s[:, 1] > s[:, 0]).sum()) and g.gate_decision_ == majority(g)


class FailOnCall(BaseEstimator, ClassifierMixin):
    """Logistic regression that raises on the ``fail_at``-th fit across all clones."""

    calls = 0

    def __init__(self, fail_at=3):
        self.fail_at = fail_at

    def fit(self, X, y):
        FailOnCall.calls += 1
        if FailOnCall.calls == self.fail_at:
            raise RuntimeError("boom")
        self.model_ = LogisticRegression(max_iter=2000).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def test_failing_fold_is_invalid_only(informative):
    df, y, states = informative
    FailOnCall.calls = 0
    with pytest.warns(UserWarning, match="failed on 1 fold"):
        g = GatedLightTAR(gate_model=FailOnCall(fail_at=3), n_jobs=1, **kw()).fit(df, y, layer_states=states)
    assert g.gate_n_valid_ == 4 and np.isnan(g.gate_scores_[1]).all() and not np.isnan(g.gate_scores_[[0, 2]]).any()
    assert g.gate_note_ == "gate model failed on 1 fold" and g.gate_decision_ == "lighttar"


def test_cv_composes_with_gate(informative, shuffled):
    df, y, states = informative
    g = GatedLightTAR(gate_model=logreg(), cv=2, **kw())
    oof = g.fit_transform(df, y, layer_states=states)
    pd.testing.assert_frame_equal(oof, LightTAR(cv=2, **kw()).fit_transform(df, y, layer_states=states))
    df, y, states = shuffled
    g = GatedLightTAR(gate_model=logreg(), cv=2, **noisy_kw())
    oof = g.fit_transform(df, y, layer_states=states)
    ref = LightTAR(cv=2, **noisy_kw()).fit_transform(df, y, layer_states=states)
    assert g.gate_decision_ == "frozen"
    full = frozen_out(states["t"][:, 0, :])
    np.testing.assert_array_equal(oof[frozen_cols("t")].to_numpy(), full)  # the PCA does not see y
    assert list(oof.columns) == frozen_cols("t") + ["num"] and ref.shape[1] == 13


def test_pickle_keeps_decision(shuffled):
    df, y, states = shuffled
    g = GatedLightTAR(gate_model=logreg(), **noisy_kw()).fit(df, y, layer_states=states)
    g2 = pickle.loads(pickle.dumps(g))
    assert g2.gate_decision_ == "frozen" and g2.gate_wins_ == g.gate_wins_
    np.testing.assert_array_equal(g2.gate_scores_, g.gate_scores_)
    pd.testing.assert_frame_equal(
        g2.transform(df, layer_states=states), g.transform(df, layer_states=states), atol=np.finfo(np.float32).eps
    )


class Recorder(LogisticRegression):
    """Logistic regression that records the width of every table it is fitted on."""

    widths: list = []
    tables: list = []

    def fit(self, X, y, sample_weight=None):
        Recorder.widths.append(X.shape[1])
        Recorder.tables.append(X)
        return super().fit(X, y, sample_weight)


@pytest.mark.parametrize("features,extra", [("all", 3), ("text", 0)])
def test_gate_features(informative, features, extra):
    df, y, states = informative
    Recorder.widths = []
    GatedLightTAR(gate_model=Recorder(max_iter=2000), gate_features=features, **kw()).fit(df, y, layer_states=states)
    assert sorted(Recorder.widths) == [8 + extra] * 5 + [30 + extra] * 5  # frozen PCA-30 and LightTAR, per fold


def test_encode_remainder():
    """The upstream (MulTaBench) preprocessing of the non-replaced columns, fitted on inner-train."""
    tr = pd.DataFrame(
        {
            "num": [1.5, np.nan, 3.0],
            "flag": [True, False, True],  # bool: numerical
            "cat": pd.Categorical(["b", "a", None], categories=["a", "b", "c"]),
            "obj": ["y", "x", "y"],
            "when": pd.to_datetime(["2020-01-02", None, "2021-03-04"]),  # dates: expanded, placed last
        }
    )
    va = pd.DataFrame(
        {
            "num": [2.0, 4.0],
            "flag": [False, True],
            "cat": pd.Categorical(["c", "b"]),
            "obj": ["z", None],
            "when": pd.to_datetime(["2022-05-06", "2020-01-02"]),
        }
    )
    A, B = encode_remainder(tr, va)
    dates = ["when_year", "when_month", "when_day", "when_total_seconds", "when_weekday"]  # no hour: all dates
    assert list(A.columns) == ["num", "flag", "cat", "obj", *dates] and list(B.columns) == list(A.columns)
    assert A["num"].dtype == np.float64 and np.isnan(A["num"][1]) and A["flag"].tolist() == [1.0, 0.0, 1.0]
    assert all(isinstance(A[c].dtype, pd.CategoricalDtype) for c in ("cat", "obj"))
    assert list(A["cat"].cat.categories) == [MISSING, "a", "b"]  # inner-train values + the missing label, sorted
    assert B["obj"].tolist() == [MISSING, MISSING]  # unseen and missing values both get the missing label
    np.testing.assert_array_equal(A["cat"].cat.codes, [2, 1, 0])
    np.testing.assert_array_equal(B["cat"].cat.codes, [0, 2])
    assert A["when_year"].tolist()[::2] == [2020.0, 2021.0] and np.isnan(A["when_year"][1])
    assert B["when_weekday"].tolist() == [5.0, 4.0]  # ISO weekday: 2022-05-06 is a Friday
    assert B["when_total_seconds"][1] == np.float64(np.float32(1577923200.0))
    far = pd.DataFrame({"d": np.array(["2020-01-02", "2999-12-31"], dtype="datetime64[us]")})
    A, _ = encode_remainder(far, far)  # outside the nanosecond range: missing, as the upstream date parsing
    assert A["d_year"].tolist()[0] == 2020.0 and np.isnan(A["d_year"][1])
    A, _ = encode_remainder(far.iloc[1:], far.iloc[1:])  # no date left: the features are a missing category
    assert isinstance(A["d_year"].dtype, pd.CategoricalDtype) and A["d_year"].tolist() == [MISSING]
    numeric_strings = pd.DataFrame({"code": [str(i) for i in range(60)] + ["?"]})  # > 50 distinct, 1 non-numeric
    A, _ = encode_remainder(numeric_strings, numeric_strings)
    assert A["code"].dtype == np.float64 and np.isnan(A["code"].iloc[-1])
    np.testing.assert_array_equal(as_float(B[["cat"]]), [[0], [2]])
    A, B = encode_remainder(tr[[]], va[[]])
    assert A.shape == (3, 0) and B.shape == (2, 0) and as_float(A).shape == (3, 0)


def test_fold_tables_demotes_a_column_that_is_not_text_on_inner_train():
    """As the research gate, a replaced column whose inner-train values are not text is a categorical column of
    that fold's table (in its frame position), not a text block."""
    n = 40
    frame = pd.DataFrame({"t": [f"w{i % 30}" for i in range(n)], "x": np.arange(n, dtype=float)})
    arms = {"t": {arm: (np.zeros((30, 2)), np.zeros((10, 2))) for arm in ("frozen", "lighttar")}}
    tr, va = np.arange(30), np.arange(30, 40)
    T = fold_tables(arms, frame, tr, va)["lighttar"][0]  # 30 distinct of 30 rows: text, block last
    assert T.shape == (30, 3) and T.dtypes.tolist() == [np.float64] * 3
    many = pd.DataFrame({"t": ["a", "b"] * 20, "x": np.arange(n, dtype=float)})  # 2 distinct: categorical
    T = fold_tables(arms, many, np.arange(30), np.arange(30, 40))["lighttar"][0]
    assert T.shape == (30, 2) and isinstance(T["f0"].dtype, pd.CategoricalDtype)


def test_table_order_remainder_first_text_sorted(informative):
    """The research gate's order: the other columns in frame order, then the text blocks by sorted name."""
    df, y, states = informative
    df = df.rename(columns={"t": "z_text"}).assign(a_text=df["t"])
    st = {"z_text": states["t"], "a_text": states["t"][:, :, ::-1].copy()}
    Recorder.tables = []
    g = GatedLightTAR(gate_model=Recorder(max_iter=2000), **kw()).fit(df, y, layer_states=st)
    assert g.text_columns_ == ["z_text", "a_text"]
    splits = inner_splits(y, 5, True, 0)[0]
    tr = splits[0][0]
    X = next(
        T
        for T in Recorder.tables
        if len(T) == len(tr) and T.shape[1] == 19 and (T[:, 0] == df["num"].to_numpy()[tr]).all()
    )
    head = g._head
    blocks = [column_arms(st[c], y, splits, True, head)[0]["lighttar"][0] for c in ("a_text", "z_text")]
    np.testing.assert_array_equal(X[:, 0], df["num"].to_numpy()[tr])
    np.testing.assert_array_equal(X[:, 1], df["flag"].to_numpy()[tr].astype(float))  # categories "False" < "True"
    np.testing.assert_array_equal(X[:, 3:], np.hstack(blocks))


def test_gate_scoring_callable(informative):
    df, y, states = informative
    seen = []

    def neg_log_loss(y_true, proba):
        seen.append(np.shape(proba))
        return -log_loss(y_true, proba)

    g = GatedLightTAR(gate_model=logreg(), gate_scoring=neg_log_loss, **kw()).fit(df, y, layer_states=states)
    assert g.gate_decision_ == "lighttar" and seen[0] == (N // 5, 2) and len(seen) == 10
    g = GatedLightTAR(gate_model=logreg(), gate_scoring=lambda yt, p: 0.5, **kw()).fit(df, y, layer_states=states)
    assert g.gate_wins_ == 0 and g.gate_decision_ == "frozen"  # ties are not wins


@pytest.mark.parametrize("task", ["regression", "multiclass"])
@pytest.mark.parametrize("model", ["lightgbm", "sklearn"])
def test_regression_and_multiclass(task, model):
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
        gate_model = "lightgbm"
    else:
        gate_model = Ridge() if task == "regression" else logreg()
    H, y = hidden(task=task)
    df = pd.DataFrame({"t": [f"x{i}" for i in range(N)]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no gate warning on the clean path
        g = GatedLightTAR(gate_model=gate_model, **kw()).fit(df, y, layer_states={"t": H})
    assert g.is_classification_ == (task == "multiclass")
    assert g.gate_decision_ == "lighttar" == majority(g) and g.gate_n_valid_ == 5 and g.gate_wins_ >= 4


def test_clone_and_params():
    g = GatedLightTAR(n_anchor=4, gate_cv=3, gate_model=LogisticRegression(C=2.0), gate_features="text")
    params = g.get_params(deep=False)
    assert set(params) == set(LightTAR().get_params()) | {
        "gate_model",
        "gate_cv",
        "gate_scoring",
        "gate_features",
        "n_jobs",
    }
    c = clone(g)
    cp = c.get_params(deep=False)
    assert c.gate_model is not g.gate_model and c.gate_model.get_params() == g.gate_model.get_params()
    assert {k: v for k, v in cp.items() if k != "gate_model"} == {k: v for k, v in params.items() if k != "gate_model"}
    assert GatedLightTAR().get_params()["gate_model"] == "lightgbm" and GatedLightTAR().n_anchor == 30


def test_small_multiclass_reduces_gate_cv():
    H, _ = hidden(n=12)
    y = np.repeat(np.arange(4), 3)  # 4 classes of 3 rows: no class fills 5 stratified folds
    df = pd.DataFrame({"t": [f"x{i}" for i in range(12)]})
    g = GatedLightTAR(columns=["t"], gate_model=logreg(), gate_cv=5, **kw()).fit(df, y, layer_states={"t": H})
    assert g.gate_k_eff_ == 3 and "reduced to 3" in g.gate_note_ and g.gate_scores_.shape == (3, 2)
    assert inner_splits(np.repeat(np.arange(3), 2), 5, True, 0)[1] == 2


def _splits_hash(splits):
    raw = b"".join(np.concatenate([tr, va]).astype(np.int64).tobytes() for tr, va in splits)
    return hashlib.sha256(raw).hexdigest()


def test_feasible_inner_splits_unchanged():
    """Fold indices on 200-row tables where K=5 is feasible, as computed before the multiclass K reduction."""
    y3 = np.repeat([0, 1, 2], [70, 70, 60])
    np.random.default_rng(7).shuffle(y3)
    rare = np.repeat([0, 1, 2, 3], [100, 96, 3, 1])
    np.random.default_rng(8).shuffle(rare)
    expected = {
        (0, "3class"): "639795907e0fb1fc1a6f7b02f5a898d5d8bb1e5d6122f9995eb2b8a5fcf2ebee",
        (3, "3class"): "4fb0ae4e5f9925e0a21789e949c792d055237bd5407967c1233774364ea54069",
        (0, "rare"): "f92a90b3935c923a9a721b8f76825a9aed0a2aba28836ffaf66557158c5abc3d",
        (3, "rare"): "d6d2aab3f537cc0deed0a6723aeb817f8c3442596fb72d443bcfebda1de9f005",
    }
    for (seed, name), digest in expected.items():
        splits, k_eff, _ = inner_splits(y3 if name == "3class" else rare, 5, True, seed)
        assert k_eff == 5 and _splits_hash(splits) == digest

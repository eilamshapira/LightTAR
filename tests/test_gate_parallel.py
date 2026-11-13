"""GatedLightTAR under parallelism: n_jobs never changes the result, and what each proxy model receives."""

import pickle
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from lighttar import GatedLightTAR
from lighttar._gate import inner_splits
from test_gate import Recorder, hidden, informative, kw, shuffled  # noqa: F401  (fixtures)

N = 600


@pytest.fixture(scope="module")
def mixed():
    """Two text columns, a numeric column carrying part of y, a category with missing values, a bool."""
    H, y = hidden(n=N, seed=1)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "t": [f"x{i}" for i in range(N)],
            "num": y + rng.standard_normal(N),
            "c": rng.choice(["a", "b", None], N),
            "flag": rng.integers(0, 2, N).astype(bool),
        }
    ).assign(t2=lambda d: d["t"])
    return df, y, {"t": H, "t2": H[:, ::-1].copy()}


def fit(model, n_jobs, data, **extra):
    df, y, states = data
    g = GatedLightTAR(encoder=None, gate_model=model, n_jobs=n_jobs, **extra)
    t0 = time.perf_counter()
    out = g.fit_transform(df, y, layer_states=states)
    return g, out, time.perf_counter() - t0


@pytest.mark.parametrize("model", ["lightgbm", "logreg"])
def test_n_jobs_is_bit_identical(mixed, shuffled, model):  # noqa: F811
    if model == "lightgbm":
        pytest.importorskip("lightgbm")
    gm = model if model == "lightgbm" else LogisticRegression(max_iter=2000)
    for data, extra in ((mixed, {}), (shuffled, dict(n_anchor=4, n_supervised=8, m=0.0))):
        runs = {nj: fit(gm, nj, data, **extra) for nj in (1, 4, None)}
        g1, out1, _ = runs[1]
        for g, out, _ in runs.values():
            np.testing.assert_array_equal(g.gate_scores_, g1.gate_scores_)
            assert (g.gate_wins_, g.gate_decision_, g.gate_n_valid_) == (g1.gate_wins_, g1.gate_decision_, 5)
            pd.testing.assert_frame_equal(out, out1)
    assert runs[1][0].gate_decision_ == "frozen"  # the shuffled data: both decisions are covered


def test_lightgbm_wall_time(mixed, capsys):
    pytest.importorskip("lightgbm")
    t = {nj: min(fit("lightgbm", nj, mixed)[2] for _ in range(2)) for nj in (1, 4)}
    with capsys.disabled():
        print(f"\n[gate wall time, lightgbm, n={N}, 2 text columns] n_jobs=1: {t[1]:.2f} s, n_jobs=4: {t[4]:.2f} s")


class FailOnFold(BaseEstimator, ClassifierMixin):
    """Logistic regression that raises when ``marker`` is missing from the first column of its training
    table, i.e. on the one inner fold that holds the marked row out."""

    def __init__(self, marker=0.0):
        self.marker = marker

    def fit(self, X, y):
        if self.marker not in np.asarray(X)[:, 0]:
            raise RuntimeError("boom")
        self.model_ = LogisticRegression(max_iter=2000).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_failing_fold_under_parallelism(informative, n_jobs):  # noqa: F811
    df, y, states = informative
    marker = df["num"].to_numpy()[inner_splits(y, 5, True, 0)[0][2][1][0]]  # a row held out by fold 2 only
    with pytest.warns(UserWarning, match="failed on 1 fold"):
        g = GatedLightTAR(gate_model=FailOnFold(marker), n_jobs=n_jobs, **kw()).fit(df, y, layer_states=states)
    assert (
        g.gate_n_valid_ == 4 and np.isnan(g.gate_scores_[2]).all() and not np.isnan(g.gate_scores_[[0, 1, 3, 4]]).any()
    )
    assert g.gate_note_ == "gate model failed on 1 fold"


def test_clone_and_pickle_keep_n_jobs(informative):  # noqa: F811
    df, y, states = informative
    g = GatedLightTAR(gate_model=LogisticRegression(max_iter=2000), n_jobs=3, **kw())
    assert clone(g).n_jobs == 3 and clone(g).get_params()["n_jobs"] == 3
    g.fit(df, y, layer_states=states)
    g2 = pickle.loads(pickle.dumps(g))
    assert g2.n_jobs == 3 and g2.gate_decision_ == g.gate_decision_
    pd.testing.assert_frame_equal(g2.transform(df, layer_states=states), g.transform(df, layer_states=states))
    for bad in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="n_jobs"):
            GatedLightTAR(n_jobs=bad, **kw()).fit(df, y, layer_states=states)


def test_lightgbm_gets_categories_sklearn_gets_floats(informative):  # noqa: F811
    lgb = pytest.importorskip("lightgbm")
    df, y, states = informative

    class Rec(lgb.LGBMClassifier):
        seen = []

        def fit(self, X, y, **kw):
            Rec.seen.append(X)
            return super().fit(X, y, **kw)

    GatedLightTAR(gate_model=Rec(n_estimators=5, verbosity=-1), **kw()).fit(df, y, layer_states=states)
    assert sorted(X.shape[1] for X in Rec.seen) == [3 + 8] * 5 + [3 + 30] * 5  # num, flag, cat + each arm
    for X in Rec.seen:
        assert isinstance(X, pd.DataFrame) and list(X.columns[:3]) == ["f0", "f1", "f2"]
        assert X["f0"].dtype == X["f1"].dtype == np.float64 and isinstance(
            X["f2"].dtype, pd.CategoricalDtype
        )  # bool: numerical
        assert (X.dtypes.iloc[3:] == np.float64).all()
    Recorder.tables = []
    GatedLightTAR(gate_model=Recorder(max_iter=2000), **kw()).fit(df, y, layer_states=states)
    assert all(isinstance(T, np.ndarray) and T.dtype == np.float64 for T in Recorder.tables)
    assert set(np.unique(Recorder.tables[0][:, 2])) <= {1.0, 2.0, 3.0}  # "cat" as codes (0 = MISSING)

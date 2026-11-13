"""The closed-form LightTAR projection on arrays of encoder states."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

Layers = Any  # an (n, d) or (n, L, d) array or tensor, or a list of (n, d_j) arrays
_EPS_SD = 1e-9  # columns with std below this are left unscaled
_EPS_TT = 1e-12  # added to t't before dividing
_BLOCK = 16384  # rows per block of the host Gram pass (bounds the float64 buffer)
_BLOCK_DEV = 2048  # rows per float32 GEMM on a torch device (bounds float32 accumulation error)
_LANCZOS_TOL = 1e-10  # residual tolerance of the supervised-step Lanczos (relative to the top eigenvalue)
_EPS_RANK = 1e-10  # a supervised direction with t't <= this * n * trace(G) has no variance left
_REF_SEED = 20261023  # seed of the fixed reference vector that sets eigenvector signs


def _is_tensor(a) -> bool:
    return type(a).__module__.split(".")[0] == "torch"


def _on_device(X) -> bool:
    """True when X (or its first layer) is a torch tensor on a non-CPU device."""
    first = X[0] if isinstance(X, (list, tuple)) and len(X) else X
    return _is_tensor(first) and first.device.type != "cpu"


def _split_layers(X: Layers, as_numpy: bool) -> tuple[list, list[int]]:
    """``(n, d)``, ``(n, L, d)`` or a list of ``(n, d_j)`` -> list of 2-D layers (views), dims."""
    if isinstance(X, (list, tuple)):
        mats = list(X)
    else:
        if not _is_tensor(X):
            X = np.asarray(X)
        if X.ndim == 3:
            mats = [X[:, j, :] for j in range(X.shape[1])]
        elif X.ndim == 2:
            mats = [X]
        else:
            raise ValueError(f"expected a 2-D or 3-D array, got shape {tuple(X.shape)}")
    if as_numpy:
        mats = [m.detach().cpu().numpy() if _is_tensor(m) else np.asarray(m) for m in mats]
    elif not all(_is_tensor(m) for m in mats):
        import torch

        dev = next(m.device for m in mats if _is_tensor(m))
        mats = [m if _is_tensor(m) else torch.as_tensor(np.asarray(m), device=dev) for m in mats]
    if any(m.ndim != 2 or len(m) != len(mats[0]) for m in mats):
        raise ValueError("every layer must be a 2-D array with the same number of rows")
    return mats, [int(m.shape[1]) for m in mats]


def is_classification_target(y) -> bool:
    """Classification iff y is non-numeric, or numeric with integral values and at most 20 distinct values."""
    if getattr(getattr(y, "dtype", None), "name", "") == "category":
        return True
    a = np.asarray(y)
    if a.dtype.kind not in "iuf":
        return True
    if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
        return False
    return len(np.unique(a)) <= 20


def raw_target(y, classification: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """One-hot over the sorted classes (classification) or the value (regression), unscaled; (Y, classes)."""
    a = np.asarray(y).ravel()
    if not classification:
        return a.astype(np.float64)[:, None], None
    try:
        classes, inv = np.unique(a, return_inverse=True)
    except TypeError:  # mixed, unorderable labels: the order does not change the result
        classes = np.array(sorted(set(a.tolist()), key=repr), dtype=object)
        pos = {c: i for i, c in enumerate(classes.tolist())}
        inv = np.array([pos[v] for v in a.tolist()])
    Y = np.zeros((len(a), len(classes)))
    Y[np.arange(len(a)), inv.ravel()] = 1.0
    return Y, classes


def target_matrix(y, classification: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """One-hot (classification) or the raw value (regression), standardized per column; returns (Y, classes)."""
    Y, classes = raw_target(y, classification)
    Y = Y - Y.mean(0)
    sd = Y.std(0)
    return Y / np.where(sd > _EPS_SD, sd, 1.0), classes


def _ref_sign(V: np.ndarray) -> np.ndarray:
    """Flip each column so its inner product with a fixed pseudo-random vector is positive."""
    ref = np.random.default_rng(_REF_SEED).standard_normal(V.shape[0])
    s = np.sign(ref @ V)
    return V * np.where(s == 0, 1.0, s)


def _top_eigvec_lanczos(matvec, v0: np.ndarray, tol: float, k: int = 64, check_every: int = 8, max_restarts: int = 20):
    """Top eigenvector of a symmetric operator by restarted Lanczos; returns (u, converged)."""
    d = len(v0)
    k = min(k, d)
    u = v0 / (np.linalg.norm(v0) or 1.0)
    for _ in range(max_restarts):
        Q = np.empty((d, k))
        MQ = np.empty((d, k))
        Q[:, 0] = u
        done = False
        for j in range(1, k + 1):
            MQ[:, j - 1] = matvec(Q[:, j - 1])
            last = j == k
            if j < k:
                q = MQ[:, j - 1].copy()
                for _ in range(2):  # classical Gram-Schmidt, twice
                    q -= Q[:, :j] @ (Q[:, :j].T @ q)
                nq = np.linalg.norm(q)
                if nq < 1e-14 * (np.linalg.norm(MQ[:, 0]) or 1.0):  # invariant subspace found
                    last = done = True
                else:
                    Q[:, j] = q / nq
            if last or j % check_every == 0:
                Qm, MQm = Q[:, :j], MQ[:, :j]
                H = Qm.T @ MQm
                theta, Sv = np.linalg.eigh((H + H.T) / 2)
                u, Mu = Qm @ Sv[:, -1], MQm @ Sv[:, -1]
                nu = np.linalg.norm(u) or 1.0  # zero only for a zero operator
                u, Mu = u / nu, Mu / nu
                if done or np.linalg.norm(Mu - theta[-1] * u) <= tol * (abs(theta[-1]) or 1.0):
                    return u, True
                if last:
                    break
    return u, False


def _finish_moments(mu, G, S, n, symmetric: bool = False):
    """Scale the centered moments to correlations: G / (n sd sd'), S / (n sd)."""
    if not symmetric:
        G = (G + G.T) / 2  # GEMM-based Gram matrices are not exactly symmetric
    sd = np.sqrt(np.maximum(np.diag(G) / n, 0.0))
    sd = np.where(sd > _EPS_SD, sd, 1.0)
    inv = 1.0 / (np.sqrt(n) * sd)
    G *= inv[:, None]
    G *= inv[None, :]
    S *= inv[:, None] / np.sqrt(n)
    return mu, sd, G, S


def _moment_sums(mats: list, Y: np.ndarray) -> np.ndarray:
    """The uncentered sums ``[X | 1 | Y]' [X | 1 | Y]`` (float64), one SYRK per row block; sums over
    disjoint row sets add up, so the moments of any union of row blocks follow by addition."""
    n = len(mats[0])
    dims = [m.shape[1] for m in mats]
    D = sum(dims)
    offs = np.cumsum([0] + dims)
    C = Y.shape[1]
    A = np.zeros((D + 1 + C, D + 1 + C))
    buf = np.empty((min(_BLOCK, n), D + 1 + C))
    buf[:, D] = 1.0
    for s in range(0, n, _BLOCK):
        e = min(s + _BLOCK, n)
        Xb = buf[: e - s]
        for j, m in enumerate(mats):
            Xb[:, offs[j] : offs[j + 1]] = m[s:e]
        Xb[:, D + 1 :] = Y[s:e]
        A += Xb.T @ Xb
    return A


def _host_moments(mats: list, Y: np.ndarray):
    """float64 moments in one pass: a single SYRK on row blocks of ``[X | 1 | Y]`` gives X'X, X'1 and
    X'Y; the centering is applied at the end in closed form."""
    n = len(mats[0])
    D = sum(m.shape[1] for m in mats)
    A = _moment_sums(mats, Y)
    G = np.ascontiguousarray(A[:D, :D])
    mu = A[:D, D] / n
    S = np.ascontiguousarray(A[:D, D + 1 :])
    G -= n * np.outer(mu, mu)
    S -= np.outer(mu, Y.sum(0))
    return _finish_moments(mu, G, S, n, symmetric=True)


def _torch_moments(mats: list, Y: np.ndarray):
    """The moments on the tensors' device: float32 GEMMs (TF32 off) on row blocks of
    ``[X - mu32 | 1 | Y]``, accumulated in float64; returns the D x D results on the host."""
    import torch

    dev = mats[0].device
    n = len(mats[0])
    f32 = [m if m.dtype == torch.float32 else m.to(torch.float32) for m in mats]
    tot = torch.zeros(sum(m.shape[1] for m in f32), dtype=torch.float64, device=dev)
    for s in range(0, n, _BLOCK_DEV):
        tot += torch.cat([m[s : s + _BLOCK_DEV].sum(0) for m in f32]).to(torch.float64)
    mu64 = tot / n
    mu = mu64.to(torch.float32)
    D = int(mu.numel())
    Yt = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=dev)
    A = torch.zeros((D + 1 + Yt.shape[1],) * 2, dtype=torch.float64, device=dev)
    ones = torch.ones((min(_BLOCK_DEV, n), 1), dtype=torch.float32, device=dev)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for s in range(0, n, _BLOCK_DEV):
            e = min(s + _BLOCK_DEV, n)
            Xb = torch.cat([m[s:e] for m in f32] + [ones[: e - s], Yt[s:e]], 1)
            Xb[:, :D] -= mu
            A += (Xb.T @ Xb).to(torch.float64)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    A = A.cpu().numpy()
    G, csum, S = A[:D, :D].copy(), A[:D, D].copy(), A[:D, D + 1 :].copy()
    mu_np = mu64.cpu().numpy()
    d = mu_np - mu.cpu().numpy().astype(np.float64)  # the blocks were centered at the float32 mean
    G -= np.outer(csum, d) + np.outer(d, csum) - n * np.outer(d, d)
    S -= np.outer(d, np.asarray(Y, dtype=np.float64).sum(0))
    return _finish_moments(mu_np, G, S, n)


class LightTARProjection(BaseEstimator, TransformerMixin):
    """Target-aware features from frozen encoder layers, in closed form.

    The output has ``n_anchor_ + n_supervised_`` float32 columns: the top principal components of the
    standardized first layer, then ``n_supervised`` deflated continuum-regression directions on the
    standardized concatenation of all layers (each the top eigenvector of ``Sxy Sxy' + gamma Sxx``
    with ``gamma = m / n``, mapped out of sample by the PLS rotation). Inputs are an ``(n, d)`` or
    ``(n, L, d)`` array, or a list of ``(n, d_j)`` arrays; torch tensors on a GPU are processed on
    that device. On ``n`` training rows both ranks are capped at ``n - 1``, and the supervised
    steps stop when the deflated states have no variance left; ``n_anchor_`` and ``n_supervised_`` are
    the fitted counts (the parameters when ``n`` exceeds both).

    Parameters
    ----------
    n_anchor : int, default 30
        Unsupervised anchor dimensions.
    n_supervised : int, default 5
        Supervised dimensions.
    m : float, default 60.0
        Shrinkage toward the anchor, ``gamma = m / n_train``; ``m=0`` is plain PLS.
    task : {"auto", "classification", "regression"}, default "auto"
        How y becomes the target block (one-hot or value).
    """

    def __init__(self, n_anchor: int = 30, n_supervised: int = 5, m: float = 60.0, task: str = "auto"):
        self.n_anchor = n_anchor
        self.n_supervised = n_supervised
        self.m = m
        self.task = task

    def fit(self, X: Layers, y) -> LightTARProjection:
        """Fit on the training rows."""
        if self.task not in ("auto", "classification", "regression"):
            raise ValueError("task must be 'auto', 'classification' or 'regression'")
        if self.task == "auto":
            self.is_classification_ = bool(is_classification_target(y))
        else:
            self.is_classification_ = self.task == "classification"
        Y, self.classes_ = target_matrix(y, self.is_classification_)
        device = _on_device(X)
        mats, dims = _split_layers(X, as_numpy=not device)
        n, D = len(mats[0]), sum(dims)
        if len(Y) != n:
            raise ValueError(f"X has {n} rows but y has {len(Y)}")
        if not 0 <= self.n_anchor <= dims[0] or not 0 <= self.n_supervised <= D:
            raise ValueError("n_anchor / n_supervised larger than the input dimension")
        self.gamma_ = float(self.m) / n
        mu, sd, G, S = _torch_moments(mats, Y) if device else _host_moments(mats, Y)
        self._fit_small(G, S, n, D, dims, mu, sd)
        return self

    def _fit_from_sums(self, A: np.ndarray, dims: list[int], classes: np.ndarray | None) -> LightTARProjection:
        """Fit from ``A = _moment_sums(X, Yraw)`` of the training rows, ``Yraw = raw_target(...)`` over
        ``classes`` (None: regression). Classes without a training row are dropped, so this equals ``fit``
        on the same rows up to the summation order. Regression sums are shift-invariant in y."""
        D, n = sum(dims), int(round(A[sum(dims), sum(dims)]))
        if not 0 <= self.n_anchor <= dims[0] or not 0 <= self.n_supervised <= D:
            raise ValueError("n_anchor / n_supervised larger than the input dimension")
        sel = np.arange(A.shape[0] - D - 1)
        if classes is not None:
            sel = np.flatnonzero(A[D, D + 1 :] > 0.5)  # class counts
        self.is_classification_, self.classes_ = classes is not None, None if classes is None else classes[sel]
        sx, sy, Sraw = A[:D, D], A[D, D + 1 + sel], A[:D, D + 1 + sel]
        mean_y = sy / n
        sd_y = np.sqrt(np.maximum(A[D + 1 + sel, D + 1 + sel] / n - mean_y**2, 0.0))
        sd_y = np.where(sd_y > _EPS_SD, sd_y, 1.0)
        mu = sx / n
        G = A[:D, :D] - n * np.outer(mu, mu)
        S = (Sraw - np.outer(sx, mean_y)) / sd_y
        self.gamma_ = float(self.m) / n
        mu, sd, G, S = _finish_moments(mu, G, S, n, symmetric=True)
        self._fit_small(G, S, n, D, list(dims), mu, sd)
        return self

    def _fit_small(self, G, S, n, D, dims, mu, sd) -> None:
        """All D x D work: anchor eigenvectors and the deflation steps on (G, S). The deflated Gram
        matrix is applied as an operator, ``G_k v = G v - A (c * (A' v))``. Ranks are capped at ``n - 1``,
        and a direction with (numerically) no variance left ends the deflation."""
        self.n_anchor_ = min(self.n_anchor, max(n - 1, 0))
        V = np.linalg.eigh(G[: dims[0], : dims[0]])[1][:, ::-1][:, : self.n_anchor_]
        A = np.zeros((D, self.n_anchor_))
        A[: dims[0]] = _ref_sign(V)
        gam = self.gamma_
        r = np.random.default_rng(0).standard_normal(D)  # fixed Lanczos start
        Ws, Ps, avs, cs = [], [], [], []
        floor = _EPS_RANK * n * float(np.trace(G))
        for _ in range(min(self.n_supervised, max(n - 1, 0))):
            Ak, ck = (np.stack(avs, 1), np.asarray(cs)) if avs else (None, None)

            def Gop(v, Ak=Ak, ck=ck):
                out = G @ v
                return out if Ak is None else out - Ak @ (ck * (Ak.T @ v))

            def Mop(v, Gop=Gop, S=S):
                return S @ (S.T @ v) + gam * Gop(v)

            w, ok = _top_eigvec_lanczos(Mop, Mop(r), tol=_LANCZOS_TOL)
            if not ok:  # not converged: full eigendecomposition
                Gk = G if Ak is None else G - (Ak * ck) @ Ak.T
                w = np.linalg.eigh(S @ S.T + gam * Gk)[1][:, -1]
            w = _ref_sign(w[:, None])[:, 0]
            a = n * Gop(w)
            b = n * (S.T @ w)
            tt = float(a @ w)
            if not tt > floor:  # rank exhausted (e.g. identical rows): fewer supervised columns
                break
            te = tt + _EPS_TT
            c = (2.0 / te - tt / te**2) / n
            avs.append(a)
            cs.append(c)
            S = S - c * np.outer(a, b)
            Ws.append(w)
            Ps.append(a / te)
        if Ws:
            W, P = np.stack(Ws, 1), np.stack(Ps, 1)
            R = W @ np.linalg.inv(P.T @ W)
        else:
            W = P = R = np.zeros((D, 0))
        self.n_supervised_ = len(Ws)
        self.mean_, self.scale_ = mu, sd
        self.anchor_components_ = A
        self.x_weights_, self.x_loadings_, self.x_rotations_ = W, P, R
        self.components_ = np.concatenate([A, R], axis=1)  # standardized X -> features
        self.layer_dims_ = dims
        self.n_features_in_ = D
        self._dev_cache = {}

    def transform(self, X: Layers) -> np.ndarray:
        """Project rows; returns an ``(n, n_anchor_ + n_supervised_)`` float32 numpy array."""
        if not hasattr(self, "components_"):
            raise RuntimeError("LightTARProjection is not fitted; call fit first")
        device = _on_device(X)
        mats, dims = _split_layers(X, as_numpy=not device)
        if list(dims) != list(self.layer_dims_):
            raise ValueError(f"layer dims {dims} differ from fit {self.layer_dims_}")
        Z = self._transform_torch(mats) if device else self._transform_host(mats)
        return Z.astype(np.float32, copy=False)

    def _coef(self, dtype) -> tuple[np.ndarray, np.ndarray]:
        """``Z = X @ C - off`` with ``C = components / scale`` and ``off = mean @ C``."""
        C = self.components_ / self.scale_[:, None]
        return np.ascontiguousarray(C, dtype=dtype), (self.mean_ @ C).astype(dtype)

    def _transform_host(self, mats: list) -> np.ndarray:
        dt = np.float32 if all(m.dtype == np.float32 for m in mats) else np.float64
        cache = self.__dict__.setdefault("_dev_cache", {})
        if ("host", dt) not in cache:
            cache[("host", dt)] = self._coef(dt)
        C, off = cache[("host", dt)]
        offs = np.cumsum([0] + list(self.layer_dims_))
        Z = mats[0].astype(dt, copy=False) @ C[offs[0] : offs[1]]
        for j in range(1, len(mats)):
            Z += mats[j].astype(dt, copy=False) @ C[offs[j] : offs[j + 1]]
        Z -= off
        return Z

    def _transform_torch(self, mats: list) -> np.ndarray:
        import torch

        dev = mats[0].device
        tdt = torch.float32 if all(m.dtype == torch.float32 for m in mats) else torch.float64
        cache = self.__dict__.setdefault("_dev_cache", {})
        key = (str(dev), str(tdt))
        if key not in cache:
            C, off = self._coef(np.float32 if tdt == torch.float32 else np.float64)
            cache[key] = (torch.as_tensor(C, device=dev), torch.as_tensor(off, device=dev))
        C, off = cache[key]
        offs = np.cumsum([0] + list(self.layer_dims_))
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            Z = mats[0].to(tdt) @ C[offs[0] : offs[1]]
            for j in range(1, len(mats)):
                Z += mats[j].to(tdt) @ C[offs[j] : offs[j + 1]]
            Z -= off
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return Z.cpu().numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        k = self.components_.shape[1] if hasattr(self, "components_") else self.n_anchor + self.n_supervised
        return np.array([f"lt{i}" for i in range(k)], dtype=object)

    def __getstate__(self):
        state = dict(super().__getstate__())
        state.pop("_dev_cache", None)  # device copies of the coefficients are rebuilt lazily
        return state

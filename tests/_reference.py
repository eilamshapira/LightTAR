"""Plain numpy reference implementation of the projection (the test oracle).

Explicit deflation of the n x D data matrix and a full eigendecomposition per step; the
package computes the same features up to column signs.
"""

import numpy as np


def ymat(y, is_cls):
    yv = np.asarray(y)
    if is_cls:
        labels = sorted(set(yv.tolist()))
        idx = {v: i for i, v in enumerate(labels)}
        Y = np.zeros((len(yv), len(labels)), np.float64)
        for i, v in enumerate(yv.tolist()):
            Y[i, idx[v]] = 1.0
    else:
        Y = yv.astype(np.float64)[:, None]
    Y = Y - Y.mean(0)
    sd = Y.std(0)
    return Y / np.where(sd > 1e-9, sd, 1.0)


def fit_layer(Xl, Y, gamma, k):
    n = len(Xl)
    mu, sd = Xl.mean(0), Xl.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Xr = (Xl - mu) / sd
    Yr = Y.copy()
    Ws, Ps = [], []
    for _ in range(k):
        Sxy = Xr.T @ Yr / n
        M = Sxy @ Sxy.T + gamma * (Xr.T @ Xr / n)
        w = np.linalg.eigh(M)[1][:, -1]
        t = Xr @ w
        tt = float(t @ t) + 1e-12
        p = Xr.T @ t / tt
        q = Yr.T @ t / tt
        Xr = Xr - np.outer(t, p)
        Yr = Yr - np.outer(t, q)
        Ws.append(w)
        Ps.append(p)
    W = np.stack(Ws, axis=1)
    Pm = np.stack(Ps, axis=1)
    return mu, sd, W @ np.linalg.inv(Pm.T @ W)


class ReferenceLightTAR:
    def __init__(self, y, is_cls, m=60.0, k_anchor=30, k_sup=5):
        self.y, self.is_cls, self.m = y, is_cls, m
        self.k_anchor, self.k_sup = k_anchor, k_sup

    def fit(self, X3):
        X3 = np.asarray(X3, np.float64)
        n = X3.shape[0]
        Xcat = X3.reshape(n, -1)
        self.pair = fit_layer(Xcat, ymat(self.y, self.is_cls), self.m / n, self.k_sup)
        Xa = X3[:, 0, :]
        mu, sd = Xa.mean(0), Xa.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        Xs = (Xa - mu) / sd
        V = np.linalg.eigh(Xs.T @ Xs / n)[1]
        self.anchor = (mu, sd, V[:, ::-1][:, : self.k_anchor])
        return self

    def transform(self, X3):
        X3 = np.asarray(X3, np.float64)
        n = X3.shape[0]
        Xcat = X3.reshape(n, -1)
        mu, sd, V = self.anchor
        pmu, psd, R = self.pair
        return np.concatenate([((X3[:, 0, :] - mu) / sd) @ V, ((Xcat - pmu) / psd) @ R], 1)

"""Pure Python pivoted-Cholesky compatibility routines.

These replace the historical f2py module.  Full-rank workflows are exact;
pivoted routines use a numerically stable greedy pivoted Cholesky.
"""
from __future__ import annotations
import numpy as np


def ichol_full(c, reltol=1e-12):
    C = np.asarray(c, dtype=float).copy()
    n = C.shape[0]
    piv = np.arange(n, dtype=np.int32)
    diag = np.diag(C).copy()
    max0 = max(float(diag.max(initial=0.0)), 0.0)
    U = np.zeros_like(C)
    m = 0
    for k in range(n):
        p = k + int(np.argmax(diag[k:]))
        if diag[p] <= max0*reltol or diag[p] <= 0: break
        if p != k:
            C[[k,p],:] = C[[p,k],:]; C[:,[k,p]] = C[:,[p,k]]
            diag[[k,p]] = diag[[p,k]]; piv[[k,p]] = piv[[p,k]]
            U[:k,[k,p]] = U[:k,[p,k]]
        U[k,k] = np.sqrt(diag[k])
        if k+1 < n:
            vals = (C[k,k+1:] - U[:k,k] @ U[:k,k+1:]) / U[k,k] if k else C[k,k+1:] / U[k,k]
            U[k,k+1:] = vals
            diag[k+1:] -= vals*vals
        m = k+1
    return U[:m, :], m, piv


def ichol(diag, reltol, rowfun, x, rl):
    x = np.asarray(x, dtype=float)
    n = len(diag)
    C = np.empty((n,n), dtype=float)
    for i in range(n):
        row = np.empty(n, dtype=float)
        rowfun(i, x, row)
        C[i] = row
    C = (C + C.T)/2
    U, m, piv = ichol_full(C, reltol)
    return U[:min(m,rl)], min(m,rl), piv


def ichol_continue(sig, diag, reltol, rowfun, piv, x, mold):
    # Compatibility fallback: reconstruct and refactorize the full covariance.
    U, m, p = ichol(diag, reltol, rowfun, x, sig.shape[0])
    sig[...] = 0.0
    sig[:U.shape[0], :U.shape[1]] = U
    return m, p


def ichol_basis(basis, nug, reltol=1e-12):
    B = np.asarray(basis, dtype=float); nug = np.asarray(nug, dtype=float)
    C = B.T @ B + np.diag(nug)
    U, m, piv = ichol_full(C, reltol)
    return U, piv, m

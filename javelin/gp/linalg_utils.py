"""NumPy/SciPy replacements for JAVELIN's historical f2py BLAS/LAPACK wrappers."""
from __future__ import annotations
import numpy as np
from scipy import linalg


def remove_duplicates(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    seen = {}
    rf, rt, unique, ui = [], [], [], []
    for i, row in enumerate(x):
        key = tuple(row.tolist())
        if key in seen:
            rf.append(seen[key]); rt.append(i)
        else:
            seen[key] = i; unique.append(row.copy()); ui.append(i)
    xu = np.asarray(unique, dtype=float).reshape(-1, x.shape[1])
    return len(rt), np.asarray(rf, dtype=np.int32), np.asarray(rt, dtype=np.int32), len(ui), xu, np.asarray(ui, dtype=np.int32)


def check_repeats(x, x_sofar, f_sofar):
    x = np.asarray(x, dtype=float); x_sofar = np.asarray(x_sofar, dtype=float); f_sofar = np.asarray(f_sofar)
    f = np.empty(x.shape[0], dtype=float)
    new = []
    lookup = {tuple(row.tolist()): f_sofar[i] for i, row in enumerate(x_sofar)}
    for i, row in enumerate(x):
        key = tuple(row.tolist())
        if key in lookup: f[i] = lookup[key]
        else: new.append(i)
    return f, np.asarray(new, dtype=np.int32), len(new)


def diag_call(x, cov_fun):
    x = np.asarray(x, dtype=float)
    return np.asarray([cov_fun(np.atleast_2d(row)) for row in x], dtype=float)


def basis_diag_call(basis_x):
    a = np.asarray(basis_x, dtype=float)
    return np.sum(a*a, axis=0)


def gp_array_logp(x, mu, sig):
    x = np.asarray(x, dtype=float) - np.asarray(mu, dtype=float)
    U = np.asarray(sig, dtype=float)
    y = linalg.solve_triangular(U.T, x, lower=True, check_finite=False)
    return -0.5*np.dot(y, y) - 0.5*x.size*np.log(2*np.pi) - np.log(np.diag(U)).sum()


def asqs(C, S, cmin=0, cmax=-1):
    C = np.asarray(C); Sarr = np.asarray(S)
    if cmax == -1: cmax = C.shape[1]
    Sarr[cmin:cmax] = np.sum(C[:, cmin:cmax]**2, axis=0)
    return None


def dcopy_wrap(x, y):
    y[...] = x


def dtrmm_wrap(a, b, side='L', transa='N', uplo='U', alpha=1.0):
    A = np.asarray(a); op = A.T if str(transa).upper().startswith('T') else A
    b[...] = alpha * (op @ b if str(side).upper().startswith('L') else b @ op)


def dtrsm_wrap(a, b, side='L', transa='N', uplo='U', alpha=1.0):
    A = np.asarray(a)
    lower = str(uplo).upper().startswith('L')
    trans = 'T' if str(transa).upper().startswith('T') else 'N'
    if str(side).upper().startswith('L'):
        b[...] = linalg.solve_triangular(A, alpha*np.asarray(b), lower=lower, trans=trans, check_finite=False)
    else:
        opA = A.T if trans == 'T' else A
        b[...] = (linalg.solve_triangular(opA.T, (alpha*np.asarray(b)).T, lower=not lower, check_finite=False)).T


def _potrf(a, lower):
    try:
        c = linalg.cholesky(np.asarray(a), lower=lower, check_finite=False)
        a[...] = c
        return 0
    except linalg.LinAlgError:
        return 1


def dpotrf_wrap(a): return _potrf(a, False)
def dpotrf2_wrap(a): return _potrf(a, True)


def dpotrs_wrap(chol_fac, b, uplo='U'):
    lower = str(uplo).upper().startswith('L')
    b[...] = linalg.cho_solve((np.asarray(chol_fac), lower), np.asarray(b), check_finite=False)
    return 0

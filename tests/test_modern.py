import io
import numpy as np


def test_import_and_version():
    import javelin
    assert isinstance(javelin.__version__, str)


def test_lcio_filelike_does_not_close_handle():
    from javelin.lcio import readlc_3c
    f = io.StringIO('1 2 0.1\n2 3 0.2\n')
    out = readlc_3c(f)
    assert len(out) == 1
    assert np.allclose(out[0][0], [1.0, 2.0])
    assert not f.closed


def test_default_sampler_factory_imports():
    import javelin
    from javelin.samplers import EnsembleSampler
    javelin.set_sampler_backend('internal')
    s = EnsembleSampler(6, 2, lambda x: -0.5*np.dot(x, x), threads=1)
    assert s is not None
    javelin.set_sampler_backend('auto')


def test_scalarize_size_one_covariance():
    from javelin.predict import _scalarize
    assert _scalarize(np.array([[2.5]])) == 2.5
    assert _scalarize(np.array([[3.5]])) == 3.5



def test_demo_continuum_likelihood_is_finite():
    from pathlib import Path
    from javelin.zylc import get_data
    from javelin.lcmodel import Cont_Model

    dat = Path(__file__).resolve().parents[1] / 'examples' / 'dat' / 'continuum.dat'
    zydata = get_data(str(dat), names=['Continuum'])
    model = Cont_Model(zydata, 'drw')
    vals = model([np.log(3.0), np.log(400.0)], set_retq=True)
    assert np.isfinite(np.asarray(vals[:-1], dtype=object)[0])
    assert float(vals[0]) > -1.0e100


def test_basis_covariance_ndarray_semantics():
    from javelin.gp.BasisCovariance import BasisCovariance

    basis = np.array([
        lambda x: np.ones(x.shape[0]),
        lambda x: x[:, 0],
    ], dtype=object)
    coef_cov = np.array([[1.0, 0.2], [0.2, 0.5]])
    cov = BasisCovariance(basis, coef_cov)
    x = np.array([[0.0], [0.5], [1.0]])

    K = cov(x, x)
    U = cov.cholesky(x)
    assert isinstance(K, np.ndarray)
    assert isinstance(U, np.ndarray)
    assert K.shape == (3, 3)
    assert np.allclose(U.T @ U, K, rtol=1e-12, atol=1e-12)
    assert np.allclose(cov(x), np.diag(K), rtol=1e-12, atol=1e-12)


def test_vectorized_spear_normal_matches_scalar_reference():
    from javelin.spear_covfunc import _fill, _fill_normal_vectorized

    rng = np.random.default_rng(1234)
    jd1 = np.sort(rng.uniform(0.0, 50.0, 17))
    jd2 = np.sort(rng.uniform(0.0, 50.0, 21))
    id1 = rng.integers(1, 4, jd1.size)
    id2 = rng.integers(1, 4, jd2.size)
    sigma, tau = 2.3, 11.7
    lags = np.array([0.0, 3.2, 8.0])
    wids = np.array([0.0, 2.5, 4.0])
    scales = np.array([1.0, 0.7, 0.4])

    reference = np.empty((jd1.size, jd2.size))
    vectorized = np.empty_like(reference)
    _fill(reference, jd1, jd2, id1, id2, sigma, tau,
          lags, wids, scales, 0, -1, 'normal')
    _fill_normal_vectorized(vectorized, jd1, jd2, id1, id2, sigma, tau,
                            lags, wids, scales, 0, -1)
    assert np.array_equal(vectorized, reference)


def test_vectorized_pmap_dpmap_match_scalar():
    import numpy as np
    import javelin.spear_covfunc as sc
    rng = np.random.default_rng(123)
    n = 48
    jd = np.sort(rng.uniform(0.0, 500.0, n))
    ids = np.r_[np.ones(n//2, dtype=int), np.full(n-n//2, 2, dtype=int)]
    lags = np.array([0.0, 100.0, 250.0])
    wids = np.array([0.0, 5.0, 7.0])
    scales = np.array([1.0, 0.5, 0.3])
    for mode in ("pmap", "dpmap"):
        ref = np.zeros((n,n), dtype=float)
        got = np.empty_like(ref)
        sc._fill(ref, jd, jd, ids, ids, 3.0, 400.0, lags, wids, scales, mode=mode)
        ref = ref + np.triu(ref, 1).T
        sc._fill_mode_vectorized(got, jd, jd, ids, ids, 3.0, 400.0, lags, wids, scales, mode=mode)
        assert np.allclose(got, ref, rtol=1e-13, atol=1e-13)

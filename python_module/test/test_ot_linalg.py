"""
Numpy-only unit tests for the dense linear algebra kernels of the OT method
(sirius.ot.ot_transformations, sirius.ot.ot_helpers) and the Gram-Schmidt routines
in sirius.edft.ortho.

These tests do not need the compiled py_sirius extension. If `import sirius` fails
(no py_sirius available), the pure python subpackages are imported directly from the
source tree.

The `_ref_*` functions are verbatim copies of the original `np.matrix` based
implementations (before the np.matrix -> ndarray cleanup) and serve as golden
references.
"""

import pathlib
import sys
import types
import warnings

import numpy as np
import pytest
from scipy.sparse import dia_matrix

try:
    import sirius  # type: ignore # noqa: F401
except ImportError:
    # py_sirius not available: register stub packages so that the pure python
    # submodules can be imported without executing sirius/__init__.py
    _root = pathlib.Path(__file__).resolve().parents[1] / "sirius"
    for _k in [k for k in sys.modules if k == "sirius" or k.startswith("sirius.")]:
        del sys.modules[_k]
    for _name, _path in [
        ("sirius", _root),
        ("sirius.ot", _root / "ot"),
        ("sirius.edft", _root / "edft"),
    ]:
        _m = types.ModuleType(_name)
        _m.__path__ = [str(_path)]
        sys.modules[_name] = _m

import sirius.ot.ot_helpers as oth  # type: ignore
import sirius.ot.ot_transformations as ott  # type: ignore
from numpy.linalg import eigh, solve
from sirius.coefficient_array import CoefficientArray, PwCoeffs  # type: ignore
from sirius.edft import ortho  # type: ignore

# ---------------------------------------------------------------------------
# reference implementations (np.matrix based, copied from the original code)
# ---------------------------------------------------------------------------


def _mat(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix(x, copy=False)


def _ref_lagrangeMult(p, c0, P=None):
    c0 = _mat(c0)
    p = _mat(p)
    if P is not None:
        oc = solve(c0.H * P * c0, c0.H * P * p)
        correction = -P * c0 * oc
    else:
        oc = solve(c0.H * c0, c0.H * p)
        correction = -c0 * oc
    return correction


def _ref_c(x, c0):
    x = _mat(x)
    XX = np.dot(x.H, x)
    w, R = eigh(XX)
    w = np.where(np.abs(w) < 1e-14, 0, w)
    w = np.sqrt(w)
    R = _mat(R)
    Wsinc = np.diag(np.sinc(w / np.pi))
    sincU = np.dot(np.dot(R, Wsinc), R.H)
    Wcos = np.diag(np.cos(w))
    cosU = np.dot(np.dot(R, Wcos), R.H)
    return np.dot(c0, cosU) + np.dot(x, sincU)


def _ref_prepare_fX(x, c0):
    x = _mat(x)
    XX = np.dot(x.H, x)
    Λ, R = eigh(XX)
    w = np.sqrt(Λ)
    R = _mat(R)
    Wsinc = np.diag(np.sinc(w / np.pi))
    sincU = np.dot(np.dot(R, Wsinc), R.H)
    Wcos = np.diag(np.cos(w))
    cosU = np.dot(np.dot(R, Wcos), R.H)
    c = np.dot(c0, cosU) + np.dot(x, sincU)
    return c, Λ, sincU, R


def _ref_apply_single(x, c0, Λ, Hc, sincU, R):
    R = _mat(R)
    sincU = _mat(sincU)
    Hc = _mat(Hc)
    c0 = _mat(c0)

    v = np.sinc(np.sqrt(Λ) / np.pi)
    v = v[:, np.newaxis]
    diffL = Λ[:, np.newaxis] - Λ[:, np.newaxis].T
    mask = np.abs(diffL) < 1e-10
    D1 = np.ma.masked_array(v - v.T, mask=mask) / diffL
    irow, icol = np.where(mask)

    def f1(la):
        pos_index = la > 1e-10
        out = np.zeros(la.shape, dtype=la.dtype)
        out[pos_index] = 0.5 * (
            np.cos(np.sqrt(la[pos_index])) / la[pos_index]
            - np.sin(np.sqrt(la[pos_index])) / (la[pos_index] ** (1.5))
        )
        out[~pos_index] = -1 / 6
        return out

    D1[irow, icol] = f1(Λ[irow])
    v = np.cos(np.sqrt(Λ))
    v = v[:, np.newaxis]
    D2 = np.ma.masked_array(v - v.T, mask=mask) / diffL
    D2[irow, icol] = -0.5 * np.sinc(np.sqrt(Λ[irow]) / np.pi)

    RtHCtxR = np.array(R.H * (Hc.H * x) * R)
    RtHCtc0R = np.array(R.H * Hc.H * c0 * R)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        K = np.matrix(RtHCtxR * np.array(D1) + RtHCtc0R * np.array(D2))
    dEdx = Hc * sincU + x * (R * (K.H + K) * R.H)
    lM = solve(c0.H @ c0, c0.H @ dEdx)
    correction = -1 * c0 @ lM
    return dEdx + correction, dEdx


def _ref_sinU(x):
    x = _mat(x)
    XX = np.dot(x.H, x)
    w, R = np.linalg.eigh(XX)
    w = np.sqrt(w)
    R = _mat(R)
    return np.dot(np.dot(R, np.diag(np.sin(w))), R.H)


def _ref_cosU(x):
    x = _mat(x)
    XX = np.dot(x.H, x)
    w, R = np.linalg.eigh(XX)
    w = np.sqrt(w)
    R = _mat(R)
    return np.dot(np.dot(R, np.diag(np.cos(w))), R.H)


def _ref_gram_schmidt(X):
    X = _mat(X)
    m = X.shape[1]
    Q = np.zeros_like(X)
    Q[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0])
    for i in range(1, m):
        Xi = X[:, i].copy()
        for j in range(i):
            Xi -= np.tensordot(Xi, np.conj(Q[:, j]), axes=2) * Q[:, j]
        Q[:, i] = Xi / np.linalg.norm(Xi)
    return Q


def _ref_modified_gram_schmidt(X):
    X = _mat(X)
    m = X.shape[1]
    Q = np.zeros_like(X)
    for k in range(m):
        Q[:, k] = X[:, k]
        for i in range(k):
            Q[:, k] = (
                Q[:, k] - np.tensordot(Q[:, k], np.conj(Q[:, i]), axes=2) * Q[:, i]
            )
        Q[:, k] = Q[:, k] / np.linalg.norm(Q[:, k])
    return Q


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

NPW = 40  # number of plane-wave coefficients
NBND = 6  # number of bands


def _randc(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _orthonormal(rng, n, m):
    q, _ = np.linalg.qr(_randc(rng, n, m))
    return q


def _tangent(rng, c0, scale=0.3):
    """random x with c0^H x = 0"""
    x = scale * _randc(rng, *c0.shape)
    return x - c0 @ (c0.conj().T @ x)


def _hermitian(rng, n):
    A = _randc(rng, n, n)
    return 0.5 * (A + A.conj().T)


def _energy(H, x, c0):
    c = ott.c(x, c0)
    return np.real(np.trace(np.conj(c).T @ (H @ c)))


def assert_no_matrix(obj, where=""):
    """Assert that neither obj nor any block of a CoefficientArray is a np.matrix."""
    blocks = obj._data.values() if isinstance(obj, CoefficientArray) else [obj]
    for b in blocks:
        assert not isinstance(b, np.matrix), f"np.matrix leaked into {where}"


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def system(rng):
    c0 = _orthonormal(rng, NPW, NBND)
    x = _tangent(rng, c0)
    H = _hermitian(rng, NPW)
    return c0, x, H


def _precond(rng, kind):
    if kind == "none":
        return None
    if kind == "scalar":
        return 0.7
    d = 1 / (rng.uniform(0.5, 3.0, NPW))
    return dia_matrix((d, 0), shape=(NPW, NPW))


# ---------------------------------------------------------------------------
# golden tests: new implementation vs. np.matrix reference
# ---------------------------------------------------------------------------

TOL = {"rtol": 1e-12, "atol": 1e-12}


@pytest.mark.parametrize("kind", ["none", "scalar", "dia_matrix"])
def test_lagrangeMult_matches_reference(rng, system, kind):
    c0, _, _ = system
    p = _randc(rng, NPW, NBND)
    P = _precond(rng, kind)
    ref = _ref_lagrangeMult(p, c0, P)
    new = ott.lagrangeMult(p, c0, P)
    np.testing.assert_allclose(np.asarray(new), np.asarray(ref), **TOL)


def test_c_matches_reference(system):
    c0, x, _ = system
    np.testing.assert_allclose(
        np.asarray(ott.c(x, c0)), np.asarray(_ref_c(x, c0)), **TOL
    )


def test_c_is_orthonormal(system):
    c0, x, _ = system
    c = np.asarray(ott.c(x, c0))
    np.testing.assert_allclose(c.conj().T @ c, np.eye(NBND), atol=1e-12)


def test_prepare_fX_matches_reference(system):
    c0, x, _ = system
    new = ott.ConstrainedGradient._prepare_fX(x, c0)
    ref = _ref_prepare_fX(x, c0)
    for a, b in zip(new, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), **TOL)


def test_apply_single_matches_reference(system):
    c0, x, H = system
    c, Λ, sincU, R = _ref_prepare_fX(x, c0)
    Hc = H @ np.asarray(c)
    new = ott.ConstrainedGradient._apply_single(x, c0, Λ, Hc, sincU, R)
    ref = _ref_apply_single(x, c0, Λ, Hc, sincU, R)
    for a, b in zip(new, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), **TOL)


def test_sinU_cosU_match_reference(system):
    _, x, _ = system
    np.testing.assert_allclose(np.asarray(oth.sinU(x)), np.asarray(_ref_sinU(x)), **TOL)
    np.testing.assert_allclose(np.asarray(oth.cosU(x)), np.asarray(_ref_cosU(x)), **TOL)


@pytest.mark.parametrize("name", ["gram_schmidt", "modified_gram_schmidt"])
def test_gram_schmidt_matches_reference(rng, name):
    X = _randc(rng, NPW, NBND)
    ref = {
        "gram_schmidt": _ref_gram_schmidt,
        "modified_gram_schmidt": _ref_modified_gram_schmidt,
    }[name]
    Q = np.asarray(getattr(ortho, name)(X))
    np.testing.assert_allclose(Q, np.asarray(ref(X)), **TOL)
    np.testing.assert_allclose(Q.conj().T @ Q, np.eye(NBND), atol=1e-12)


# ---------------------------------------------------------------------------
# finite-difference check of the OT gradient
# ---------------------------------------------------------------------------


def test_apply_single_gradient_fd(rng, system):
    """dE/dx from ConstrainedGradient vs. central finite difference along a tangent p."""
    c0, x, H = system
    c, Λ, sincU, R = ott.ConstrainedGradient._prepare_fX(x, c0)
    Hc = H @ np.asarray(c)
    Gx, dEdx = ott.ConstrainedGradient._apply_single(x, c0, Λ, Hc, sincU, R)

    p = _tangent(rng, c0, scale=1.0)
    h = 1e-5
    slope_fd = (_energy(H, x + h * p, c0) - _energy(H, x - h * p, c0)) / (2 * h)
    slope = 2 * np.real(np.vdot(np.asarray(dEdx), p))
    slope_G = 2 * np.real(np.vdot(np.asarray(Gx), p))
    assert slope == pytest.approx(slope_fd, rel=1e-7)
    # projected gradient differs from dEdx only within span(c0), p is orthogonal to it
    assert slope_G == pytest.approx(slope_fd, rel=1e-7)


# ---------------------------------------------------------------------------
# no np.matrix may leak out of the kernels
# ---------------------------------------------------------------------------


class _DenseHamiltonian:
    """Applies a dense Hermitian matrix to PwCoeffs or to a single block."""

    def __init__(self, H):
        self.H = H

    def __call__(self, c, ki=None, ispn=None):
        if isinstance(c, CoefficientArray):
            out = PwCoeffs()
            for key, v in c.items():
                out[key] = self.H @ np.asarray(v)
            return out
        return self.H @ np.asarray(c)


def _pw(*blocks):
    out = PwCoeffs()
    for i, b in enumerate(blocks):
        out[(i // 2, i % 2)] = b
    return out


@pytest.mark.xfail(strict=True, reason="np.matrix still used in OT kernels")
@pytest.mark.parametrize("kind", ["none", "scalar", "dia_matrix"])
def test_no_matrix_lagrangeMult(rng, system, kind):
    c0, _, _ = system
    p = _randc(rng, NPW, NBND)
    assert type(ott.lagrangeMult(p, c0, _precond(rng, kind))) is np.ndarray


@pytest.mark.xfail(strict=True, reason="np.matrix still used in OT kernels")
def test_no_matrix_kernels(system):
    c0, x, H = system
    assert type(ott.c(x, c0)) is np.ndarray
    c, Λ, sincU, R = ott.ConstrainedGradient._prepare_fX(x, c0)
    for a in (c, Λ, sincU, R):
        assert type(a) is np.ndarray
    for a in ott.ConstrainedGradient._apply_single(x, c0, Λ, H @ c, sincU, R):
        assert type(a) is np.ndarray
    assert type(oth.sinU(x)) is np.ndarray
    assert type(oth.cosU(x)) is np.ndarray
    assert type(ortho.gram_schmidt(x)) is np.ndarray
    assert type(ortho.modified_gram_schmidt(x)) is np.ndarray


@pytest.mark.xfail(strict=True, reason="np.matrix still used in OT kernels")
def test_no_matrix_constrained_gradient(rng, system):
    """Exercise both branches of ConstrainedGradient.__call__ (PwCoeffs and ndarray)."""
    c0, x, H = system
    c0b = _orthonormal(rng, NPW, NBND)
    xb = _tangent(rng, c0b)
    hamiltonian = _DenseHamiltonian(H)

    # PwCoeffs branch (two spin components in one k-point)
    c0_pw, x_pw = _pw(c0, c0b), _pw(x, xb)
    assert_no_matrix(ott.c(x_pw, c0_pw), "c(x, c0)")
    Gx, Hx = ott.ConstrainedGradient(hamiltonian, c0_pw)(x_pw)
    assert_no_matrix(Gx, "ConstrainedGradient Gx")
    assert_no_matrix(Hx, "ConstrainedGradient Hx")
    assert_no_matrix(ott.lagrangeMult(x_pw, c0_pw), "lagrangeMult")

    # ndarray branch, must agree with the PwCoeffs branch
    G0, H0 = ott.ConstrainedGradient(hamiltonian, c0)(x, ki=0, ispn=0)
    assert type(G0) is np.ndarray and type(H0) is np.ndarray
    np.testing.assert_allclose(G0, Gx[(0, 0)], **TOL)
    np.testing.assert_allclose(H0, Hx[(0, 0)], **TOL)

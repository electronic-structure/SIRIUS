from __future__ import annotations
import numpy as np
from mpi4py import MPI
from scipy.sparse import dia_matrix
from numpy.typing import ArrayLike, NDArray
from typing import Union, Tuple, Any, Callable, get_type_hints, TypeAlias
from collections import abc
import copy

scalar_t: TypeAlias = Union[int, float, complex, np.float64, np.complex128]


def check_return_type(func: Callable) -> bool:
    annotations = func.__annotations__
    return annotations.get("return") == Tuple[Any, Any]


def threaded(f):
    """Decorator for threaded application over CoefficientArray."""

    def _f(x, *args, **kwargs):
        if isinstance(x, CoefficientArray):
            out = type(x)()
            for k in x._data.keys():
                out[k] = f(x[k], *args, **kwargs)
            return out
        else:
            return f(x, *args, **kwargs)

    return _f


def threaded_class(f):
    """Decorator for threaded (in class members)
    application over CoefficientArray."""

    def _f(self, x, *args, **kwargs):
        if isinstance(x, CoefficientArray):
            out = type(x)()
            for k in x._data.keys():
                out[k] = f(self, x[k], *args, **kwargs)
            return out
        else:
            return f(self, x, *args, **kwargs)

    return _f


@threaded
def sort(x):
    return np.sort(x)


@threaded
def shape(x):
    return np.shape(x)


@threaded
def trace(x):
    return np.trace(x)


@threaded
def identity_like(x):
    return np.eye(*x.shape)


def eye_like(shapes):
    """
    Arguments:
    shapes -- a map of (key,tuple)

    Returns:
    out -- CoefficientArray of identities according to shapes
    """
    out = CoefficientArray()
    for k in shapes:
        out[k] = np.eye(*shapes[k])
    return out


def diag(x):
    """
    TODO: add a check not to flatten a 2d matrix
    """
    if isinstance(x, CoefficientArray):
        out = type(x)()
        for key, val in x._data.items():
            if val.size == max(val.shape):
                out[key] = np.diag(np.array(val, copy=True).flatten())
            else:
                out[key] = np.diag(val)
        return out
    else:
        return np.diag(x)


def spdiag(x):
    """
    Diagonal matrix (scipy.sparse.dia_matrix)
    """
    if isinstance(x, CoefficientArray):
        out = type(x)()
        for key, val in x._data.items():
            n = np.size(val)
            out[key] = dia_matrix((val, 0), shape=(n, n))
        return out
    n = np.size(x)
    return dia_matrix((x, 0), shape=(n, n))


def ones_like(x, *args):
    """Numpy ones_like."""
    if isinstance(x, CoefficientArray):
        return CoefficientArray.ones_like(x, *args)
    return np.ones_like(x, *args)


def zeros_like(x, *args):
    """Numpy zeros_like."""
    if isinstance(x, CoefficientArray):
        return CoefficientArray.zeros_like(x, *args)
    return np.zeros_like(x, *args)


def inner(a, b):
    """Complex inner product."""
    return (a * b.conjugate()).sum()


def l2norm(a):
    """L2-norm."""
    return np.sqrt(np.real(inner(a, a)))


def einsum(expr, *operands):
    """
    map einsum over elements of CoefficientArray
    """

    assert operands

    try:
        return np.einsum(expr, *operands)
    except (ValueError, TypeError):
        out = type(operands[0])()
        for key in operands[0]._data.keys():
            out[key] = np.einsum(expr, *list(map(lambda x: x[key], operands)))
        return out


class CoefficientArrayBase(abc.Mapping):
    def __init__(self):
        self._data = {}

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)


class CoefficientArray(CoefficientArrayBase):
    """CoefficientArray class."""

    def __init__(self):
        """ """
        super().__init__()

    def __getitem__(self, key: Tuple[int, int]):
        """
        key -- (k, ispn)
        """

        # # return as view, not required
        # return self._data[key][:]

        # return as view
        return self._data[key]

    def __setitem__(
        self, key: Tuple[int, int], item: Union[ArrayLike, dia_matrix, Any]
    ):
        """ """
        if (key in self._data) and hasattr(item, "__array__"):
            x = self._data[key]
            # make sure shapes don't change
            x[:] = item
        else:
            if isinstance(item, np.ndarray):
                self._data[key] = copy.deepcopy(item)
            else:
                self._data[key] = item

    def __array_ufunc__(self, ufunc, method, *inputs, **_):
        if ufunc not in (
            np.sqrt,
            np.log,
            np.absolute,
            np.conjugate,
            np.multiply,
            np.divide,
        ):
            return NotImplemented
        if method == "__call__":
            if ufunc == np.sqrt:
                return self.sqrt()
            elif ufunc == np.log:
                return self.log()
            elif ufunc == np.absolute:
                return self.abs()
            elif ufunc == np.conjugate:
                return self.conjugate()
            elif ufunc == np.multiply:
                if isinstance(inputs[0], CoefficientArray):
                    return NotImplemented
                else:
                    return self.__mul__(inputs[0])
            elif ufunc == np.divide:
                if isinstance(inputs[0], CoefficientArray):
                    return NotImplemented
                else:
                    return self.__rtruediv__(inputs[0])
            else:
                return NotImplemented
        else:
            return NotImplemented

    def sum(self, **kwargs):
        """ """
        loc_sum = np.array(sum([np.sum(v, **kwargs) for _, v in self.items()]))
        reduced = MPI.COMM_WORLD.allreduce(loc_sum, op=MPI.SUM)
        return np.ndarray.item(reduced)

    def log(self, **kwargs):
        out = type(self)()
        for key in self._data.keys():
            out[key] = np.log(self._data[key], **kwargs)
        return out

    def __mul__(self, other):
        """ """
        out = type(self)()
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum("...,...->...", self._data[key], other._data[key])
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] * other
        else:
            raise TypeError("wrong type")
        return out

    def max(self, **kwargs):
        if kwargs:
            return [np.max(self._data[key], **kwargs) for key in self._data.keys()]
        else:
            return np.max([np.max(self._data[key]) for key in self._data.keys()])

    def min(self, **kwargs):
        if kwargs:
            return [np.min(self._data[key], **kwargs) for key in self._data.keys()]
        else:
            return np.min([np.min(self._data[key]) for key in self._data.keys()])

    def __truediv__(self, other):
        out = type(self)()
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum(
                    "...,...->...", self._data[key], 1 / other._data[key]
                )
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] / other
        else:
            raise TypeError("wrong type")
        return out

    def __rtruediv__(self, other):
        out = type(self)()
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum(
                    "...,...->...", 1 / self._data[key], other._data[key]
                )
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = other / self._data[key]
        else:
            raise TypeError("wrong type")
        return out

    def __matmul__(self, other: CoefficientArrayBase):
        """ """
        out = type(self)()
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = self._data[key] @ other._data[key]
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] * other
        else:
            raise TypeError("wrong type")
        return out

    def __add__(self, other: Union[CoefficientArrayBase, scalar_t]) -> CoefficientArray:
        out = type(self)()
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = self._data[key] + other._data[key]
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] + other
        return out

    def __neg__(self) -> CoefficientArray:
        out = type(self)()
        for key, val in self._data.items():
            out[key] = -val
        return out

    def abs(self) -> CoefficientArray:
        out = type(self)()
        for key in self._data.keys():
            out[key] = np.abs(self._data[key])
        return out

    def sqrt(self) -> CoefficientArray:
        out = type(self)()
        for key in self._data.keys():
            out[key] = np.sqrt(self._data[key])
        return out

    def svd(self, **args):
        """
        returns U, s, Vh
        """
        U = type(self)()
        s = type(self)()
        Vh = type(self)()
        for key in self._data.keys():
            Ul, sl, Vhl = np.linalg.svd(self[key], **args)
            U[key] = Ul
            s[key] = sl
            Vh[key] = Vhl
        return U, s, Vh

    def eigh(self, **args):
        w = type(self)()
        V = type(self)()
        for key in self._data.keys():
            w[key], V[key] = np.linalg.eigh(self[key], **args)
        return w, V

    def eig(self, **args):
        w = type(self)()
        V = type(self)()
        for key in self._data.keys():
            w[key], V[key] = np.linalg.eig(self[key], **args)
        return w, V

    def qr(self, **args):
        Q = type(self)()
        R = type(self)()
        for key in self._data.keys():
            Q[key], R[key] = np.linalg.qr(self[key], **args)
        return Q, R

    def keys(self):
        return self._data.keys()

    def __sub__(self, other) -> CoefficientArray:
        return self.__add__(-1 * other)

    def __rsub__(self, other) -> CoefficientArray:
        return -(self.__add__(-1 * other))

    def __pow__(self, a):
        """ """
        out = type(self)()
        for key in self._data.keys():
            out[key] = self._data[key] ** a
        return out

    def conjugate(self) -> CoefficientArray:
        out = type(self)()
        for key, val in self._data.items():
            out[key] = np.array(np.conj(val), copy=True)
        return out

    def conj(self) -> CoefficientArray:
        return self.conjugate()

    def flatten(self, *args):
        out = type(self)()
        for key, val in self._data.items():
            out[key] = val.flatten(*args)
        return out

    def asarray(self) -> CoefficientArray:
        """ """
        out = type(self)()
        for key, val in self._data.items():
            out[key] = np.asarray(val)
        return out

    def __len__(self):
        """ """
        return len(self._data)

    def items(self):
        """ """
        return self._data.items()

    def __contains__(self, key):
        """ """
        return self._data.__contains__(key)

    @property
    def real(self):
        out = type(self)()
        for key, val in self._data.items():
            out[key] = np.real(val)
        return out

    @property
    def imag(self):
        out = type(self)()
        for key, val in self._data.items():
            out[key] = np.imag(val)
        return out

    @property
    def H(self):
        """
        Hermitian conjugate
        """
        out = type(self)()
        for key, val in self._data.items():
            out[key] = np.conj(np.atleast_2d(val)).T
        return out

    @property
    def T(self):
        """
        Transpose
        """
        out = type(self)()
        for key, val in self._data.items():
            out[key] = np.atleast_2d(val).T
        return out

    def __str__(self):
        return "\n".join(
            ["\n".join(map(str, [key, val, "---"])) for key, val in self._data.items()]
        )

    def to_array(self) -> NDArray:
        """
        convert to numpy array
        """
        if len(self) > 0:
            return np.concatenate(list(map(np.atleast_1d, self._data.values())), axis=0)
        else:
            return np.array([])

    def from_array(self, X):
        """
        set internal data from numpy array
        assumes that each entry has the same number of columns
        """
        offset = 0
        for _, val in self._data.items():
            val[:] = X[offset : offset + val.shape[0], ...]
            offset += val.shape[0]

    def _repr_pretty_(self, p, *_):
        for key, val in self._data.items():
            p.text("key: ")
            p.pretty(key)
            p.text("\n")
            p.pretty(val)
            p.text("\n")

    def __repr__(self):
        for key in self._data:
            print(key, self._data[key])

    @staticmethod
    def ones_like(x, *args):
        out = type(x)()
        for k in x._data.keys():
            out[k] = np.ones_like(x[k], *args)
        return out

    @staticmethod
    def zeros_like(x, *args):
        out = type(x)()
        for k in x._data.keys():
            out[k] = np.zeros_like(x[k], *args)
        return out

    __lmul__ = __mul__
    __rmul__ = __mul__
    __radd__ = __add__
    __ladd__ = __add__


class PwCoeffs(CoefficientArray):
    def __init__(self, kpointset=None):
        super().__init__()

        # load plane-wave coefficients from kpointset
        if kpointset is not None:
            num_sc = kpointset.ctx().num_spins()
            for ki in range(len(kpointset)):
                k = kpointset[ki]
                for ispn in range(num_sc):
                    key = ki, ispn
                    val = np.array(
                        k.spinor_wave_functions().pw_coeffs(ispn), order="F", copy=True
                    )
                    self.__setitem__(key, val)

    def __setitem__(self, key, item):
        """
        key -- (k, ispn)
        """
        # return as view
        assert len(key) == 2
        return super(PwCoeffs, self).__setitem__(key, item)

    def kview(self, k):
        """ """
        out = PwCoeffs()
        out._data = {
            (ki, ispn): self._data[(ki, ispn)] for ki, ispn in self._data if ki == k
        }
        return out

    def kvalues(self):
        """
        TODO: make an iterator
        """
        if len(self._data.keys()) > 0:
            ks, _ = zip(*self._data.keys())
        else:
            ks = []
        return ks

    def by_k(self):
        """
        returns a dictionary, where each element is a list of tuples:
        {k: [(ispn, cn), ...]}
        """
        sdict = {k: [] for k in self.kvalues()}
        for k, ispn in self._data:
            sdict[k].append((ispn, self._data[(k, ispn)]))
        return sdict


def allthreaded(f, return_type=PwCoeffs):
    """Decorator for threaded application over CoefficientArray. In contrast
    to `threaded`, all positional arguments of f are "threaded".
    """
    type_hints = get_type_hints(f)
    return_type_hint = type_hints["return"]
    nret = 1
    if hasattr(return_type_hint, "__origin__") and return_type_hint.__origin__ is tuple:
        nret = len(return_type_hint.__args__)

    def _f(*args):
        if nret > 1:
            # f is returning a tuple
            out = tuple(return_type() for _ in range(nret))
            for k in args[0]._data.keys():
                tmp = f(*map(lambda x: x.__getitem__(k), args))
                for out_n, tmp_n in zip(out, tmp):
                    out_n[k] = tmp_n
            return out
        else:
            # f is returning a single object
            out = return_type()
            for k in args[0]._data.keys():
                out[k] = f(*map(lambda x: x.__getitem__(k), args))
            return out

    return _f


if __name__ == "__main__":
    shapes_ = [(10, 12), (3, 100), (4, 80), (5, 60)]
    keys = [(1, 0), (1, 1), (2, 0), (2, 1)]

    CC = PwCoeffs()

    for k, sh in zip(keys, shapes_):
        print("k:", k)
        print("sh:", sh)
        CC[k] = np.random.rand(*sh)

    # scale
    CC = 4 * CC
    CC = CC + 1
    CC = 2 + CC + 1
    CC = CC - 1j
    CC = np.conj(CC)
    print("np.conj(CC) has type: ", type(CC))
    # not working
    # CC = abs(CC)

    CCv = CC.kview(1)
    for k, item in CC.by_k().items():
        print("list of k:", k, "has ", len(item), " entries")

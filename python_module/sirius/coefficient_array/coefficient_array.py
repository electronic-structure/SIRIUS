import numpy as np
from mpi4py import MPI
from scipy.sparse import dia_matrix

def threaded(f):
    """
    decorator for threaded application over CoefficientArray
    """
    def _f(x, *args, **kwargs):
        if isinstance(x, CoefficientArray):
            out = type(x)(dtype=x.dtype, ctype=x.ctype)
            for k in x._data.keys():
                out[k] = f(x[k], *args, **kwargs)
            return out
        else:
            return f(x, *args, **kwargs)
    return _f


def is_complex(x):
    if isinstance(x, CoefficientArray):
        return np.iscomplexobj(x.dtype())
    else:
        return np.iscomplexobj(x)


@threaded
def sort(x):
    return np.sort(x)


@threaded
def shape(x):
    return np.shape(x)


@threaded
def trace(x):
    return np.trace(x)


def diag(x):
    """
    TODO: make a check not to flatten a 2d matrix
    """
    if isinstance(x, CoefficientArray):
        out = type(x)(dtype=x.dtype, ctype=x.ctype)
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
        out = type(x)(dtype=x.dtype, ctype=x.ctype)
        for key, val in x._data.items():
            n = np.size(val)
            out[key] = dia_matrix((val, 0), shape=(n, n))
        return out
    n = np.size(x)
    return dia_matrix((x, 0), shape=(n, n))


def ones_like(x, dtype=None):
    """Numpy ones_like."""
    if isinstance(x, CoefficientArray):
        return CoefficientArray.ones_like(x, dtype=dtype)
    if dtype is None:
        dtype = x.dtype
    return np.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
    """Numpy zeros_like."""
    if isinstance(x, CoefficientArray):
        return CoefficientArray.zeros_like(x, dtype=dtype)
    if dtype is None:
        dtype = x.dtype
    return np.zeros_like(x, dtype=dtype)


def inner(a, b):
    """Complex inner product."""
    return (a * b.conjugate()).sum()


def l2norm(a):
    """L2-norm. """
    return np.sqrt(np.real(inner(a, a)))


def einsum(expr, *operands):
    """
    map einsum over elements of CoefficientArray
    """

    assert operands

    try:
        return np.einsum(expr, *operands)
    except (ValueError, TypeError):
        out = type(operands[0])(dtype=operands[0].dtype, ctype=np.array)
        for key in operands[0]._data.keys():
            out[key] = np.einsum(expr,
                                 *list(map(lambda x: x[key], operands)))
        return out


class CoefficientArray:
    """CoefficientArray class."""
    def __init__(self, dtype=np.complex, ctype=np.matrix):
        """
        dtype -- number type
        ctype -- container type (default np.matrix)
        """
        self.dtype = dtype
        self.ctype = ctype
        self._data = {}

    def __getitem__(self, key):
        """
        key -- (k, ispn)
        """

        # # return as view, not required
        # return self._data[key][:]

        # return as view
        return self._data[key]

    def __setitem__(self, key, item):
        """
        """
        if isinstance(key, slice):
            for k in self._data:
                self._data[key] = item
        else:
            if key in self._data:
                x = self._data[key]
                # make sure shapes don't change
                try:
                    # view, no copy needed
                    x[:] = self.ctype(item, copy=False)
                except TypeError:
                    # not a view, make a copy
                    self._data[key] = self.ctype(item)
            else:
                if isinstance(item, np.ndarray):
                    self._data[key] = self.ctype(item, dtype=self.dtype, copy=True)
                else:
                    self._data[key] = item

    def sum(self, **kwargs):
        """
        """
        from mpi4py import MPI
        loc_sum = np.array(
            sum([np.sum(v) for _, v in self.items()]), dtype=self.dtype)
        reduced = MPI.COMM_WORLD.allreduce(loc_sum, op=MPI.SUM)
        return np.asscalar(reduced)

    def log(self, **kwargs):
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key in self._data.keys():
            out[key] = np.log(self._data[key], **kwargs)
        return out

    def __mul__(self, other):
        """
        """
        if is_complex(self) or is_complex(other):
            dtype = np.complex
        else:
            dtype = np.float

        out = type(self)(dtype=dtype, ctype=self.ctype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum('...,...->...', self._data[key],
                                     other._data[key])
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] * other
        else:
            raise TypeError('wrong type')
        return out

    def __truediv__(self, other):
        if is_complex(self) or is_complex(other):
            dtype = np.complex
        else:
            dtype = np.float

        out = type(self)(dtype=dtype, ctype=self.ctype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum('...,...->...', self._data[key],
                                     1 / other._data[key])
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] / other
        else:
            raise TypeError('wrong type')
        return out

    def __rtruediv__(self, other):
        if is_complex(self) or is_complex(other):
            dtype = np.complex
        else:
            dtype = np.float

        out = type(self)(dtype=dtype, ctype=self.ctype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum('...,...->...', 1 / self._data[key],
                                     other._data[key])
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = other / self._data[key]
        else:
            raise TypeError('wrong type')
        return out

    def __matmul__(self, other):
        """
        """
        if is_complex(self) or is_complex(other):
            dtype = np.complex
        else:
            dtype = np.float
        out = type(self)(dtype=dtype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = self._data[key] @ other._data[key]
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] @ other
        else:
            raise TypeError('wrong type')
        return out

    def __add__(self, other):
        if is_complex(self) or is_complex(other):
            dtype = np.complex
        else:
            dtype = np.float

        out = type(self)(dtype=dtype, ctype=self.ctype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = self._data[key] + other._data[key]
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] + other
        return out

    def __neg__(self):
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key, val in self._data.items():
            out[key] = -val
        return out

    def abs(self):
        out = type(self)(dtype=np.float, ctype=self.ctype)
        for key in self._data.keys():
            out[key] = np.abs(self._data[key])
        return out

    def sqrt(self):
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key in self._data.keys():
            out[key] = np.sqrt(self._data[key])
        return out

    def svd(self, **args):
        """
        returns U, s, Vh
        """
        U = type(self)(dtype=self.dtype, ctype=self.ctype)
        s = type(self)(dtype=np.float64, ctype=np.array)
        Vh = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key in self._data.keys():
            Ul, sl, Vhl = np.linalg.svd(self[key], **args)
            U[key] = Ul
            s[key] = sl
            Vh[key] = Vhl
        return U, s, Vh

    def eigh(self, **args):
        w = type(self)(dtype=np.float64, ctype=np.array)
        V = type(self)(dtype=self.dtype, ctype=np.matrix)
        for key in self._data.keys():
            w[key], V[key] = np.linalg.eigh(self[key], **args)
        return w, V

    def eig(self, **args):
        w = type(self)(dtype=np.complex, ctype=np.array)
        V = type(self)(dtype=self.dtype, ctype=np.matrix)
        for key in self._data.keys():
            w[key], V[key] = np.linalg.eig(self[key], **args)
        return w, V

    def qr(self, **args):
        Q = type(self)(dtype=self.dtype, ctype=np.matrix)
        R = type(self)(dtype=self.dtype, ctype=np.matrix)
        for key in self._data.keys():
            Q[key], R[key] = np.linalg.qr(self[key], **args)
        return Q, R

    def keys(self):
        return self._data.keys()

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return -(self.__add__(-1 * other))

    def __pow__(self, a):
        """

        """
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key in self._data.keys():
            out[key] = self._data[key]**a
        return out

    def conjugate(self):
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key, val in self._data.items():
            out[key] = np.array(np.conj(val), copy=True)
        return out

    def conj(self):
        return self.conjugate()

    def flatten(self, ctype=None):
        if ctype is None:
            ctype = self.ctype
        out = type(self)(dtype=self.dtype, ctype=ctype)
        for key, val in self._data.items():
            out[key] = ctype(val).flatten()
        return out

    def asarray(self):
        """

        """
        out = type(self)(dtype=self.dtype, ctype=np.array)
        for key, val in self._data.items():
            out[key] = np.array(val)
        return out

    def cols(self, indices):
        """

        """
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for k, jj in indices._data.items():
            out[k] = self[k][:, jj]
        return out

    def __len__(self):
        """
        """
        return len(self._data)

    def items(self):
        """
        """
        return self._data.items()

    def __contains__(self, key):
        """
        """
        return self._data.__contains(key)

    @property
    def real(self):
        out = type(self)(dtype=np.double)
        for key, val in self._data.items():
            out[key] = np.real(val)
        return out

    @property
    def imag(self):
        out = type(self)(dtype=np.double)
        for key, val in self._data.items():
            out[key] = np.imag(val)
        return out

    @property
    def H(self):
        """
        Hermitian conjugate
        """
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key, val in self._data.items():
            out[key] = np.atleast_2d(val).H
        return out

    @property
    def T(self):
        """
        Tranpose
        """
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        for key, val in self._data.items():
            out[key] = np.atleast_2d(val).T
        return out

    def __str__(self):
        return '\n'.join([
            '\n'.join(map(str, [key, val, '---']))
            for key, val in self._data.items()
        ])

    def to_array(self):
        """
        convert to numpy array
        """
        if len(self) > 0:
            return np.concatenate(list(map(np.atleast_1d, self._data.values())),
                                  axis=0)
        else:
            return np.array([])

    def from_array(self, X):
        """
        set internal data from numpy array
        assumes that each entry has the same number of columns
        """
        offset = 0
        for key, val in self._data.items():
            val[:] = X[offset:offset + val.shape[0], ...]
            offset += val.shape[0]

    def _repr_pretty_(self, p, cycle):
        for key, val in self._data.items():
            p.text('key: ')
            p.pretty(key)
            p.text('\n')
            p.pretty(val)
            p.text('\n')

    @staticmethod
    def ones_like(x, dtype=None, ctype=None):
        if ctype is None:
            ctype = x.ctype
        if dtype is None:
            dtype = x.dtype
        out = type(x)(dtype=dtype, ctype=ctype)
        for k in x._data.keys():
            out[k] = np.ones_like(x[k], dtype=dtype)
        return out

    @staticmethod
    def zeros_like(x, dtype=None, ctype=None):
        if ctype is None:
            ctype = x.ctype
        if dtype is None:
            dtype = x.dtype
        out = type(x)(dtype=dtype, ctype=ctype)
        for k in x._data.keys():
            out[k] = np.zeros_like(x[k], dtype=dtype)
        return out


    __lmul__ = __mul__
    __rmul__ = __mul__
    __radd__ = __add__
    __ladd__ = __add__


class PwCoeffs(CoefficientArray):
    def __init__(self, kpointset=None, dtype=np.complex, ctype=np.matrix):
        super().__init__(dtype=dtype, ctype=ctype)

        # load plane-wave coefficients from kpointset
        if kpointset is not None:
            num_sc = kpointset.ctx().num_spins()
            for ki in range(len(kpointset)):
                k = kpointset[ki]
                for ispn in range(num_sc):
                    key = ki, ispn
                    val = np.matrix(k.spinor_wave_functions().pw_coeffs(ispn))
                    self.__setitem__(key, val)

    def __setitem__(self, key, item):
        """
        key -- (k, ispn)
        """
        # return as view
        assert (len(key) == 2)
        return super(PwCoeffs, self).__setitem__(key, item)

    def kview(self, k):
        """
        """
        out = PwCoeffs(dtype=self.dtype)
        out._data = {(ki, ispn): self._data[(ki, ispn)]
                     for ki, ispn in self._data if ki == k}
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


if __name__ == '__main__':
    shapes = [(10, 12), (3, 100), (4, 80), (5, 60)]
    keys = [(1, 0), (1, 1), (2, 0), (2, 1)]

    CC = PwCoeffs()

    for k, sh in zip(keys, shapes):
        print('k:', k)
        print('sh:', sh)
        CC[k] = np.random.rand(*sh)

    # scale
    CC = 4 * CC
    CC = CC + 1
    CC = 2 + CC + 1
    CC = CC - 1j
    CC = np.conj(CC)
    print('np.conj(CC) has type: ', type(CC))
    # not working
    # CC = abs(CC)

    CCv = CC.kview(1)
    for k, item in CC.by_k().items():
        print('list of k:', k, 'has ', len(item), ' entries')

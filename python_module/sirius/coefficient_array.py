import numpy as np


def inner(a, b):
    """
    complex inner product
    """
    try:
        return np.sum(
            np.array(a, copy=False) * np.array(np.conj(b), copy=False))
    except ValueError:
        # is of type CoefficientArray (cannot convert to array)
        return np.sum(a * np.conj(b), copy=False)


def einsum(expr, a, b):
    """
    map einsum over elements of CoefficientArray
    """
    try:
        return np.einsum(expr, a, b)
    except ValueError:
        out = type(a)(dtype=a.dtype)
        for key in a._data.keys():
            out[key] = np.einsum(expr, a[key], b[key])
        return out


class CoefficientArray:
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
            sum([np.sum(v) for _, v in self.items()]), dtype=np.complex128)
        rcvBuf = np.array(0.0, dtype=np.complex128)

        MPI.COMM_WORLD.Allreduce([loc_sum, MPI.DOUBLE_COMPLEX],
                                 [rcvBuf, MPI.DOUBLE_COMPLEX],
                                 op=MPI.SUM)
        return np.asscalar(rcvBuf)

    def __mul__(self, other):
        """
        Returns a new object of type type(self)
        """
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
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
        out = type(self)(dtype=self.dtype)
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

    def __matmul__(self, other):
        """
        TODO
        """
        # TODO: complex | double -> complex, double | double -> double
        out = type(self)(dtype=np.complex)
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
        out = type(self)(dtype=self.dtype, ctype=self.ctype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = self._data[key] + other._data[key]
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] + other
        return out

    def __neg__(self):
        out = type(self)(dtype=self.dtype)
        for key, val in self._data.items():
            out[key] = -val
        return out

    def abs(self):
        out = type(self)(dtype=self.dtype)
        for key in self._data.keys():
            out[key] = np.abs(self._data[key])
        return out

    def keys(self):
        return self._data.keys()

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def conjugate(self):
        out = type(self)(dtype=self.dtype)
        for key, val in self._data.items():
            out[key] = np.conj(val)
        return out

    def conj(self):
        return self.conjugate()

    def flatten(self, ctype=None):
        if ctype is None:
            ctype = self.ctype
        out = type(self)(dtype=np.double, ctype=ctype)
        for key, val in self._data.items():
            out[key] = ctype(val).flatten()
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
        out = type(self)(dtype=self.dtype)
        for key, val in self._data.items():
            out[key] = val.H
        return out

    @property
    def T(self):
        """
        Tranpose
        """
        out = type(self)(dtype=self.dtype)
        for key, val in self._data.items():
            out[key] = val.T
        return out

    def __str__(self):
        return '\n'.join([
            '\n'.join(map(str, [key, val, '---']))
            for key, val in self._data.items()
        ])

    def _repr_pretty_(self, p, cycle):
        for key, val in self._data.items():
            p.text('key: ')
            p.pretty(key)
            p.text('\n')
            p.pretty(val)
            p.text('\n')


    __lmul__ = __mul__
    __rmul__ = __mul__
    __radd__ = __add__
    __ladd__ = __add__
    __lsub__ = __sub__
    __rsub__ = __sub__


class PwCoeffs(CoefficientArray):
    def __init__(self, kpointset=None, dtype=np.complex, ctype=np.matrix):
        super().__init__(dtype=dtype, ctype=ctype)

        # load plane wave-coefficients from kpointset
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

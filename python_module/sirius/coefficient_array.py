import numpy as np


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
        # return as view
        return self._data[key][:]

    def __setitem__(self, key, item):
        """
        """
        if key in self._data:
            x = self._data[key]
            # make sure shapes don't change
            x[:] = self.ctype(item, copy=False)
        else:
            self._data[key] = self.ctype(item, dtype=self.dtype, copy=True)

    def sum(self, **kwargs):
        """
        """
        return sum([np.sum(v) for _, v in self.items()])

    def __mul__(self, other):
        """
        Returns a new object of type type(self)
        """
        out = type(self)(dtype=self.dtype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = np.einsum('ij,ij->ij', self._data[key], other._data[key])
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] * other
        else:
            raise TypeError('wrong type')
        return out

    def __add__(self, other):
        """
        """
        out = type(self)(dtype=self.dtype)
        if isinstance(other, CoefficientArray):
            for key in other._data.keys():
                out[key] = self._data[key] + other._data[key]
        elif np.isscalar(other):
            for key in self._data.keys():
                out[key] = self._data[key] + other
        return out

    def abs(self):
        """
        """
        out = type(self)(dtype=self.dtype)
        for key in self._data.keys():
            out[key] = np.abs(self._data[key])
        return out

    def keys(self):
        return self._data.keys()

    def __sub__(self, other):
        """
        """
        return self.__add__(-1*other)

    def conjugate(self):
        """
        """
        out = type(self)(dtype=self.dtype)
        for key, val in self._data.items():
            out[key] = np.conj(val)
        return out

    def conj(self):
        """
        """
        return self.conjugate()

    __lmul__ = __mul__
    __rmul__ = __mul__
    __radd__ = __add__
    __ladd__ = __add__
    __lsub__ = __sub__
    __rsub__ = __sub__


class PwCoeffs(CoefficientArray):
    def __init__(self, kpointset=None, dtype=np.complex):
        super().__init__(dtype)

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
        ks, _ = zip(*self._data.keys())
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

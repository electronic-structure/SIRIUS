import numpy as np


class CGHistNeugebaur:
    """
    Store CG variables for every iteration
    """

    types = [('iter', 'i4'),
             ('kappa', 'f8'),
             ('resX', 'f8'),
             ('resF', 'f8'),
             ('F', 'f8'),
             ('gamma', 'f8'),
             ('steplen', 'f8')]

    def __init__(self):
        self.data = np.array([], dtype=[('iter', 'i4'), ('kappa', 'f8'),
                                        ('resX', 'f8'), ('resF', 'f8'),
                                        ('F', 'f8'), ('gamma', 'f8')])

    def __call__(self, it, kappa, resX, resF, F, gamma):
        new = np.array((it, kappa, resX, resF, F, gamma),
                       dtype=CGHistNeugebaur.types)
        self.data = np.concatenate((self.data, new), axis=0)

    def __item__(self, key):
        return self.data[key]

    @property
    def iter(self):
        return self.data['iter']

    @property
    def kappa(self):
        return self.data['kappa']

    @property
    def F(self):
        return self.data['F']

    @property
    def resX(self):
        return self.data['resX']

    @property
    def resF(self):
        return self.data['resF']

    @property
    def gamma(self):
        return self.data['gamma']

    @property
    def steplen(self):
        return self.data['steplen']

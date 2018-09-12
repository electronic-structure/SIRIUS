

def make_dict(ctx, ks, x_ticks, x_axis):
    dict = {}
    dict["header"] = {}
    dict["header"]["x_axis"] = x_axis
    dict["header"]["x_ticks"] = []
    dict["header"]["num_bands"] = ctx.num_bands()
    dict["header"]["num_mag_dims"] = ctx.num_mag_dims()

    for e in enumerate(x_ticks):
        j = {}
        j["x"] = e[1][0]
        j["label"] = e[1][1]
        dict["header"]["x_ticks"].append(j)

    dict["bands"] = []

    for ik in range(len(ks)):
        bnd_k = {}
        bnd_k["kpoint"] = [0.0, 0.0, 0.0]
        for x in range(3):
            bnd_k["kpoint"][x] = ks(ik).vk()(x)
        bnd_e = []

        bnd_e = ks.get_band_energies(ik, 0)

        bnd_k["values"] = bnd_e
        dict["bands"].append(bnd_k)
    return dict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Logger:
    __metaclass__ = Singleton

    def __init__(self, fout=None, comm=None, all_print=False):
        from mpi4py import MPI
        self.fout = fout
        self._all_print = all_print
        if self.fout is not None:
            with open(self.fout, 'w'):
                print('')
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

    def print(self, *args):
        """

        """
        if self.comm.rank == 0 or self._all_print:
            if self.fout is not None:
                with open(self.fout, 'a') as fh:
                    print(*args, file=fh)
            else:
                print(*args)

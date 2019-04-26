from mpi4py import MPI

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
        self.fout = fout
        self._all_print = all_print
        if self.fout is not None:
            with open(self.fout, 'w'):
                print('')
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

    def log(self, arg1, *args):
        """

        """
        if self.comm.rank == 0 or self._all_print:
            if self.fout is not None:
                with open(self.fout, 'a') as fh:
                    print(arg1, *args, file=fh)
                    print(arg1, *args, file=sys.stdout)
            else:
                print(arg1, *args)
        elif self._all_print:
            print(arg1, *args)

    def __call__(self, *args):
        self.log(*args)

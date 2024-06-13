from mpi4py import MPI
import sys


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger:
    __metaclass__ = Singleton

    def __init__(self, fout=None, comm=None):
        self.fout = fout
        if self.fout is not None:
            with open(self.fout, "w"):
                print("")
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

    def log(self, arg1, *args, all_print=False):
        """ """
        if self.comm.rank == 0 or all_print:
            if self.fout is not None:
                with open(self.fout, "a") as fh:
                    print(arg1, *args, file=fh, flush=True)
                    print(arg1, *args, file=sys.stdout, flush=True)
            else:
                print(arg1, *args)
        elif all_print:
            print(arg1, *args)
        sys.stdout.flush()

    def __call__(self, *args, all_print=False):
        self.log(*args, all_print=all_print)

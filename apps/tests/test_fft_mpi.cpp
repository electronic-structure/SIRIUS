#include <sirius.h>
#include <thread>

using namespace sirius;

void test_fft_mpi(vector3d<int>& dims__)
{
    printf("test of threaded FFTs (OMP version)\n");

    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    Communicator comm(MPI_COMM_WORLD);

    MPI_FFT3D fft(dims__, 1, comm);

    int num_phi = 160;

    Timer t1("fft_mpi");
    for (int i = 0; i < num_phi; i++)
    {
        fft.transform(1);
        fft.transform(-1);
    }
    double tval = t1.stop();

    printf("performance: %f, (FFT/sec.)\n", 2 * num_phi / tval);

}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");

    Platform::initialize(1);

    test_fft_mpi(dims);

    Platform::finalize();
}

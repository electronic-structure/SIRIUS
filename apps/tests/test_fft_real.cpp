#include <sirius.h>
#include <thread>
#include <wave_functions.h>

using namespace sirius;

void test_fft_real(vector3d<int> const& dims__, double cutoff__)
{
    Communicator comm(MPI_COMM_WORLD);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(dims__, Platform::max_num_threads(), comm, CPU);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), comm, 1, false, false);
    Gvec gvec_r(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), comm, 1, false, true);

    printf("num_gvec: %i, num_gvec_reduced: %i\n", gvec.num_gvec(), gvec_r.num_gvec());
    printf("num_gvec_loc: %i %i\n", gvec.num_gvec(comm.rank()), gvec_r.num_gvec(comm.rank()));

    mdarray<double_complex, 1> phi(gvec_r.num_gvec_fft());
    for (int i = 0; i < gvec_r.num_gvec_fft(); i++) phi(i) = type_wrapper<double_complex>::random();
    phi(0) = 1.0;
    fft.transform<1>(gvec_r, &phi(0));

    mdarray<double_complex, 1> phi1(gvec_r.num_gvec_fft());
    for (int i = 0; i < fft.local_size(); i++)
    {
        if (fft.buffer(i).imag() > 1e-10)
        {
            printf("function is not real at %i\n", i);
        }
    }
    fft.transform<-1>(gvec_r, &phi1(0));

    double diff = 0;
    for (int i = 0; i < gvec_r.num_gvec_fft(); i++)
    {
        diff += std::abs(phi(i) - phi1(i));
    }
    printf("diff: %18.12f\n", diff);
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");
    double cutoff = args.value<double>("cutoff", 1);

    Platform::initialize(1);

    test_fft_real(dims, cutoff);
    
    Timer::print();

    Platform::finalize();
}

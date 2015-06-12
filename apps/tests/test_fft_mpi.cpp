#include <sirius.h>
#include <thread>

using namespace sirius;

void test_fft_mpi_correctness(vector3d<int>& dims__)
{
    Communicator comm(MPI_COMM_WORLD);

    double a1[] = {10, 2, 0};
    double a2[] = {0, 8, 0};
    double a3[] = {-8, 0, 10};

    Simulation_parameters p;
    Unit_cell uc(p, comm);
    uc.set_lattice_vectors(a1, a2, a3);

    auto& rlv = uc.reciprocal_lattice_vectors();

    MPI_FFT3D fft(dims__, 1, comm);
    fft.init_gvec(5.0, rlv);

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    std::vector<double_complex> pw_coefs(fft.num_gvec());
    
    if (comm.rank() == 0) printf("num_gvec: %i\n", fft.num_gvec());

    for (int ig = 0; ig < std::min(40, fft.num_gvec()); ig++)
    {
        memset(&pw_coefs[0], 0, pw_coefs.size() * sizeof(double_complex));
        auto gvec = fft.gvec(ig);

        if (comm.rank() == 0) printf("G: %i %i %i\n", gvec[0], gvec[1], gvec[2]);

        pw_coefs[ig] = 1.0;
        fft.input_pw(fft.num_gvec(), &pw_coefs[0]);
        fft.transform(1);

        mdarray<double_complex, 3> tmp(&fft.buffer(0), fft.size(0), fft.size(1), fft.local_size_z());

        double d = 0;
        // loop over 3D array (real space)
        for (int j0 = 0; j0 < fft.size(0); j0++)
        {
            for (int j1 = 0; j1 < fft.size(1); j1++)
            {
                for (int j2 = 0; j2 < fft.local_size_z(); j2++)
                {
                    // get real space fractional coordinate
                    vector3d<double> fv(double(j0) / fft.size(0), 
                                        double(j1) / fft.size(1), 
                                        double(j2 + fft.offset_z()) / fft.size(2));
                    double_complex z = std::exp(twopi * double_complex(0.0, (fv * gvec)));
                    d += std::pow(std::abs(tmp(j0, j1, j2) - z), 2);
                    //printf("pos: %f %f %f, fft: %f %f   exp: %f %f\n", fv[0], fv[1], fv[2],
                    //       std::real(tmp(j0, j1, j2)), std::imag(tmp(j0, j1, j2)), std::real(z), std::imag(z));
                              
                }
            }
        }
        comm.allreduce(&d, 1);
        if (comm.rank() == 0) printf("difference: %12.6f\n", d);
    }
}


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

    //test_fft_mpi(dims);
    test_fft_mpi_correctness(dims);

    Platform::finalize();
}

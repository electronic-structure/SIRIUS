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
    auto gv = fft.init_gvec(vector3d<double>(0, 0, 0), 15.0, rlv);

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    std::vector<double_complex> pw_coefs(gv.num_gvec_);
    
    if (comm.rank() == 0) printf("num_gvec: %i\n", gv.num_gvec_);
    DUMP("num_gvec_loc: %i", gv.num_gvec_loc_);

    int n = (fft.size() < 100000) ? gv.num_gvec_ : std::min(50, gv.num_gvec_);

    for (int ig = 0; ig < n; ig++)
    {
        memset(&pw_coefs[0], 0, pw_coefs.size() * sizeof(double_complex));
        auto gvec = fft.gvec_by_index(gv.gvec_index_(ig));

        if (comm.rank() == 0) printf("G: %i %i %i\n", gvec[0], gvec[1], gvec[2]);

        pw_coefs[ig] = 1.0;
        fft.input_pw(gv.num_gvec_loc_, &gv.index_map_local_to_local_(0), &pw_coefs[gv.gvec_offset_]);
        fft.transform(1);

        mdarray<double_complex, 3> tmp(&fft.buffer(0), fft.size(0), fft.size(1), fft.local_size_z());

        double dmax = 0;
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
                    dmax = std::max(dmax, std::abs(tmp(j0, j1, j2) - z));
                }
            }
        }
        comm.allreduce<double, op_max>(&dmax, 1);
        if (dmax > 1e-12)
        {
            printf("maximum difference: %18.12f\n", dmax);
            exit(-1);
        }
    }
    printf("OK\n");
}


void test_fft_mpi_performance(vector3d<int>& dims__)
{
    matrix3d<double> reciprocal_lattice_vectors;
    for (int i = 0; i < 3; i++) reciprocal_lattice_vectors(i, i) = 1.0;

    Communicator comm(MPI_COMM_WORLD);

    MPI_FFT3D fft(dims__, 1, comm);

    int num_phi = 100;

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

    test_fft_mpi_correctness(dims);
    test_fft_mpi_performance(dims);

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    Platform::finalize();
}

#include <sirius.h>

using namespace sirius;

void test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__)
{
    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world); 
    
    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    FFT3D_grid fft_grid(2.01 * cutoff__, M);

    int num_fft_streams = 1;
    int num_threads_fft = omp_get_max_threads();
    if (mpi_grid_dims__[1] == 1) std::swap(num_fft_streams, num_threads_fft);

    FFT3D_context fft_ctx(mpi_grid, fft_grid, num_fft_streams, num_threads_fft, CPU);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_grid, mpi_grid.communicator(1 << 1), mpi_grid.communicator(1 << 0).size(), false, false);

    std::vector<double> pw_ekin(gvec.num_gvec(), 0);
    std::vector<double> veff(fft_ctx.fft()->local_size(), 1.0);

    if (mpi_comm_world.rank() == 0)
    {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.num_gvec(0));
        printf("FFT grid size: %i %i %i\n", fft_grid.size(0), fft_grid.size(1), fft_grid.size(2));
        printf("number of FFT streams: %i\n", num_fft_streams);
        printf("number of FFT threads: %i\n", num_threads_fft);
    }
    
    Hloc_operator hloc(fft_ctx, gvec, pw_ekin, veff);

    Wave_functions phi(num_bands__, gvec, mpi_grid, true);
    Wave_functions hphi(num_bands__, gvec, mpi_grid, true);

    hloc.apply(phi, hphi, 0, num_bands__);
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--help", "print this help and exit");
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--num_bands=", "{int} number of bands");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 2.0);
    auto num_bands = args.value<int>("num_bands", 10);

    Platform::initialize(1);
    test_hloc(mpi_grid_dims, cutoff, num_bands);
    mpi_comm_world.barrier();
    Timer::print();
    Platform::finalize();
}

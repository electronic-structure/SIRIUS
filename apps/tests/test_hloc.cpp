#include <sirius.h>

using namespace sirius;

void test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__,
               int num_fft_streams__, int num_threads_fft__, int use_gpu, double gpu_workload__)
{
    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world); 
    
    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    FFT3D_grid fft_grid(2.01 * cutoff__, M);

    FFT3D_context fft_ctx(mpi_grid, fft_grid, num_fft_streams__, num_threads_fft__,
                          static_cast<processing_unit_t>(use_gpu), gpu_workload__);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_grid, mpi_grid.communicator(1 << 1),
              mpi_grid.communicator(1 << 0).size(), false, false);

    std::vector<double> pw_ekin(gvec.num_gvec(), 0);
    std::vector<double> veff(fft_ctx.fft()->local_size(), 2.0);

    if (mpi_comm_world.rank() == 0)
    {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.num_gvec(0));
        printf("FFT grid size: %i %i %i\n", fft_grid.size(0), fft_grid.size(1), fft_grid.size(2));
        printf("number of FFT streams: %i\n", fft_ctx.num_fft_streams());
    }

    fft_ctx.allocate_workspace();
    
    Hloc_operator hloc(fft_ctx, gvec, pw_ekin, veff);

    Wave_functions<false> phi(4 * num_bands__, gvec, mpi_grid, static_cast<processing_unit_t>(use_gpu));
    for (int i = 0; i < 4 * num_bands__; i++)
    {
        for (int j = 0; j < phi.num_gvec_loc(); j++)
        {
            phi(j, i) = type_wrapper<double_complex>::random();
        }
    }
    Wave_functions<false> hphi(4 * num_bands__, num_bands__, gvec, mpi_grid, static_cast<processing_unit_t>(use_gpu));
    
    Timer t1("h_loc");
    for (int i = 0; i < 4; i++)
    {
        hphi.copy_from(phi, i * num_bands__, num_bands__);
        hloc.apply(hphi, i * num_bands__, num_bands__);
    }
    t1.stop();

    double diff = 0;
    for (int i = 0; i < 4 * num_bands__; i++)
    {
        for (int j = 0; j < phi.num_gvec_loc(); j++)
        {
            diff += std::abs(2.0 * phi(j, i) - hphi(j, i));
        }
    }
    mpi_comm_world.allreduce(&diff, 1);
    if (mpi_comm_world.rank() == 0)
    {
        printf("diff: %18.12f\n", diff);
    }

    fft_ctx.deallocate_workspace();
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--help", "print this help and exit");
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--num_fft_streams=", "{int} number of independent FFT streams");
    args.register_key("--num_threads_fft=", "{int} number of threads for each FFT");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--gpu_workload=", "{double} worload of GPU");

    args.parse_args(argn, argv);
    if (args.exist("help"))
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto num_fft_streams = args.value<int>("num_fft_streams", 1);
    auto num_threads_fft = args.value<int>("num_threads_fft", omp_get_max_threads());
    auto cutoff = args.value<double>("cutoff", 2.0);
    auto num_bands = args.value<int>("num_bands", 10);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto gpu_workload = args.value<double>("gpu_workload", 0.8);

    Platform::initialize(1);
    test_hloc(mpi_grid_dims, cutoff, num_bands, num_fft_streams, num_threads_fft, use_gpu, gpu_workload);
    mpi_comm_world.barrier();
    Timer::print();
    Platform::finalize();
}

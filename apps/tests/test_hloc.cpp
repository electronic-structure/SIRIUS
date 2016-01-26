#include <sirius.h>

using namespace sirius;

void test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__,
               int use_gpu, double gpu_workload__)
{
    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world()); 
    
    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    FFT3D_grid fft_grid(2.01 * cutoff__, M);

    FFT3D fft(fft_grid, mpi_grid.communicator(1 << 0), static_cast<processing_unit_t>(use_gpu), gpu_workload__);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_grid, mpi_grid.communicator(1 << 0),
              mpi_grid.dimension_size(1), false, false);

    std::vector<double> pw_ekin(gvec.num_gvec(), 0);
    std::vector<double> veff(fft.local_size(), 2.0);

    if (mpi_comm_world().rank() == 0)
    {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.num_gvec(0));
        printf("FFT grid size: %i %i %i\n", fft_grid.size(0), fft_grid.size(1), fft_grid.size(2));
        printf("number of FFT threads: %i\n", omp_get_max_threads());
        printf("number of FFT groups: %i\n", mpi_grid.dimension_size(1));
        printf("MPI grid: %i %i\n", mpi_grid.dimension_size(0), mpi_grid.dimension_size(1));
        printf("number of z-columns: %li\n", gvec.z_columns().size());
    }

    fft.prepare();
    
    Hloc_operator hloc(fft, gvec, veff);

    Wave_functions<false> phi(4 * num_bands__, gvec, mpi_grid, CPU);
    for (int i = 0; i < 4 * num_bands__; i++)
    {
        for (int j = 0; j < phi.num_gvec_loc(); j++)
        {
            phi(j, i) = type_wrapper<double_complex>::random();
        }
    }
    Wave_functions<false> hphi(4 * num_bands__, num_bands__, gvec, mpi_grid, CPU);
    
    mpi_comm_world().barrier();
    runtime::Timer t1("h_loc");
    for (int i = 0; i < 4; i++)
    {
        hphi.copy_from(phi, i * num_bands__, num_bands__);
        hloc.apply(0, hphi, i * num_bands__, num_bands__);
    }
    mpi_comm_world().barrier();
    t1.stop();

    double diff = 0;
    for (int i = 0; i < 4 * num_bands__; i++)
    {
        for (int j = 0; j < phi.num_gvec_loc(); j++)
        {
            diff += std::abs(2.0 * phi(j, i) - hphi(j, i));
        }
    }
    mpi_comm_world().allreduce(&diff, 1);
    if (mpi_comm_world().rank() == 0)
    {
        printf("diff: %18.12f\n", diff);
    }

    fft.dismiss();
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
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
    auto cutoff = args.value<double>("cutoff", 2.0);
    auto num_bands = args.value<int>("num_bands", 10);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto gpu_workload = args.value<double>("gpu_workload", 0.8);

    sirius::initialize(1);
    test_hloc(mpi_grid_dims, cutoff, num_bands, use_gpu, gpu_workload);
    mpi_comm_world().barrier();
    runtime::Timer::print();
    sirius::finalize();
}

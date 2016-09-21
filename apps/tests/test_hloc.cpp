#include <sirius.h>

using namespace sirius;

void test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__,
               int use_gpu__, double gpu_workload__)
{
    device_t pu = static_cast<device_t>(use_gpu__);
    Simulation_parameters params;
    params.set_processing_unit(pu);
    params.set_esm_type("ultrasoft_pseudopotential");

    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world()); 
    
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D_grid fft_box(2.01 * cutoff__, M);

    FFT3D fft(fft_box, mpi_grid.communicator(1 << 0), pu, gpu_workload__);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_grid.size(), mpi_grid.communicator(1 << 0), false);

    std::vector<double> veff(fft.local_size(), 2.0);

    if (mpi_comm_world().rank() == 0)
    {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("FFT grid size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
        printf("number of FFT threads: %i\n", omp_get_max_threads());
        printf("number of FFT groups: %i\n", mpi_grid.dimension_size(1));
        printf("MPI grid: %i %i\n", mpi_grid.dimension_size(0), mpi_grid.dimension_size(1));
        printf("number of z-columns: %i\n", gvec.num_zcol());
        if (use_gpu__) printf("GPU workload: %f\n", gpu_workload__);
    }

    fft.prepare(gvec.partition());
    
    Hloc_operator hloc(fft, gvec.partition(), mpi_grid.communicator(1 << 1), veff);

    wave_functions phi(params, mpi_comm_world(), gvec, 4 * num_bands__);
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.pw_coeffs().num_rows_loc(); j++) {
            phi.pw_coeffs().prime(j, i) = type_wrapper<double_complex>::random();
        }
    }
    wave_functions hphi(params, mpi_comm_world(), gvec, 4 * num_bands__);

    #ifdef __GPU
    if (pu == GPU) {
        phi.pw_coeffs().allocate_on_device();
        phi.pw_coeffs().copy_to_device(0, 4 * num_bands__);
        hphi.pw_coeffs().allocate_on_device();
    }
    #endif
    
    mpi_comm_world().barrier();
    runtime::Timer t1("h_loc");
    for (int i = 0; i < 4; i++)
    {
        hphi.copy_from(phi, i * num_bands__, num_bands__);
        #ifdef __GPU
        if (pu == GPU && !fft.gpu_only()) {
            hphi.pw_coeffs().copy_to_host(i * num_bands__, num_bands__);
        }
        #endif
        hloc.apply(0, hphi, i * num_bands__, num_bands__);
    }
    mpi_comm_world().barrier();
    t1.stop();

    #ifdef __GPU
    if (pu == GPU && fft.gpu_only()) {
        hphi.pw_coeffs().copy_to_host(0, 4 * num_bands__);
    }
    #endif

    double diff{0};
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.pw_coeffs().num_rows_loc(); j++) {
            diff += std::abs(2.0 * phi.pw_coeffs().prime(j, i) - hphi.pw_coeffs().prime(j, i));
        }
    }
    if (diff != diff) {
        TERMINATE("NaN");
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
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 2.0);
    auto num_bands = args.value<int>("num_bands", 10);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto gpu_workload = args.value<double>("gpu_workload", 0.8);

    sirius::initialize(1);
    for (int i = 0; i < 10; i++) {
        test_hloc(mpi_grid_dims, cutoff, num_bands, use_gpu, gpu_workload);
    }
    mpi_comm_world().barrier();
    runtime::Timer::print();
    runtime::Timer::print_all();
    sirius::finalize();
}

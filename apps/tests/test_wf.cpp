#include <sirius.h>

using namespace sirius;

void test_wf(std::vector<int> mpi_grid_dims__,
             double cutoff__,
             int num_bands__,
             int use_gpu__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims__[0], mpi_grid_dims__[1]);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    
    /* create FFT box */
    FFT3D_grid fft_box(2.01 * cutoff__, M);
    /* create G-vectors */
    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_box, mpi_comm_world().size(), mpi_comm_world(), false);
    /* parameters to pass to wave-functions */
    Simulation_parameters params;
    params.set_processing_unit(pu);
    params.set_esm_type("ultrasoft_pseudopotential");

    if (mpi_comm_world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("FFT grid size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
    }

    wave_functions phi(params, mpi_comm_world(), gvec, num_bands__);

    for (int i = 0; i < num_bands__; i++) {
        for (int j = 0; j < phi.pw_coeffs().num_rows_loc(); j++) {
            phi.pw_coeffs().prime(j, i) = type_wrapper<double_complex>::random();
        }
    }
    if (pu == GPU) {
        phi.pw_coeffs().allocate_on_device();
        phi.pw_coeffs().copy_to_device(0, num_bands__);
    }
    auto cs = phi.checksum(0, num_bands__);
    printf("checksum: %18.10f %18.10f\n", cs.real(), cs.imag());
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");

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

    sirius::initialize(1);
    test_wf(mpi_grid_dims, cutoff, num_bands, use_gpu);
    sirius::finalize();
}

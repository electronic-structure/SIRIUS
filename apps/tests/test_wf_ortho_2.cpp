#include <sirius.h>

using namespace sirius;

void test_wf_ortho(std::vector<int> mpi_grid_dims__,
                   double cutoff__,
                   int num_bands__,
                   int use_gpu__,
                   int bs__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims__[0], mpi_grid_dims__[1]);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    
    /* create G-vectors */
    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    int num_atoms = 10;
    auto nmt = [](int i) {
        return 20;
    };

    wave_functions phi(pu, gvec, num_atoms, nmt, 2 * num_bands__);
    wave_functions hphi(pu, gvec, num_atoms, nmt, 2 * num_bands__);
    wave_functions tmp(pu, gvec, num_atoms, nmt, 2 * num_bands__);
    
    phi.pw_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};
    phi.mt_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};
    hphi.pw_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};
    hphi.mt_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};

    dmatrix<double_complex> ovlp(2 * num_bands__, 2 * num_bands__, blacs_grid, bs__, bs__);
    
    orthogonalize<double_complex>(0, num_bands__, phi, hphi, ovlp, tmp);
    orthogonalize<double_complex>(num_bands__, num_bands__, phi, hphi, ovlp, tmp);

    inner(phi, 0, 2 * num_bands__, phi, 0, 2 * num_bands__, 0.0, ovlp, 0, 0);

    for (int j = 0; j < ovlp.num_cols_local(); j++) {
        for (int i = 0; i < ovlp.num_rows_local(); i++) {
            double_complex z = (ovlp.irow(i) == ovlp.icol(j)) ? ovlp(i, j) - 1.0 : ovlp(i, j);
            if (std::abs(z) > 1e-12) {
                TERMINATE("wrong overlap");
            }
        }
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    //args.register_key("--bs=", "{int} block size");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 8.0);
    auto use_gpu = args.value<int>("use_gpu", 0);
    //auto bs = args.value<int>("bs", 16);

    sirius::initialize(1);
    for (int bs = 1; bs < 16; bs++) {
        for (int i = 1; i < 30; i++) {
            test_wf_ortho(mpi_grid_dims, cutoff, i, use_gpu, bs);
        }
    }
    mpi_comm_world().barrier();
    sirius::finalize();
}

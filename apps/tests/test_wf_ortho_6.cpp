#include <sirius.h>

using namespace sirius;

void test_wf_ortho(BLACS_grid const& blacs_grid__,
                   double cutoff__,
                   int num_bands__,
                   int use_gpu__,
                   int bs__,
                   int num_mag_dims__)
{
    int nsp = (num_mag_dims__ == 0) ? 1 : 2;

    device_t pu = static_cast<device_t>(use_gpu__);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    
    /* create G-vectors */
    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("number of G-vectors: %i\n", gvec.num_gvec());
    }

    int num_atoms = 10;
    auto nmt = [](int i) {
        return 20;
    };

    experimental::Wave_functions phi(gvec, num_atoms, nmt, 2 * num_bands__, nsp);
    experimental::Wave_functions tmp(gvec, num_atoms, nmt, 2 * num_bands__, nsp);
    
    for (int ispn = 0; ispn < nsp; ispn++) {
        phi.pw_coeffs(ispn).prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};
        phi.mt_coeffs(ispn).prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};
    }
    //phi.mt_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};
    //hphi.mt_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};

    dmatrix<double_complex> ovlp(2 * num_bands__, 2 * num_bands__, blacs_grid__, bs__, bs__);

#ifdef __GPU
    if (pu == GPU) {
        for (int is = 0; is < nsp; is++) {
            phi.pw_coeffs(is).allocate_on_device();
            phi.pw_coeffs(is).copy_to_device(0, 2 * num_bands__);
            if (phi.has_mt()) {
                phi.mt_coeffs(is).allocate_on_device();
                phi.mt_coeffs(is).copy_to_device(0, 2 * num_bands__);
            }
            tmp.pw_coeffs(is).allocate_on_device();
        }
        ovlp.allocate(memory_t::device);
    }
#endif

    if (num_mag_dims__ == 3) {
        orthogonalize<double_complex, 0, 0>(pu, 2, {&phi}, 0, num_bands__, ovlp, tmp);
        orthogonalize<double_complex, 0, 0>(pu, 2, {&phi}, num_bands__, num_bands__, ovlp, tmp);
    } else {
        for (int ispn = 0; ispn < nsp; ispn++) {
            orthogonalize<double_complex, 0, 0>(pu, ispn, {&phi}, 0, num_bands__, ovlp, tmp);
            orthogonalize<double_complex, 0, 0>(pu, ispn, {&phi}, num_bands__, num_bands__, ovlp, tmp);
        }
    }

    int err{0};
    if (num_mag_dims__ == 3) {
        inner(pu, 2, phi, 0, 2 * num_bands__, phi, 0, 2 * num_bands__, ovlp, 0, 0);
    
        for (int j = 0; j < ovlp.num_cols_local(); j++) {
            for (int i = 0; i < ovlp.num_rows_local(); i++) {
                double_complex z = (ovlp.irow(i) == ovlp.icol(j)) ? ovlp(i, j) - 1.0 : ovlp(i, j);
                if (std::abs(z) > 1e-12) {
                    err = 1;
                }
            }
        }
    } else {
        for (int ispn = 0; ispn < nsp; ispn++) {
            inner(pu, ispn, phi, 0, 2 * num_bands__, phi, 0, 2 * num_bands__, ovlp, 0, 0);
    
            for (int j = 0; j < ovlp.num_cols_local(); j++) {
                for (int i = 0; i < ovlp.num_rows_local(); i++) {
                    double_complex z = (ovlp.irow(i) == ovlp.icol(j)) ? ovlp(i, j) - 1.0 : ovlp(i, j);
                    if (std::abs(z) > 1e-12) {
                        err = 1;
                    }
                }
            }
        }
    }
    if (err) {
        printf("\x1b[31m" "Failed\n" "\x1b[0m" "\n");
    } else {
        printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
    }
}

void call_test(std::vector<int> mpi_grid__,
               double cutoff__,
               int num_bands__,
               int use_gpu__,
               int bs__,
               int repeat__)
{
    int np = mpi_grid__[0] * mpi_grid__[1];
    BLACS_grid blacs_grid((np == 1) ? mpi_comm_self() : mpi_comm_world(), mpi_grid__[0], mpi_grid__[1]);
    for (int i = 0; i < repeat__; i++) {
        test_wf_ortho(blacs_grid, cutoff__, num_bands__, use_gpu__, bs__, 0);
        test_wf_ortho(blacs_grid, cutoff__, num_bands__, use_gpu__, bs__, 1);
        test_wf_ortho(blacs_grid, cutoff__, num_bands__, use_gpu__, bs__, 3);
    }
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--num_bands=", "{int} block size");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--repeat=", "{int} number of repeats");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 8.0);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto bs = args.value<int>("bs", 32);
    auto num_bands = args.value<int>("num_bands", 100);
    auto repeat = args.value<int>("repeat", 2);

    sirius::initialize(1);
    call_test(mpi_grid_dims, cutoff, num_bands, use_gpu, bs, repeat);
    mpi_comm_world().barrier();
    sddk::timer::print();
    sirius::finalize();
}

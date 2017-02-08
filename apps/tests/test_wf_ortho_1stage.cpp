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

    if (mpi_comm_world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
    }

    wave_functions phi(pu, gvec, num_bands__);
    wave_functions hphi(pu, gvec, num_bands__);
    wave_functions ophi(pu, gvec, num_bands__);
    wave_functions tmp(pu, gvec, num_bands__);
    tmp.pw_coeffs().prime().zero();

    for (int i = 0; i < num_bands__; i++) {
        for (int j = 0; j < phi.pw_coeffs().num_rows_loc(); j++) {
            phi.pw_coeffs().prime(j, i) = type_wrapper<double_complex>::random();
            hphi.pw_coeffs().prime(j, i) = phi.pw_coeffs().prime(j, i) * double(i + 1);
            ophi.pw_coeffs().prime(j, i) = phi.pw_coeffs().prime(j, i);
        }
    }
    dmatrix<double_complex> ovlp(num_bands__, num_bands__, blacs_grid, bs__, bs__);

    #ifdef __GPU
    if (pu == GPU) {
        phi.allocate_on_device();
        phi.copy_to_device(0, num_bands__);

        hphi.allocate_on_device();
        hphi.copy_to_device(0, num_bands__);
        
        ophi.allocate_on_device();
        ophi.copy_to_device(0, num_bands__);

        tmp.allocate_on_device();

        if (mpi_comm_world().size() == 1) {
            ovlp.allocate(memory_t::device);
        }
    }
    #endif

    mpi_comm_world().barrier();
    sddk::timer t1("ortho");
    orthogonalize<double_complex>(0, num_bands__, phi, hphi, ophi, ovlp, tmp);
    mpi_comm_world().barrier();
    double tval = t1.stop();

    int k = gvec.num_gvec();
    
    // one inner product
    long double flop1 = 8.0 * num_bands__ * num_bands__ * k;
    // one Cholesky + one inversion, inversion cost is half-Cholesky
    long double flop2 = 1.5 * (8.0 / 3) * num_bands__ * num_bands__ * num_bands__;
    // three transformations
    long double flop3 = 3.0 * 8.0 * num_bands__ * k * num_bands__;

    long double num_gflop = 1e-9 * (flop1 + flop2 + flop3);
    
    if (mpi_comm_world().rank() == 0) {
        printf("total performance            : %18.6Lf GFlop/s\n", num_gflop / tval);
        printf("average MPI rank performance : %18.6Lf GFlop/s\n", num_gflop / tval / mpi_comm_world().size());
    }

    inner(phi, 0, num_bands__, ophi, 0, num_bands__, ovlp, 0, 0);

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
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--repeat=", "{int} number of repeats");

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
    auto repeat = args.value<int>("repeat", 1);
    auto bs = args.value<int>("bs", 16);

    sirius::initialize(1);
    if (mpi_comm_world().rank() == 0) {
        printf("Running on %i x %i MPI grid\n", mpi_grid_dims[0], mpi_grid_dims[1]);
    }
    for (int i = 0; i < repeat; i++) {
        test_wf_ortho(mpi_grid_dims, cutoff, num_bands, use_gpu, bs);
    }
    mpi_comm_world().barrier();
    sddk::timer::print();
    //sddk::timer::print_all();
    sirius::finalize();
}

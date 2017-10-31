#include <sirius.h>

using namespace sirius;

void test_diag(BLACS_grid const& blacs_grid__,
               double cutoff__,
               int num_bands__,
               int use_gpu__,
               int bs__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    
    /* create G-vectors */
    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("number of G-vectors: %i\n", gvec.num_gvec());
    }

    wave_functions phi(pu, gvec, num_bands__);
    wave_functions hphi(pu, gvec, num_bands__);

    phi.pw_coeffs().prime() = [](int64_t i0, int64_t i1){return type_wrapper<double_complex>::random();};

    for (int i = 0; i < num_bands__; i++) {
        for (int ig = 0; ig < gvec.count(); ig++) {
            auto G = gvec.gkvec_cart(gvec.offset() + ig);
            hphi.pw_coeffs().prime(ig, i) = dot(G, G) * phi.pw_coeffs().prime(ig, i) * 0.5;
        }
    }

    dmatrix<double_complex> A(num_bands__ * 2, num_bands__ * 2, blacs_grid__, bs__, bs__);
    dmatrix<double_complex> B(num_bands__ * 2, num_bands__ * 2, blacs_grid__, bs__, bs__);
    dmatrix<double_complex> A_ref(num_bands__ * 2, num_bands__ * 2, blacs_grid__, bs__, bs__);
    dmatrix<double_complex> B_ref(num_bands__ * 2, num_bands__ * 2, blacs_grid__, bs__, bs__);
    dmatrix<double_complex> Z(num_bands__ * 2, num_bands__ * 2, blacs_grid__, bs__, bs__);

#ifdef __GPU
    if (pu == GPU) {
        phi.pw_coeffs().allocate_on_device();
        phi.pw_coeffs().copy_to_device(0, num_bands__);
        hphi.pw_coeffs().allocate_on_device();
        hphi.pw_coeffs().copy_to_device(0, num_bands__);
        A.allocate(memory_t::device);
        B.allocate(memory_t::device);
    }
#endif
    A.zero();
    B.zero();
    
    inner(phi, 0, num_bands__, hphi, 0, num_bands__, 0.0, A, 0, 0);
    A >> A_ref;
    inner(phi, 0, num_bands__, phi, 0, num_bands__, 0.0, B, 0, 0);
    B >> B_ref;

    Eigenproblem_elpa1 evp(blacs_grid__, bs__);

    experimental::Eigenproblem_base<double_complex>* evp1;
    //evp1 = new experimental::Eigenproblem_elpa1<double_complex>();
    evp1 = new experimental::Eigenproblem_lapack<double_complex>();
    //experimental::Eigenproblem_scalapack evp1;


    //Eigenproblem_scalapack evp(blacs_grid__, bs__, bs__, 1e-12);
    
    //int nev{50};
    int nev = num_bands__;
    std::vector<double> eval(num_bands__);

    if (B.blacs_grid().comm().rank() == 0) {
        printf("num_bands = %i\n", num_bands__);
        printf("nev = %i\n", nev);
        printf("bs = %i\n", bs__);
        printf("== calling eigensolver ==\n");
    }
    evp1->solve(num_bands__, nev, A, B, eval.data(), Z);

    //evp.solve(num_bands__, nev, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), eval.data(), Z.at<CPU>(), Z.ld(), A.num_rows_local(), A.num_cols_local()); 

    /* check residuals */
    if (B.blacs_grid().comm().rank() == 0) {
        printf("== gemm1: B*Z ==\n");
    }
    linalg<CPU>::gemm(0, 0, num_bands__, nev, num_bands__, double_complex(1, 0), B_ref, Z, double_complex(0, 0), B);
    for (int j = 0; j < B.num_cols_local(); j++) {
        for (int i = 0; i < B.num_rows_local(); i++) {
            if (B.icol(j) < num_bands__) {
                B(i, j) *= eval[B.icol(j)];
            }
        }
    }
    if (B.blacs_grid().comm().rank() == 0) {
        printf("== gemm2: AZ-lambda*B*Z ==\n");
    }
    linalg<CPU>::gemm(0, 0, num_bands__, nev, num_bands__, double_complex(1, 0), A_ref, Z, double_complex(-1, 0), B);

    double diff{0};
    for (int j = 0; j < B.num_cols_local(); j++) {
        for (int i = 0; i < B.num_rows_local(); i++) {
            diff += std::abs(B(i, j));
        }
    }
    B.blacs_grid().comm().allreduce(&diff, 1);
    if (B.blacs_grid().comm().rank() == 0) {
        printf("residual: %18.12f\n", diff);
    }
}

void call_test(std::vector<int> mpi_grid__,
               double cutoff__,
               int num_bands__,
               int use_gpu__,
               int bs__,
               int repeat__)
{
    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid__[0], mpi_grid__[1]);
    for (int i = 0; i < repeat__; i++) {
        test_diag(blacs_grid, cutoff__, num_bands__, use_gpu__, bs__);
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

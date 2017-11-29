#include <sirius.h>

using namespace sirius;

template <typename T>
dmatrix<T> random_symmetric(int N__, int bs__, BLACS_grid const& blacs_grid__)
{
    dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = type_wrapper<T>::random();
        }
    }

    linalg<CPU>::tranc(N__, N__, A, 0, 0, B, 0, 0);

    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = 0.5 * (A(i, j) + B(i, j));
        }
    }
    return std::move(A);
}

template <typename T>
dmatrix<T> random_positive_definite(int N__, int bs__, BLACS_grid const& blacs_grid__)
{
    dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__);
    dmatrix<T> B(N__, N__, blacs_grid__, bs__, bs__);
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = type_wrapper<T>::random();
        }
    }

    linalg<CPU>::tranc(N__, N__, A, 0, 0, B, 0, 0);
    linalg<CPU>::gemm(2, 0, N__, N__, N__, linalg_const<T>::one(), A, A, linalg_const<T>::zero(), B);

    return std::move(B);
}


template <typename T>
void test_diag(BLACS_grid const& blacs_grid__,
               int N__,
               int n__,
               int nev__,
               int bs__,
               bool test_gen__)
{
    dmatrix<T> A = random_symmetric<T>(N__, bs__, blacs_grid__);
    dmatrix<T> A_ref(N__, N__, blacs_grid__, bs__, bs__);
    A >> A_ref;

    dmatrix<T> Z(N__, N__, blacs_grid__, bs__, bs__);

    dmatrix<T> B;
    dmatrix<T> B_ref;
    if (test_gen__) {
        B = random_positive_definite<T>(N__, bs__, blacs_grid__);
        B_ref = dmatrix<T>(N__, N__, blacs_grid__, bs__, bs__);
        B >> B_ref;
    }

    auto solver = experimental::Eigenproblem_factory<T>(experimental::ev_solver_t::scalapack);
    
    std::vector<double> eval(nev__);

    if (blacs_grid__.comm().rank() == 0) {
        printf("N = %i\n", N__);
        printf("n = %i\n", n__);
        printf("nev = %i\n", nev__);
        printf("bs = %i\n", bs__);
        printf("== calling eigensolver ==\n");
    }
    if (test_gen__) {
        if (n__ == nev__) {
            solver->solve(n__, A, B, eval.data(), Z);
        } else {
            solver->solve(n__, nev__, A, B, eval.data(), Z);
        }
    } else {
        if (n__ == nev__) {
            solver->solve(n__, A, eval.data(), Z);
        } else {
            solver->solve(n__, nev__, A, eval.data(), Z);
        }
    }
    
    if (blacs_grid__.comm().rank() == 0) {
        printf("eigen-values (min, max): %18.12f %18.12f\n", eval.front(), eval.back());
    }

    /* check residuals */

    /* A = lambda * Z */
    for (int j = 0; j < Z.num_cols_local(); j++) {
        for (int i = 0; i < Z.num_rows_local(); i++) {
            if (Z.icol(j) < nev__) {
                A(i, j) = eval[Z.icol(j)] * Z(i, j);
            }
        }
    }
    if (test_gen__) {
        /* lambda * B * Z */
        linalg<CPU>::gemm(0, 0, n__, nev__, n__, linalg_const<T>::one(), B_ref, A, linalg_const<T>::zero(), B);
        B >> A;
    }
        //for (int j = 0; j < B.num_cols_local(); j++) {
        //    for (int i = 0; i < B.num_rows_local(); i++) {
        //        if (B.icol(j) < nev__) {
        //            B(i, j) *= eval[B.icol(j)];
        //        }
        //    }
        //}

    /* A * Z - lambda * B * Z */
    linalg<CPU>::gemm(0, 0, n__, nev__, n__, linalg_const<T>::one(), A_ref, Z, linalg_const<T>::m_one(), A);

    double diff{0};
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            if (A.icol(j) < nev__ && A.irow(i) < n__) {
                diff = std::max(diff, std::abs(A(i, j)));
            }
        }
    }
    blacs_grid__.comm().template allreduce<double, mpi_op_t::max>(&diff, 1);
    if (blacs_grid__.comm().rank() == 0) {
        printf("maximum difference: %22.18f\n", diff);
    }
    if (diff > 1e-10) {
        TERMINATE("wrong residual");
    }
}

void call_test(std::vector<int> mpi_grid__,
               int N__,
               int n__,
               int nev__,
               int bs__,
               bool test_gen__,
               int repeat__)
{
    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid__[0], mpi_grid__[1]);
    for (int i = 0; i < repeat__; i++) {
        //test_diag<double>(blacs_grid, N__, n__, nev__, bs__, test_gen__);
        test_diag<double_complex>(blacs_grid, N__, n__, nev__, bs__, test_gen__);
    }

}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--N=", "{int} total size of the matrix");
    args.register_key("--n=", "{int} size of the sub-matrix to diagonalize");
    args.register_key("--nev=", "{int} number of eigen-vectors");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--repeat=", "{int} number of repeats");
    args.register_key("--gen", "test generalized problem");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto N        = args.value<int>("N", 200);
    auto n        = args.value<int>("n", 100);
    auto nev      = args.value<int>("nev", 50);
    auto bs       = args.value<int>("bs", 32);
    auto repeat   = args.value<int>("repeat", 2);
    bool test_gen = args.exist("gen");

    sirius::initialize(1);
    call_test(mpi_grid_dims, N, n, nev, bs, test_gen, repeat);
    mpi_comm_world().barrier();
    sddk::timer::print();
    sirius::finalize();
}

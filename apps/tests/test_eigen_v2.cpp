#include "test.hpp"

using namespace sirius;

template <typename T>
double test_diag(BLACS_grid const& blacs_grid__,
               int N__,
               int n__,
               int nev__,
               int bs__,
               bool test_gen__,
               std::string name__,
               Eigensolver& solver)
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

    std::vector<double> eval(nev__);

    if (acc::num_devices() > 0) {
        A.allocate(memory_t::device);
        A.copy_to(memory_t::device);

        if (test_gen__) {
            B.allocate(memory_t::device);
            B.copy_to(memory_t::device);
        }
        Z.allocate(memory_t::device);
    }

    if (blacs_grid__.comm().rank() == 0) {
        printf("N = %i\n", N__);
        printf("n = %i\n", n__);
        printf("nev = %i\n", nev__);
        printf("bs = %i\n", bs__);
        printf("== calling %s ", name__.c_str());
        if (test_gen__) {
            printf("generalized ");
        }
        printf("eigensolver ==\n");
        if (std::is_same<T, double>::value) {
            printf("real data type\n");
        }
        if (std::is_same<T, double_complex>::value) {
            printf("complex data type\n");
        }
    }
    double t = -utils::wtime();
    if (test_gen__) {
        if (n__ == nev__) {
            solver.solve(n__, A, B, eval.data(), Z);
        } else {
            solver.solve(n__, nev__, A, B, eval.data(), Z);
        }
    } else {
        if (n__ == nev__) {
            solver.solve(n__, A, eval.data(), Z);
        } else {
            solver.solve(n__, nev__, A, eval.data(), Z);
        }
    }
    t += utils::wtime();

    if (blacs_grid__.comm().rank() == 0) {
        printf("eigen-values (min, max): %18.12f %18.12f\n", eval.front(), eval.back());
        printf("time: %f sec.\n", t);
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
        linalg(linalg_t::scalapack).gemm('N', 'N', n__, nev__, n__, &linalg_const<T>::one(), B_ref, 0, 0, A, 0, 0,
            &linalg_const<T>::zero(), B, 0, 0);
        B >> A;
    }

    /* A * Z - lambda * B * Z */
    linalg(linalg_t::scalapack).gemm('N', 'N', n__, nev__, n__, &linalg_const<T>::one(), A_ref, 0, 0, Z, 0, 0,
        &linalg_const<T>::m_one(), A, 0, 0);

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
    if (blacs_grid__.comm().rank() == 0) {
        if (diff > 1e-10) {
            printf("\x1b[31m" "Wrong residual\n" "\x1b[0m" "\n");
        } else {
            printf("\x1b[32m" "OK\n" "\x1b[0m" "\n");
        }
    }
    return t;
}

void test_diag2(BLACS_grid const& blacs_grid__,
                int bs__,
                std::string name__,
                std::string fname__)
{
    auto solver = Eigensolver_factory(get_ev_solver_t(name__));

    matrix<double_complex> full_mtrx;
    int n;
    if (blacs_grid__.comm().rank() == 0) {
        HDF5_tree h5(fname__, hdf5_access_t::read_only);
        h5.read("/nrow", &n, 1);
        int m;
        h5.read("/ncol", &m, 1);
        if (n != m) {
            TERMINATE("not a square matrix");
        }
        full_mtrx = matrix<double_complex>(n, n);
        h5.read("/mtrx", full_mtrx);
        blacs_grid__.comm().bcast(&n, 1, 0);
        blacs_grid__.comm().bcast(full_mtrx.at(memory_t::host), static_cast<int>(full_mtrx.size()), 0);
    } else {
        blacs_grid__.comm().bcast(&n, 1, 0);
        full_mtrx = matrix<double_complex>(n, n);
        blacs_grid__.comm().bcast(full_mtrx.at(memory_t::host), static_cast<int>(full_mtrx.size()), 0);
    }
    if (blacs_grid__.comm().rank() == 0) {
        printf("matrix size: %i\n", n);
    }

    std::vector<double> eval(n);
    dmatrix<double_complex> A(n, n, blacs_grid__, bs__, bs__);
    dmatrix<double_complex> Z(n, n, blacs_grid__, bs__, bs__);

    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            A(i, j) = full_mtrx(A.irow(i), A.icol(j));
        }
    }

    if (solver->solve(n, A, eval.data(), Z)) {
        TERMINATE("diagonalization failed");
    }
    if (blacs_grid__.comm().rank() == 0) {
        printf("lowest eigen-value: %18.12f\n", eval[0]);
    }
}

void call_test(std::vector<int> mpi_grid__,
               int N__,
               int n__,
               int nev__,
               int bs__,
               bool test_gen__,
               std::string name__,
               std::string fname__,
               int repeat__,
               int type__)
{
    auto solver = Eigensolver_factory(get_ev_solver_t(name__));
    BLACS_grid blacs_grid(Communicator::world(), mpi_grid__[0], mpi_grid__[1]);
    if (fname__.length() == 0) {
        Measurement m;
        for (int i = 0; i < repeat__; i++) {
            double t;
            if (type__ == 0) {
                t = test_diag<double>(blacs_grid, N__, n__, nev__, bs__, test_gen__, name__, *solver);
            } else {
                t = test_diag<double_complex>(blacs_grid, N__, n__, nev__, bs__, test_gen__, name__, *solver);
            }
            /* skip first "warmup" measurment */
            if (i) {
                m.push_back(t);
            }
        }
        if (blacs_grid.comm().rank() == 0) {
            printf("average time: %f (sec.), sigma: %f (sec.) \n", m.average(), m.sigma());
        }
    } else {
        test_diag2(blacs_grid, bs__, name__, fname__);
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
    args.register_key("--name=", "{string} name of the solver");
    args.register_key("--file=", "{string} input file name");
    args.register_key("--type=", "{int} data type: 0-real, 1-complex");

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
    auto type     = args.value<int>("type", 0);
    auto test_gen = args.exist("gen");
    auto name     = args.value<std::string>("name", "lapack");
    auto fname    = args.value<std::string>("file", "");

    sirius::initialize(1);
    call_test(mpi_grid_dims, N, n, nev, bs, test_gen, name, fname, repeat, type);
    sirius::finalize();
}

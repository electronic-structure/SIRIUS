#include "sirius.hpp"
#include "testing.hpp"

using namespace sirius;

template <typename T>
double
test_diag(sddk::BLACS_grid const& blacs_grid__, int N__, int n__, int nev__, int bs__, bool test_gen__, std::string name__,
          Eigensolver& solver)
{
    auto A_ref = random_symmetric<T>(N__, bs__, blacs_grid__);
    sddk::dmatrix<T> A(N__, N__, blacs_grid__, bs__, bs__, solver.host_memory_t());
    A_ref >> A;

    sddk::dmatrix<T> Z(N__, N__, blacs_grid__, bs__, bs__, solver.host_memory_t());

    sddk::dmatrix<T> B;
    sddk::dmatrix<T> B_ref;
    if (test_gen__) {
        B_ref = random_positive_definite<T>(N__, bs__, blacs_grid__);
        B = sddk::dmatrix<T>(N__, N__, blacs_grid__, bs__, bs__, solver.host_memory_t());
        B_ref >> B;
    }

    std::vector<double> eval(nev__);

    if (acc::num_devices() > 0) {
        A.allocate(sddk::memory_t::device);
        A.copy_to(sddk::memory_t::device);

        if (test_gen__) {
            B.allocate(sddk::memory_t::device);
            B.copy_to(sddk::memory_t::device);
        }
        Z.allocate(sddk::memory_t::device);
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
    if (blacs_grid__.comm().rank() == 0) {
        sirius::print_memory_usage(__FILE__, __LINE__, std::cout);
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
        sirius::print_memory_usage(__FILE__, __LINE__, std::cout);
    }

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
#if defined(SIRIUS_SCALAPACK)
        sddk::linalg(sddk::linalg_t::scalapack).gemm('N', 'N', n__, nev__, n__, &sddk::linalg_const<T>::one(),
                B_ref, 0, 0, A, 0, 0, &sddk::linalg_const<T>::zero(), B, 0, 0);
#else
        sddk::linalg(sddk::linalg_t::blas).gemm('N', 'N', n__, nev__, n__, &sddk::linalg_const<T>::one(),
                &B_ref(0, 0), B_ref.ld(), &A(0, 0), A.ld(), &sddk::linalg_const<T>::zero(), &B(0, 0), B.ld());
#endif
        B >> A;
    }

    /* A * Z - lambda * B * Z */
#if defined(SIRIUS_SCALAPACK)
    sddk::linalg(sddk::linalg_t::scalapack).gemm('N', 'N', n__, nev__, n__, &sddk::linalg_const<T>::one(), A_ref, 0, 0, Z, 0, 0,
        &sddk::linalg_const<T>::m_one(), A, 0, 0);
#else
    sddk::linalg(sddk::linalg_t::blas).gemm('N', 'N', n__, nev__, n__, &sddk::linalg_const<T>::one(), &A_ref(0, 0), A_ref.ld(),
            &Z(0, 0), Z.ld(), &sddk::linalg_const<T>::m_one(), &A(0, 0), A.ld());
#endif
    double diff{0};
    for (int j = 0; j < A.num_cols_local(); j++) {
        for (int i = 0; i < A.num_rows_local(); i++) {
            if (A.icol(j) < nev__ && A.irow(i) < n__) {
                diff = std::max(diff, std::abs(A(i, j)));
            }
        }
    }
    blacs_grid__.comm().template allreduce<double, sddk::mpi_op_t::max>(&diff, 1);
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

void test_diag2(sddk::BLACS_grid const& blacs_grid__,
                int bs__,
                std::string name__,
                std::string fname__)
{
    auto solver = Eigensolver_factory(name__);

    sddk::matrix<double_complex> full_mtrx;
    int n;
    if (blacs_grid__.comm().rank() == 0) {
        sddk::HDF5_tree h5(fname__, sddk::hdf5_access_t::read_only);
        h5.read("/nrow", &n, 1);
        int m;
        h5.read("/ncol", &m, 1);
        if (n != m) {
            TERMINATE("not a square matrix");
        }
        full_mtrx = sddk::matrix<double_complex>(n, n);
        h5.read("/mtrx", full_mtrx);
        blacs_grid__.comm().bcast(&n, 1, 0);
        blacs_grid__.comm().bcast(full_mtrx.at(sddk::memory_t::host), static_cast<int>(full_mtrx.size()), 0);
    } else {
        blacs_grid__.comm().bcast(&n, 1, 0);
        full_mtrx = sddk::matrix<double_complex>(n, n);
        blacs_grid__.comm().bcast(full_mtrx.at(sddk::memory_t::host), static_cast<int>(full_mtrx.size()), 0);
    }
    if (blacs_grid__.comm().rank() == 0) {
        printf("matrix size: %i\n", n);
    }

    std::vector<double> eval(n);
    sddk::dmatrix<double_complex> A(n, n, blacs_grid__, bs__, bs__);
    sddk::dmatrix<double_complex> Z(n, n, blacs_grid__, bs__, bs__);

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
    auto solver = Eigensolver_factory(name__);
    sddk::BLACS_grid blacs_grid(sddk::Communicator::world(), mpi_grid__[0], mpi_grid__[1]);
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
    cmd_args args(argn, argv, {
        {"mpi_grid_dims=", "{int int} dimensions of MPI grid"},
        {"N=", "{int} total size of the matrix"},
        {"n=", "{int} size of the sub-matrix to diagonalize"},
        {"nev=", "{int} number of eigen-vectors"},
        {"bs=", "{int} block size"},
        {"repeat=", "{int} number of repeats"},
        {"gen", "test generalized problem"},
        {"name=", "{string} name of the solver"},
        {"file=", "{string} input file name"},
        {"type=", "{int} data type: 0-real, 1-complex"}
    });

    auto mpi_grid_dims = args.value("mpi_grid_dims", std::vector<int>({1, 1}));
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

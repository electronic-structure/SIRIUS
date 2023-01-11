#include "eigenproblem.hpp"

#if defined(SIRIUS_ELPA)
#include <elpa/elpa.h>
#endif

namespace la {

#if defined(SIRIUS_ELPA)

Eigensolver_elpa::Eigensolver_elpa(int stage__)
    : Eigensolver(ev_solver_t::elpa, true, sddk::memory_t::host, sddk::memory_t::host)
    , stage_(stage__)
{
    if (!(stage_ == 1 || stage_ == 2)) {
        TERMINATE("wrong type of ELPA solver");
    }
}

void Eigensolver_elpa::initialize()
{
    if (elpa_init(20170403) != ELPA_OK) {
        TERMINATE("ELPA API version not supported");
    }
}

void Eigensolver_elpa::finalize()
{
    int ierr;
    elpa_uninit(&ierr);
}

/// Solve a generalized eigen-value problem for N lowest eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<double>& A__, la::dmatrix<double>& B__,
          double* eval__, la::dmatrix<double>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_gen");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        TERMINATE("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_gen|setup");

    int bs = A__.bs_row();

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    elpa_set_integer(handle, "na", matrix_size__, &error);
    elpa_set_integer(handle, "nev", nev__, &error);
    elpa_set_integer(handle, "local_nrows", A__.num_rows_local(), &error);
    elpa_set_integer(handle, "local_ncols", A__.num_cols_local(), &error);
    elpa_set_integer(handle, "nblk", bs, &error);
    elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(A__.blacs_grid().comm().native()), &error);
    elpa_set_integer(handle, "process_row", A__.blacs_grid().comm_row().rank(), &error);
    elpa_set_integer(handle, "process_col", A__.blacs_grid().comm_col().rank(), &error);
    elpa_set_integer(handle, "blacs_context", A__.blacs_grid().context(), &error);
    elpa_set_integer(handle, "omp_threads", nt, &error);
    //if (error != ELPA_OK) {
    //    TERMINATE("can't set elpa threads");
    //}
    if (acc::num_devices() != 0) {
        elpa_set_integer(handle, "gpu", 1, &error);
    }
    if (stage_ == 1) {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
    } else {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
    }
    elpa_setup(handle);
    PROFILE_STOP("Eigensolver_elpa|solve_gen|setup");

    auto& mph = get_memory_pool(sddk::memory_t::host);

    auto w = mph.get_unique_ptr<double>(matrix_size__);

    elpa_generalized_eigenvectors_d(handle, A__.at(sddk::memory_t::host), B__.at(sddk::memory_t::host),
        w.get(), Z__.at(sddk::memory_t::host), 0, &error);

    if (error != ELPA_OK) {
        elpa_deallocate(handle, &error);
        return 1;
    }

    elpa_deallocate(handle, &error);

    std::copy(w.get(), w.get() + nev__, eval__);

    if (nt != omp_get_max_threads()) {
        std::stringstream s;
        s << "number of OMP threads was changed by elpa" << std::endl
          << "  initial number of threads : " << nt << std::endl
          << "  new number of threads : " <<  omp_get_max_threads();
        TERMINATE(s);
    }

    return 0;

    //to_std(matrix_size__, A__, B__, Z__);

    ///* solve a standard problem */
    //int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
    //if (result) {
    //    return result;
    //}

    //bt(matrix_size__, nev__, A__, B__, Z__);
    //return 0;
}

/// Solve a generalized eigen-value problem for N lowest eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<std::complex<double>>& A__, la::dmatrix<std::complex<double>>& B__,
          double* eval__, la::dmatrix<std::complex<double>>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_gen");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        TERMINATE("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_gen|setup");

    int bs = A__.bs_row();

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    elpa_set_integer(handle, "na", matrix_size__, &error);
    elpa_set_integer(handle, "nev", nev__, &error);
    elpa_set_integer(handle, "local_nrows", A__.num_rows_local(), &error);
    elpa_set_integer(handle, "local_ncols", A__.num_cols_local(), &error);
    elpa_set_integer(handle, "nblk", bs, &error);
    elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(A__.blacs_grid().comm().native()), &error);
    elpa_set_integer(handle, "process_row", A__.blacs_grid().comm_row().rank(), &error);
    elpa_set_integer(handle, "process_col", A__.blacs_grid().comm_col().rank(), &error);
    elpa_set_integer(handle, "blacs_context", A__.blacs_grid().context(), &error);
    elpa_set_integer(handle, "omp_threads", nt, &error);
    //if (error != ELPA_OK) {
    //    TERMINATE("can't set elpa threads");
    //}
    if (acc::num_devices() != 0) {
        elpa_set_integer(handle, "gpu", 1, &error);
    }
    if (stage_ == 1) {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
    } else {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
    }
    elpa_setup(handle);
    PROFILE_STOP("Eigensolver_elpa|solve_gen|setup");

    auto& mph = get_memory_pool(sddk::memory_t::host);

    auto w = mph.get_unique_ptr<double>(matrix_size__);

    elpa_generalized_eigenvectors_dc(handle, A__.at(sddk::memory_t::host), B__.at(sddk::memory_t::host),
        w.get(), Z__.at(sddk::memory_t::host), 0, &error);

    if (error != ELPA_OK) {
        elpa_deallocate(handle, &error);
        return 1;
    }

    elpa_deallocate(handle, &error);

    std::copy(w.get(), w.get() + nev__, eval__);

    if (nt != omp_get_max_threads()) {
        std::stringstream s;
        s << "number of OMP threads was changed by elpa" << std::endl
          << "  initial number of threads : " << nt << std::endl
          << "  new number of threads : " <<  omp_get_max_threads();
        TERMINATE(s);
    }

    return 0;
    //to_std(matrix_size__, A__, B__, Z__);

    ///* solve a standard problem */
    //int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
    //if (result) {
    //    return result;
    //}

    //bt(matrix_size__, nev__, A__, B__, Z__);
    //return 0;
}

/// Solve a generalized eigen-value problem for all eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<double>& A__, la::dmatrix<double>& B__, double* eval__, la::dmatrix<double>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
}

/// Solve a generalized eigen-value problem for all eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<std::complex<double>>& A__, la::dmatrix<std::complex<double>>& B__, double* eval__,
          la::dmatrix<std::complex<double>>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
}

/// Solve a standard eigen-value problem for N lowest eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<double>& A__, double* eval__, la::dmatrix<double>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_std");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        TERMINATE("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_std|setup");

    int bs = A__.bs_row();

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    elpa_set_integer(handle, "na", matrix_size__, &error);
    elpa_set_integer(handle, "nev", nev__, &error);
    elpa_set_integer(handle, "local_nrows", A__.num_rows_local(), &error);
    elpa_set_integer(handle, "local_ncols", A__.num_cols_local(), &error);
    elpa_set_integer(handle, "nblk", bs, &error);
    elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(A__.blacs_grid().comm().native()), &error);
    elpa_set_integer(handle, "process_row", A__.blacs_grid().comm_row().rank(), &error);
    elpa_set_integer(handle, "process_col", A__.blacs_grid().comm_col().rank(), &error);
    elpa_set_integer(handle, "blacs_context", A__.blacs_grid().context(), &error);
    elpa_set_integer(handle, "omp_threads", nt, &error);
    //if (error != ELPA_OK) {
    //    TERMINATE("can't set elpa threads");
    //}
    if (acc::num_devices() != 0) {
        elpa_set_integer(handle, "gpu", 1, &error);
    }
    if (stage_ == 1) {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
    } else {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
    }
    elpa_setup(handle);
    PROFILE_STOP("Eigensolver_elpa|solve_std|setup");

    auto& mph = get_memory_pool(sddk::memory_t::host);
    auto w = mph.get_unique_ptr<double>(matrix_size__);

    elpa_eigenvectors_a_h_a_d(handle, A__.at(sddk::memory_t::host), w.get(), Z__.at(sddk::memory_t::host), &error);

    elpa_deallocate(handle, &error);

    std::copy(w.get(), w.get() + nev__, eval__);
    if (nt != omp_get_max_threads()) {
        std::stringstream s;
        s << "number of OMP threads was changed by elpa" << std::endl
          << "  initial number of threads : " << nt << std::endl
          << "  new number of threads : " <<  omp_get_max_threads();
        TERMINATE(s);
    }
    return 0;
}

/// Solve a standard eigen-value problem for N lowest eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<std::complex<double>>& A__, double* eval__,
          la::dmatrix<std::complex<double>>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_std");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        TERMINATE("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_std|setup");

    int bs = A__.bs_row();

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    elpa_set_integer(handle, "na", matrix_size__, &error);
    elpa_set_integer(handle, "nev", nev__, &error);
    elpa_set_integer(handle, "local_nrows", A__.num_rows_local(), &error);
    elpa_set_integer(handle, "local_ncols", A__.num_cols_local(), &error);
    elpa_set_integer(handle, "nblk", bs, &error);
    elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(A__.blacs_grid().comm().native()), &error);
    elpa_set_integer(handle, "process_row", A__.blacs_grid().comm_row().rank(), &error);
    elpa_set_integer(handle, "process_col", A__.blacs_grid().comm_col().rank(), &error);
    elpa_set_integer(handle, "blacs_context", A__.blacs_grid().context(), &error);
    elpa_set_integer(handle, "omp_threads", nt, &error);
    //if (error != ELPA_OK) {
    //    TERMINATE("can't set elpa threads");
    //}
    if (acc::num_devices() != 0) {
        elpa_set_integer(handle, "gpu", 1, &error);
    }
    if (stage_ == 1) {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
    } else {
        elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
    }
    elpa_setup(handle);
    PROFILE_STOP("Eigensolver_elpa|solve_std|setup");

    auto& mph = get_memory_pool(sddk::memory_t::host);
    auto w = mph.get_unique_ptr<double>(matrix_size__);

    auto A_ptr = A__.size_local() ? A__.at(sddk::memory_t::host) : nullptr;
    auto Z_ptr = Z__.size_local() ? Z__.at(sddk::memory_t::host) : nullptr;
    elpa_eigenvectors_a_h_a_dc(handle, A_ptr, w.get(), Z_ptr, &error);

    elpa_deallocate(handle, &error);

    std::copy(w.get(), w.get() + nev__, eval__);

    if (nt != omp_get_max_threads()) {
        std::stringstream s;
        s << "number of OMP threads was changed by elpa" << std::endl
          << "  initial number of threads : " << nt << std::endl
          << "  new number of threads : " <<  omp_get_max_threads();
        TERMINATE(s);
    }

    return 0;
}

/// Solve a standard eigen-value problem for all eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<double>& A__, double* eval__, la::dmatrix<double>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
}

/// Solve a standard eigen-value problem for all eigen-pairs.
int Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<std::complex<double>>& A__, double* eval__, la::dmatrix<std::complex<double>>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
}

#endif

}

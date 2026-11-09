/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <limits>
#include <type_traits>
#include "eigenproblem.hpp"

#if defined(SIRIUS_CUDA)
#include "core/acc/cusolver_api.hpp"
#endif

#if defined(SIRIUS_ELPA)
#include <elpa/elpa.h>
#endif

#if defined(SIRIUS_DLAF)
#include "core/la/dlaf.hpp"
#endif

namespace sirius {

namespace la {

#if defined(SIRIUS_ELPA)

template <typename M>
void
setup_handler(elpa_t& handle__, int stage__, M const& m__, int na__, int nev__)
{
    int error;
    int nt = omp_get_max_threads();

    elpa_set_integer(handle__, "na", na__, &error);
    elpa_set_integer(handle__, "nev", nev__, &error);
    elpa_set_integer(handle__, "local_nrows", m__.num_rows_local(), &error);
    elpa_set_integer(handle__, "local_ncols", m__.num_cols_local(), &error);
    elpa_set_integer(handle__, "nblk", m__.bs_row(), &error);
    elpa_set_integer(handle__, "mpi_comm_parent", MPI_Comm_c2f(m__.blacs_grid().comm().native()), &error);
    elpa_set_integer(handle__, "process_row", m__.blacs_grid().comm_row().rank(), &error);
    elpa_set_integer(handle__, "process_col", m__.blacs_grid().comm_col().rank(), &error);
    elpa_set_integer(handle__, "blacs_context", m__.blacs_grid().context(), &error);
    elpa_set_integer(handle__, "omp_threads", nt, &error);
    if (acc::num_devices() != 0) {
        elpa_set_integer(handle__, "nvidia-gpu", 1, &error);
    }
    if (stage__ == 1) {
        elpa_set_integer(handle__, "solver", ELPA_SOLVER_1STAGE, &error);
    } else {
        elpa_set_integer(handle__, "solver", ELPA_SOLVER_2STAGE, &error);
    }
    elpa_setup(handle__);
}

Eigensolver_elpa::Eigensolver_elpa(int stage__)
    : Eigensolver(ev_solver_t::elpa, true, memory_t::host, memory_t::host)
    , stage_(stage__)
{
    if (!(stage_ == 1 || stage_ == 2)) {
        RTE_THROW("wrong type of ELPA solver");
    }
}

void
Eigensolver_elpa::initialize()
{
    if (elpa_init(20170403) != ELPA_OK) {
        RTE_THROW("ELPA API version not supported");
    }
}

void
Eigensolver_elpa::finalize()
{
    int ierr;
    elpa_uninit(&ierr);
}

/// Solve a generalized eigen-value problem for N lowest eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<double>& A__, la::dmatrix<double>& B__,
                        double* eval__, la::dmatrix<double>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_gen");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        RTE_THROW("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_gen|setup");

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    setup_handler(handle, stage_, A__, matrix_size__, nev__);

    PROFILE_STOP("Eigensolver_elpa|solve_gen|setup");

    auto& mph = get_memory_pool(memory_t::host);

    auto w = mph.get_unique_ptr<double>(matrix_size__);

#if ELPA_API_VERSION < 20250131
    elpa_generalized_eigenvectors_d(handle, A__.at(memory_t::host), B__.at(memory_t::host), w.get(),
                                    Z__.at(memory_t::host), 0, &error);
#else
    elpa_generalized_eigenvectors(handle, A__.at(memory_t::host), B__.at(memory_t::host), w.get(),
                                  Z__.at(memory_t::host), 0, &error);
#endif

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
          << "  new number of threads : " << omp_get_max_threads();
        RTE_THROW(s);
    }

    return 0;

    // to_std(matrix_size__, A__, B__, Z__);

    ///* solve a standard problem */
    // int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
    // if (result) {
    //     return result;
    // }

    // bt(matrix_size__, nev__, A__, B__, Z__);
    // return 0;
}

/// Solve a generalized eigen-value problem for N lowest eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<std::complex<double>>& A__,
                        la::dmatrix<std::complex<double>>& B__, double* eval__, la::dmatrix<std::complex<double>>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_gen");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        RTE_THROW("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_gen|setup");

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    setup_handler(handle, stage_, A__, matrix_size__, nev__);

    PROFILE_STOP("Eigensolver_elpa|solve_gen|setup");

    auto& mph = get_memory_pool(memory_t::host);

    auto w = mph.get_unique_ptr<double>(matrix_size__);

#if ELPA_API_VERSION < 20250131
    elpa_generalized_eigenvectors_dc(handle, A__.at(memory_t::host), B__.at(memory_t::host), w.get(),
                                     Z__.at(memory_t::host), 0, &error);
#else
    elpa_generalized_eigenvectors(handle, A__.at(memory_t::host), B__.at(memory_t::host), w.get(),
                                  Z__.at(memory_t::host), 0, &error);
#endif

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
          << "  new number of threads : " << omp_get_max_threads();
        RTE_THROW(s);
    }

    return 0;
    // to_std(matrix_size__, A__, B__, Z__);

    ///* solve a standard problem */
    // int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
    // if (result) {
    //     return result;
    // }

    // bt(matrix_size__, nev__, A__, B__, Z__);
    // return 0;
}

/// Solve a generalized eigen-value problem for all eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<double>& A__, la::dmatrix<double>& B__, double* eval__,
                        la::dmatrix<double>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
}

/// Solve a generalized eigen-value problem for all eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<std::complex<double>>& A__,
                        la::dmatrix<std::complex<double>>& B__, double* eval__, la::dmatrix<std::complex<double>>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
}

/// Solve a standard eigen-value problem for N lowest eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<double>& A__, double* eval__,
                        la::dmatrix<double>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_std");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        RTE_THROW("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_std|setup");

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    setup_handler(handle, stage_, A__, matrix_size__, nev__);

    PROFILE_STOP("Eigensolver_elpa|solve_std|setup");

    auto& mph = get_memory_pool(memory_t::host);
    auto w    = mph.get_unique_ptr<double>(matrix_size__);

    elpa_eigenvectors_a_h_a_d(handle, A__.at(memory_t::host), w.get(), Z__.at(memory_t::host), &error);

    elpa_deallocate(handle, &error);

    std::copy(w.get(), w.get() + nev__, eval__);
    if (nt != omp_get_max_threads()) {
        std::stringstream s;
        s << "number of OMP threads was changed by elpa" << std::endl
          << "  initial number of threads : " << nt << std::endl
          << "  new number of threads : " << omp_get_max_threads();
        RTE_THROW(s);
    }
    return 0;
}

/// Solve a standard eigen-value problem for N lowest eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, ftn_int nev__, la::dmatrix<std::complex<double>>& A__, double* eval__,
                        la::dmatrix<std::complex<double>>& Z__)
{
    PROFILE("Eigensolver_elpa|solve_std");

    int nt = omp_get_max_threads();

    if (A__.num_cols_local() != Z__.num_cols_local()) {
        RTE_THROW("number of columns in A and Z don't match");
    }

    PROFILE_START("Eigensolver_elpa|solve_std|setup");

    int error;
    elpa_t handle;

    handle = elpa_allocate(&error);
    if (error != ELPA_OK) {
        return 1;
    }
    setup_handler(handle, stage_, A__, matrix_size__, nev__);

    PROFILE_STOP("Eigensolver_elpa|solve_std|setup");

    auto& mph = get_memory_pool(memory_t::host);
    auto w    = mph.get_unique_ptr<double>(matrix_size__);

    auto A_ptr = A__.size_local() ? A__.at(memory_t::host) : nullptr;
    auto Z_ptr = Z__.size_local() ? Z__.at(memory_t::host) : nullptr;
    elpa_eigenvectors_a_h_a_dc(handle, A_ptr, w.get(), Z_ptr, &error);

    elpa_deallocate(handle, &error);

    std::copy(w.get(), w.get() + nev__, eval__);

    if (nt != omp_get_max_threads()) {
        std::stringstream s;
        s << "number of OMP threads was changed by elpa" << std::endl
          << "  initial number of threads : " << nt << std::endl
          << "  new number of threads : " << omp_get_max_threads();
        RTE_THROW(s);
    }

    return 0;
}

/// Solve a standard eigen-value problem for all eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<double>& A__, double* eval__, la::dmatrix<double>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
}

/// Solve a standard eigen-value problem for all eigen-pairs.
int
Eigensolver_elpa::solve(ftn_int matrix_size__, la::dmatrix<std::complex<double>>& A__, double* eval__,
                        la::dmatrix<std::complex<double>>& Z__)
{
    return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
}

#endif

#if defined(SIRIUS_DLAF)

void
Eigensolver_dlaf::finalize()
{
    dlaf::finalize();
}

#endif

#if defined(SIRIUS_CUDA)

namespace {

template <typename T>
int
cusolver_solve_std(ftn_int matrix_size__, int nev__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
{
    cusolverEigMode_t jobz   = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo    = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

    auto& mpd = get_memory_pool(memory_t::device);
    auto w    = mpd.get_unique_ptr<real_type<T>>(matrix_size__);
    acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);

    int lwork;
    int h_meig;
    auto vl = -std::numeric_limits<real_type<T>>::infinity();
    auto vu = std::numeric_limits<real_type<T>>::infinity();

    if (std::is_same<T, double>::value) {
        CALL_CUSOLVER(cusolverDnDsyevdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                       reinterpret_cast<double*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<double*>(w.get()), &lwork));
    } else if (std::is_same<T, float>::value) {
        CALL_CUSOLVER(cusolverDnSsyevdx_bufferSize, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                                     reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(), vl,
                                                     vu, 1, nev__, &h_meig, reinterpret_cast<float*>(w.get()), &lwork));
    } else if (std::is_same<T, std::complex<double>>::value) {
        CALL_CUSOLVER(cusolverDnZheevdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                       reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__,
                       &h_meig, reinterpret_cast<double*>(w.get()), &lwork));
    } else if (std::is_same<T, std::complex<float>>::value) {
        CALL_CUSOLVER(cusolverDnCheevdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                       reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<float*>(w.get()), &lwork));
    }

    auto work = mpd.get_unique_ptr<T>(lwork);

    int info;
    auto dinfo = mpd.get_unique_ptr<int>(1);
    if (std::is_same<T, double>::value) {
        CALL_CUSOLVER(cusolverDnDsyevdx,
                      (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                       reinterpret_cast<double*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<double*>(w.get()), reinterpret_cast<double*>(work.get()), lwork, dinfo.get()));
    } else if (std::is_same<T, float>::value) {
        CALL_CUSOLVER(cusolverDnSsyevdx,
                      (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                       reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<float*>(w.get()), reinterpret_cast<float*>(work.get()), lwork, dinfo.get()));
    } else if (std::is_same<T, std::complex<double>>::value) {
        CALL_CUSOLVER(cusolverDnZheevdx, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                          reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(), vl,
                                          vu, 1, nev__, &h_meig, reinterpret_cast<double*>(w.get()),
                                          reinterpret_cast<cuDoubleComplex*>(work.get()), lwork, dinfo.get()));
    } else if (std::is_same<T, std::complex<float>>::value) {
        CALL_CUSOLVER(cusolverDnCheevdx, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                          reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(), vl, vu,
                                          1, nev__, &h_meig, reinterpret_cast<float*>(w.get()),
                                          reinterpret_cast<cuFloatComplex*>(work.get()), lwork, dinfo.get()));
    }

    acc::copyout(&info, dinfo.get(), 1);
    if (!info) {
        acc::copyout(eval__, w.get(), nev__);
        acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
    }
    return info;
}

template <typename T>
int
cusolver_solve_gen(ftn_int matrix_size__, int nev__, dmatrix<T>& A__, dmatrix<T>& B__, real_type<T>* eval__,
                   dmatrix<T>& Z__)
{
    cusolverEigType_t itype  = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    cusolverEigMode_t jobz   = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo    = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;

    auto& mpd = get_memory_pool(memory_t::device);
    auto w    = mpd.get_unique_ptr<real_type<T>>(matrix_size__);
    acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);
    acc::copyin(B__.at(memory_t::device), B__.ld(), B__.at(memory_t::host), B__.ld(), matrix_size__, matrix_size__);

    int lwork;
    int h_meig;
    auto vl = -std::numeric_limits<real_type<T>>::infinity();
    auto vu = std::numeric_limits<real_type<T>>::infinity();

    if (std::is_same<T, double>::value) {
        CALL_CUSOLVER(cusolverDnDsygvdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                       reinterpret_cast<double*>(A__.at(memory_t::device)), A__.ld(),
                       reinterpret_cast<double*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<double*>(w.get()), &lwork));
    } else if (std::is_same<T, float>::value) {
        CALL_CUSOLVER(cusolverDnSsygvdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                       reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(),
                       reinterpret_cast<float*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<float*>(w.get()), &lwork));
    } else if (std::is_same<T, std::complex<double>>::value) {
        CALL_CUSOLVER(cusolverDnZhegvdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                       reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                       reinterpret_cast<cuDoubleComplex*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__,
                       &h_meig, reinterpret_cast<double*>(w.get()), &lwork));
    } else if (std::is_same<T, std::complex<float>>::value) {
        CALL_CUSOLVER(cusolverDnChegvdx_bufferSize,
                      (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                       reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(),
                       reinterpret_cast<cuFloatComplex*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<float*>(w.get()), &lwork));
    }

    auto work = mpd.get_unique_ptr<T>(lwork);

    int info;
    auto dinfo = mpd.get_unique_ptr<int>(1);
    if (std::is_same<T, double>::value) {
        CALL_CUSOLVER(cusolverDnDsygvdx,
                      (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                       reinterpret_cast<double*>(A__.at(memory_t::device)), A__.ld(),
                       reinterpret_cast<double*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<double*>(w.get()), reinterpret_cast<double*>(work.get()), lwork, dinfo.get()));
    } else if (std::is_same<T, float>::value) {
        CALL_CUSOLVER(cusolverDnSsygvdx,
                      (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                       reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(),
                       reinterpret_cast<float*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__, &h_meig,
                       reinterpret_cast<float*>(w.get()), reinterpret_cast<float*>(work.get()), lwork, dinfo.get()));
    } else if (std::is_same<T, std::complex<double>>::value) {
        CALL_CUSOLVER(cusolverDnZhegvdx, (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                                          reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                          reinterpret_cast<cuDoubleComplex*>(B__.at(memory_t::device)), B__.ld(), vl,
                                          vu, 1, nev__, &h_meig, reinterpret_cast<double*>(w.get()),
                                          reinterpret_cast<cuDoubleComplex*>(work.get()), lwork, dinfo.get()));
    } else if (std::is_same<T, std::complex<float>>::value) {
        CALL_CUSOLVER(cusolverDnChegvdx, (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                                          reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(),
                                          reinterpret_cast<cuFloatComplex*>(B__.at(memory_t::device)), B__.ld(), vl, vu,
                                          1, nev__, &h_meig, reinterpret_cast<float*>(w.get()),
                                          reinterpret_cast<cuFloatComplex*>(work.get()), lwork, dinfo.get()));
    }

    acc::copyout(&info, dinfo.get(), 1);
    if (!info) {
        acc::copyout(eval__, w.get(), nev__);
        acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
    }
    return info;
}

} // namespace

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__)
{
    PROFILE("Eigensolver_cusolver|dsyevdx");
    return cusolver_solve_std(matrix_size__, nev__, A__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<double>& A__, double* eval__,
                            dmatrix<double>& Z__)
{
    PROFILE("Eigensolver_cusolver|ssyevdx");
    return cusolver_solve_std(matrix_size__, nev__, A__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<float>>& A__, float* eval__,
                            dmatrix<std::complex<float>>& Z__)
{
    PROFILE("Eigensolver_cusolver|cheevdx");
    return cusolver_solve_std(matrix_size__, nev__, A__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
                            dmatrix<std::complex<double>>& Z__)
{
    PROFILE("Eigensolver_cusolver|zheevdx");
    return cusolver_solve_std(matrix_size__, nev__, A__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<double>& A__, dmatrix<double>& B__,
                            double* eval__, dmatrix<double>& Z__)
{
    PROFILE("Eigensolver_cusolver|dsygvdx");
    return cusolver_solve_gen(matrix_size__, nev__, A__, B__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__,
                            dmatrix<float>& Z__)
{
    PROFILE("Eigensolver_cusolver|ssygvdx");
    return cusolver_solve_gen(matrix_size__, nev__, A__, B__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<double>>& A__,
                            dmatrix<std::complex<double>>& B__, double* eval__, dmatrix<std::complex<double>>& Z__)
{
    PROFILE("Eigensolver_cusolver|zhegvdx");
    return cusolver_solve_gen(matrix_size__, nev__, A__, B__, eval__, Z__);
}

int
Eigensolver_cusolver::solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<float>>& A__,
                            dmatrix<std::complex<float>>& B__, float* eval__, dmatrix<std::complex<float>>& Z__)
{
    PROFILE("Eigensolver_cusolver|chegvdx");
    return cusolver_solve_gen(matrix_size__, nev__, A__, B__, eval__, Z__);
}

#endif

} // namespace la

} // namespace sirius

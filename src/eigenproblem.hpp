// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file eigenproblem.hpp
 *
 *  \brief Contains definition and implementaiton of various eigenvalue solver interfaces.
 */

#ifndef __EIGENPROBLEM_HPP__
#define __EIGENPROBLEM_HPP__

#include <omp.h>
#include "utils/profiler.hpp"
#include "linalg.hpp"

#if defined(__ELPA)
#include <elpa_constants.h>
extern "C" {
#include "elpa.h"
}
#endif

#if defined(__GPU) && defined(__MAGMA)
#include "GPU/magma.hpp"
#endif

#if defined(__GPU) && defined(__CUDA)
#include "GPU/cusolver.hpp"
#endif

using namespace sddk;

//TODO use ELPA functions to transform to standard eigen-problem

/// Type of eigen-value solver.
enum class ev_solver_t
{
    /// LAPACK
    lapack,

    /// ScaLAPACK
    scalapack,

    /// ELPA 1-stage solver
    elpa1,

    /// ELPA 2-stage solver
    elpa2,

    /// MAGMA with CPU pointers
    magma,

    /// MAGMA with GPU pointers
    magma_gpu,

    /// PLASMA
    plasma,

    /// CUDA eigen-solver
    cusolver
};

inline ev_solver_t get_ev_solver_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    static const std::map<std::string, ev_solver_t> map_to_type = {
        {"lapack", ev_solver_t::lapack}, {"scalapack", ev_solver_t::scalapack}, {"elpa1", ev_solver_t::elpa1},
        {"elpa2", ev_solver_t::elpa2},   {"magma", ev_solver_t::magma},         {"magma_gpu", ev_solver_t::magma_gpu},
        {"plasma", ev_solver_t::plasma}, {"cusolver", ev_solver_t::cusolver}};

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of eigen-solver : " << name__;
        TERMINATE(s);
    }

    return map_to_type.at(name__);
}

const std::string error_msg_not_implemented = "solver is not implemented";

class Eigensolver
{
  protected:
    /// Memory pool for CPU work buffers.
    memory_pool mp_h_;
    /// Memory pool for CPU work buffers using pinned memory.
    memory_pool mp_hp_;
    /// Memory pool for GPU work buffers.
    memory_pool mp_d_;

  public:
    Eigensolver()
        : mp_h_(memory_pool(memory_t::host))
        , mp_hp_(memory_pool(memory_t::host_pinned))
        , mp_d_(memory_pool(memory_t::device))
    {
    }

    virtual ~Eigensolver()
    {
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }
    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__, dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
                      dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
                      dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
                      dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
                      dmatrix<double>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
                      double* eval__, dmatrix<double_complex>& Z__)
    {
        TERMINATE(error_msg_not_implemented);
        return -1;
    }

    virtual bool is_parallel() = 0;
};

class Eigensolver_lapack : public Eigensolver
{
  public:
    inline bool is_parallel()
    {
        return false;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_lapack|dsyevd");

        ftn_int info;
        ftn_int lda = A__.ld();

        ftn_int lwork  = 1 + 6 * matrix_size__ + 2 * matrix_size__ * matrix_size__;
        ftn_int liwork = 3 + 5 * matrix_size__;

        auto work  = mp_h_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(dsyevd)("V", "U", &matrix_size__, A__.at(memory_t::host), &lda, eval__, work.get(), &lwork,
                        iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        if (!info) {
            for (int i = 0; i < matrix_size__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }
        return info;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__, dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_lapack|zheevd");

        ftn_int info;
        ftn_int lda = A__.ld();

        ftn_int lwork  = 2 * matrix_size__ + matrix_size__ * matrix_size__;
        ftn_int lrwork = 1 + 5 * matrix_size__ + 2 * matrix_size__ * matrix_size__;
        ftn_int liwork = 3 + 5 * matrix_size__;

        auto work  = mp_h_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_h_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(zheevd)("V", "U", &matrix_size__, A__.at(memory_t::host), &lda, eval__, work.get(),
                        &lwork, rwork.get(), &lrwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        if (!info) {
            for (int i = 0; i < matrix_size__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }
        return info;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_lapack|dsyevr");

        double vl, vu;
        ftn_int il{1};
        ftn_int m{-1};
        ftn_int info;

        auto w      = mp_h_.get_unique_ptr<double>(matrix_size__);
        auto isuppz = mp_h_.get_unique_ptr<ftn_int>(2 * matrix_size__);

        ftn_int lda = A__.ld();
        ftn_int ldz = Z__.ld();

        double abs_tol = 2 * linalg_base::dlamch('S');

        ftn_int liwork = 10 * matrix_size__;
        auto iwork     = mp_h_.get_unique_ptr<ftn_int>(liwork);

        int nb        = linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size__, -1, -1, -1);
        ftn_int lwork = (nb + 6) * matrix_size__;
        auto work     = mp_h_.get_unique_ptr<double>(lwork);

        FORTRAN(dsyevr)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &lda, &vl, &vu, &il, &nev__, &abs_tol, &m, w.get(),
         Z__.at(memory_t::host), &ldz, isuppz.get(), work.get(), &lwork, iwork.get(), &liwork, &info, (ftn_int)1,
         (ftn_int)1, (ftn_int)1);

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eign-values: " << nev__ << std::endl
              << "number of eign-values found: " << m;
            WARNING(s);
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_lapack|zheevx");

        double vl, vu;
        ftn_int il{1};
        ftn_int m{-1};
        ftn_int info;

        auto w     = mp_h_.get_unique_ptr<double>(matrix_size__);
        auto ifail = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);

        ftn_int lda = A__.ld();
        ftn_int ldz = Z__.ld();

        double abs_tol = 2 * linalg_base::dlamch('S');

        ftn_int liwork = 5 * matrix_size__;
        auto iwork     = mp_h_.get_unique_ptr<ftn_int>(liwork);

        int nb        = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size__, -1, -1, -1);
        ftn_int lwork = (nb + 1) * matrix_size__;
        auto work     = mp_h_.get_unique_ptr<double_complex>(lwork);

        ftn_int lrwork = 7 * matrix_size__;
        auto rwork     = mp_h_.get_unique_ptr<double>(lrwork);

        //FORTRAN(zheevr)
        //("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &lda, &vl, &vu, &il, &nev__, &abs_tol, &m, w.get(),
        // Z__.at(memory_t::host), &ldz, isuppz.get(), work.get(), &lwork, rwork.get(), &lrwork, iwork.get(), &liwork,
        // &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        FORTRAN(zheevx)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &lda, &vl, &vu, &il, &nev__, &abs_tol, &m, w.get(),
         Z__.at(memory_t::host), &ldz, work.get(), &lwork, rwork.get(), iwork.get(), ifail.get(),
         &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eign-values: " << nev__ << std::endl
              << "number of eign-values found: " << m;
            WARNING(s);
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
              dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_lapack|dsygvx");

        ftn_int info;

        ftn_int lda = A__.ld();
        ftn_int ldb = B__.ld();
        ftn_int ldz = Z__.ld();

        double abs_tol = 2 * linalg_base::dlamch('S');
        double vl{0};
        double vu{0};
        ftn_int ione{1};
        ftn_int m{0};

        auto w     = mp_h_.get_unique_ptr<double>(matrix_size__);
        auto ifail = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);

        int nb     = linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size__, 0, 0, 0);
        int lwork  = (nb + 3) * matrix_size__ + 1024;
        int liwork = 5 * matrix_size__;

        auto work  = mp_h_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(dsygvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &lda, B__.at(memory_t::host), &ldb, &vl, &vu,
         &ione, &nev__, &abs_tol, &m, w.get(), Z__.at(memory_t::host), &ldz, work.get(), &lwork, iwork.get(),
         ifail.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eign-values: " << nev__ << std::endl
              << "number of eign-values found: " << m;
            WARNING(s);
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
              double* eval__, dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_lapack|zhegvx");

        ftn_int info;

        ftn_int lda = A__.ld();
        ftn_int ldb = B__.ld();
        ftn_int ldz = Z__.ld();

        double abs_tol = 2 * linalg_base::dlamch('S');
        double vl{0};
        double vu{0};
        ftn_int ione{1};
        ftn_int m{0};

        auto w     = mp_h_.get_unique_ptr<double>(matrix_size__);
        auto ifail = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);

        int nb     = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size__, 0, 0, 0);
        int lwork  = (nb + 1) * matrix_size__;
        int lrwork = 7 * matrix_size__;
        int liwork = 5 * matrix_size__;

        auto work  = mp_h_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_h_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(zhegvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &lda, B__.at(memory_t::host), &ldb, &vl, &vu,
         &ione, &nev__, &abs_tol, &m, w.get(), Z__.at(memory_t::host), &ldz, work.get(), &lwork, rwork.get(),
         iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eign-values: " << nev__ << std::endl
              << "number of eign-values found: " << m;
            WARNING(s);
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }
};

#ifdef __ELPA
class Eigensolver_elpa : public Eigensolver
{
  private:
    int stage_;

    template <typename T>
    void to_std(ftn_int matrix_size__, dmatrix<T>& A__, dmatrix<T>& B__, dmatrix<T>& Z__) const
    {
        PROFILE("Eigensolver_elpa|to_std");

        if (A__.num_cols_local() != Z__.num_cols_local()) {
            std::stringstream s;
            s << "number of columns in A and Z doesn't match" << std::endl
              << "  number of cols in A (local and global): " << A__.num_cols_local() << " " << A__.num_cols()
              << std::endl
              << "  number of cols in B (local and global): " << B__.num_cols_local() << " " << B__.num_cols()
              << std::endl
              << "  number of cols in Z (local and global): " << Z__.num_cols_local() << " " << Z__.num_cols()
              << std::endl
              << "  number of rows in A (local and global): " << A__.num_rows_local() << " " << A__.num_rows()
              << std::endl
              << "  number of rows in B (local and global): " << B__.num_rows_local() << " " << B__.num_rows()
              << std::endl
              << "  number of rows in Z (local and global): " << Z__.num_rows_local() << " " << Z__.num_rows()
              << std::endl;
            TERMINATE(s);
        }
        if (A__.bs_row() != A__.bs_col()) {
            TERMINATE("wrong block size");
        }

        /* Cholesky factorization B = U^{H}*U */
        linalg(linalg_t::scalapack).potrf(matrix_size__, B__.at(memory_t::host), B__.ld(), B__.descriptor());
        /* inversion of the triangular matrix */
        linalg(linalg_t::scalapack).trtri(matrix_size__, B__.at(memory_t::host), B__.ld(), B__.descriptor());
        /* U^{-1} is upper triangular matrix */
        for (int i = 0; i < matrix_size__; i++) {
            for (int j = i + 1; j < matrix_size__; j++) {
                B__.set(j, i, 0);
            }
        }
        /* transform to standard eigen-problem */
        /* A * U{-1} -> Z */
        linalg(linalg_t::scalapack).gemm('N', 'N', matrix_size__, matrix_size__, matrix_size__,
            &linalg_const<T>::one(), A__, 0, 0, B__, 0, 0, &linalg_const<T>::zero(), Z__, 0, 0);
        /* U^{-H} * Z = U{-H} * A * U^{-1} -> A */
        linalg(linalg_t::scalapack).gemm('C', 'N', matrix_size__, matrix_size__, matrix_size__,
            &linalg_const<T>::one(), B__, 0, 0, Z__, 0, 0,  &linalg_const<T>::zero(), A__, 0, 0);
    }

    template <typename T>
    void bt(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, dmatrix<T>& Z__) const
    {
        PROFILE("Eigensolver_elpa|bt");
        /* back-transform of eigen-vectors */
        linalg(linalg_t::scalapack).gemm('N', 'N', matrix_size__, nev__, matrix_size__, &linalg_const<T>::one(),
                  B__, 0, 0, Z__, 0, 0, &linalg_const<T>::zero(), A__, 0, 0);
        A__ >> Z__;

    }
  public:
    Eigensolver_elpa(int stage__)
        : stage_(stage__)
    {
        if (!(stage_ == 1 || stage_ == 2)) {
            TERMINATE("wrong type of ELPA solver");
        }
    }

    inline bool is_parallel()
    {
        return true;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
              dmatrix<double>& Z__)
    {
        to_std(matrix_size__, A__, B__, Z__);

        /* solve a standard problem */
        int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
        if (result) {
            return result;
        }

        bt(matrix_size__, nev__, A__, B__, Z__);
        return 0;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
              double* eval__, dmatrix<double_complex>& Z__)
    {
        to_std(matrix_size__, A__, B__, Z__);

        /* solve a standard problem */
        int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
        if (result) {
            return result;
        }

        bt(matrix_size__, nev__, A__, B__, Z__);
        return 0;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__, dmatrix<double>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_elpa|solve_std");

        if (A__.num_cols_local() != Z__.num_cols_local()) {
            TERMINATE("number of columns in A and Z don't match");
        }

        PROFILE_START("Eigensolver_elpa|solve_std|setup");

        int bs = A__.bs_row();

        int error;
        elpa_t handle;

        handle = elpa_allocate(&error);
        elpa_set_integer(handle, "na", matrix_size__, &error);
        elpa_set_integer(handle, "nev", nev__, &error);
        elpa_set_integer(handle, "local_nrows", A__.num_rows_local(), &error);
        elpa_set_integer(handle, "local_ncols", A__.num_cols_local(), &error);
        elpa_set_integer(handle, "nblk", bs, &error);
        elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(A__.blacs_grid().comm().mpi_comm()), &error);
        elpa_set_integer(handle, "process_row", A__.blacs_grid().comm_row().rank(), &error);
        elpa_set_integer(handle, "process_col", A__.blacs_grid().comm_col().rank(), &error);
        elpa_setup(handle);
        if (stage_ == 1) {
            elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
        } else {
            elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
        }
        PROFILE_STOP("Eigensolver_elpa|solve_std|setup");

        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        elpa_eigenvectors_d(handle, A__.at(memory_t::host), w.get(), Z__.at(memory_t::host), &error);

        elpa_deallocate(handle, &error);

        std::copy(w.get(), w.get() + nev__, eval__);

        return 0;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_elpa|solve_std");

        if (A__.num_cols_local() != Z__.num_cols_local()) {
            TERMINATE("number of columns in A and Z don't match");
        }

        int bs = A__.bs_row();

        int error;
        elpa_t handle;

        handle = elpa_allocate(&error);
        elpa_set_integer(handle, "na", matrix_size__, &error);
        elpa_set_integer(handle, "nev", nev__, &error);
        elpa_set_integer(handle, "local_nrows", A__.num_rows_local(), &error);
        elpa_set_integer(handle, "local_ncols", A__.num_cols_local(), &error);
        elpa_set_integer(handle, "nblk", bs, &error);
        elpa_set_integer(handle, "mpi_comm_parent", MPI_Comm_c2f(A__.blacs_grid().comm().mpi_comm()), &error);
        elpa_set_integer(handle, "process_row", A__.blacs_grid().comm_row().rank(), &error);
        elpa_set_integer(handle, "process_col", A__.blacs_grid().comm_col().rank(), &error);
        elpa_setup(handle);
        if (stage_ == 1) {
            elpa_set_integer(handle, "solver", ELPA_SOLVER_1STAGE, &error);
        } else {
            elpa_set_integer(handle, "solver", ELPA_SOLVER_2STAGE, &error);
        }
        PROFILE_STOP("Eigensolver_elpa|solve_std|setup");

        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        elpa_eigenvectors_dc(handle, A__.at(memory_t::host), w.get(), Z__.at(memory_t::host), &error);

        elpa_deallocate(handle, &error);

        std::copy(w.get(), w.get() + nev__, eval__);

        return 0;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__, dmatrix<double_complex>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
    }
};
#else
class Eigensolver_elpa : public Eigensolver
{
  public:
    Eigensolver_elpa(int stage__)
    {
    }

    inline bool is_parallel()
    {
        return true;
    }
};
#endif

#ifdef __SCALAPACK
class Eigensolver_scalapack : public Eigensolver
{
  private:
    double const ortfac_{1e-6};
    double const abstol_{1e-12};

  public:
    inline bool is_parallel()
    {
        return true;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__, dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_scalapack|pzheevd");
        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        ftn_int info;
        ftn_int ione{1};

        ftn_int lwork{-1};
        ftn_int lrwork{-1};
        ftn_int liwork{-1};
        double_complex work1;
        double rwork1;
        ftn_int iwork1;

        /* work size query */
        FORTRAN(pzheevd)
        ("V", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, eval__, Z__.at(memory_t::host), &ione,
         &ione, descz, &work1, &lwork, &rwork1, &lrwork, &iwork1, &liwork, &info, (ftn_int)1, (ftn_int)1);

        lwork  = static_cast<ftn_int>(work1.real()) + 1;
        lrwork = static_cast<ftn_int>(rwork1) + 1;
        liwork = iwork1;

        auto work  = mp_h_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_h_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(pzheevd)
        ("V", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, eval__, Z__.at(memory_t::host), &ione,
         &ione, descz, work.get(), &lwork, rwork.get(), &lrwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        return info;
    }

    int solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_scalapack|pdsyevd");

        ftn_int info;
        ftn_int ione{1};

        ftn_int lwork{-1};
        ftn_int liwork{-1};
        double work1[10];
        ftn_int iwork1[10];

        /* work size query */
        FORTRAN(pdsyevd)
        ("V", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, const_cast<ftn_int*>(A__.descriptor()), eval__,
         Z__.at(memory_t::host), &ione, &ione, const_cast<ftn_int*>(Z__.descriptor()), work1, &lwork, iwork1, &liwork, &info, (ftn_int)1, (ftn_int)1);

        lwork  = static_cast<ftn_int>(work1[0]) + 1;
        liwork = iwork1[0];

        auto work  = mp_h_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(pdsyevd)
        ("V", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, const_cast<ftn_int*>(A__.descriptor()), eval__,
         Z__.at(memory_t::host), &ione, &ione, const_cast<ftn_int*>(Z__.descriptor()), work.get(), &lwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        return info;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_scalapack|pdsyevx");

        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        double d1;
        ftn_int info{-1};

        auto ifail   = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mp_h_.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mp_h_.get_unique_ptr<double>(A__.blacs_grid().comm().size());
        auto w       = mp_h_.get_unique_ptr<double>(matrix_size__);

        /* work size query */
        double work3[3];
        ftn_int iwork1;
        ftn_int lwork{-1};
        ftn_int liwork{-1};

        FORTRAN(pdsyevx)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, &d1, &d1, &ione, &nev__, &abstol_,
         &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host), &ione, &ione, descz, work3, &lwork, &iwork1, &liwork,
         ifail.get(), iclustr.get(), gap.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        lwork  = static_cast<ftn_int>(work3[0]) + 4 * (1 << 20);
        liwork = iwork1;

        auto work  = mp_h_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(pdsyevx)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, &d1, &d1, &ione, &nev__, &abstol_,
         &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host), &ione, &ione, descz, work.get(), &lwork, iwork.get(),
         &liwork, ifail.get(), iclustr.get(), gap.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        if ((m != nev__) || (nz != nev__)) {
            WARNING("Not all eigen-vectors or eigen-values are found.");
            return 1;
        }

        if (info) {
            if ((info / 2) % 2) {
                std::stringstream s;
                s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
                  << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                int k = A__.blacs_grid().comm().size();
                for (int i = 0; i < A__.blacs_grid().comm().size() - 1; i++) {
                    if ((iclustr.get()[2 * i + 1] != 0) && (iclustr.get()[2 * (i + 1)] == 0)) {
                        k = i + 1;
                        break;
                    }
                }

                s << "number of eigenvalue clusters : " << k << std::endl;
                for (int i = 0; i < k; i++) {
                    s << iclustr.get()[2 * i] << " : " << iclustr.get()[2 * i + 1] << std::endl;
                }
                WARNING(s);
            }

            std::stringstream s;
            s << "pdsyevx returned " << info;
            WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_scalapack|pzheevx");

        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        double d1;
        ftn_int info{-1};

        auto ifail   = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mp_h_.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mp_h_.get_unique_ptr<double>(A__.blacs_grid().comm().size());
        auto w       = mp_h_.get_unique_ptr<double>(matrix_size__);

        /* work size query */
        double_complex work3[3];
        double rwork3[3];
        ftn_int iwork1;
        ftn_int lwork  = -1;
        ftn_int lrwork = -1;
        ftn_int liwork = -1;
        FORTRAN(pzheevx)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, &d1, &d1, &ione, &nev__, &abstol_,
         &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host), &ione, &ione, descz, work3, &lwork, rwork3, &lrwork,
         &iwork1, &liwork, ifail.get(), iclustr.get(), gap.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        lwork  = static_cast<int32_t>(work3[0].real()) + (1 << 16);
        lrwork = static_cast<int32_t>(rwork3[0]) + (1 << 16);
        liwork = iwork1;

        auto work  = mp_h_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_h_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(pzheevx)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, &d1, &d1, &ione, &nev__, &abstol_,
         &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host), &ione, &ione, descz, work.get(), &lwork, rwork.get(),
         &lrwork, iwork.get(), &liwork, ifail.get(), iclustr.get(), gap.get(), &info, (ftn_int)1, (ftn_int)1,
         (ftn_int)1);

        if ((m != nev__) || (nz != nev__)) {
            WARNING("Not all eigen-vectors or eigen-values are found.");
            return 1;
        }

        if (info) {
            if ((info / 2) % 2) {
                std::stringstream s;
                s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
                  << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                int k = A__.blacs_grid().comm().size();
                for (int i = 0; i < A__.blacs_grid().comm().size() - 1; i++) {
                    if ((iclustr.get()[2 * i + 1] != 0) && (iclustr.get()[2 * (i + 1)] == 0)) {
                        k = i + 1;
                        break;
                    }
                }

                s << "number of eigenvalue clusters : " << k << std::endl;
                for (int i = 0; i < k; i++) {
                    s << iclustr.get()[2 * i] << " : " << iclustr.get()[2 * i + 1] << std::endl;
                }
                WARNING(s);
            }

            std::stringstream s;
            s << "pzheevx returned " << info;
            WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
              dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_scalapack|pdsygvx");

        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descb[9];
        linalg_base::descinit(descb, matrix_size__, matrix_size__, B__.bs_row(), B__.bs_col(), 0, 0,
                              B__.blacs_grid().context(), B__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        auto ifail   = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mp_h_.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mp_h_.get_unique_ptr<double>(A__.blacs_grid().comm().size());
        auto w       = mp_h_.get_unique_ptr<double>(matrix_size__);

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        double d1;
        ftn_int info{-1};

        double work1[3];
        ftn_int lwork  = -1;
        ftn_int liwork = -1;
        /* work size query */
        FORTRAN(pdsygvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, B__.at(memory_t::host),
         &ione, &ione, descb, &d1, &d1, &ione, &nev__, &abstol_, &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host),
         &ione, &ione, descz, work1, &lwork, &liwork, &lwork, ifail.get(), iclustr.get(), gap.get(), &info, (ftn_int)1,
         (ftn_int)1, (ftn_int)1);

        lwork = static_cast<int32_t>(work1[0]) + 4 * (1 << 20);

        auto work  = mp_h_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(pdsygvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, B__.at(memory_t::host),
         &ione, &ione, descb, &d1, &d1, &ione, &nev__, &abstol_, &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host),
         &ione, &ione, descz, work.get(), &lwork, iwork.get(), &liwork, ifail.get(), iclustr.get(), gap.get(), &info,
         (ftn_int)1, (ftn_int)1, (ftn_int)1);

        if ((m != nev__) || (nz != nev__)) {
            WARNING("Not all eigen-vectors or eigen-values are found.");
            return 1;
        }

        if (info) {
            if ((info / 2) % 2) {
                std::stringstream s;
                s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
                  << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                int k = A__.blacs_grid().comm().size();
                for (int i = 0; i < A__.blacs_grid().comm().size() - 1; i++) {
                    if ((iclustr.get()[2 * i + 1] != 0) && (iclustr.get()[2 * (i + 1)] == 0)) {
                        k = i + 1;
                        break;
                    }
                }

                s << "number of eigenvalue clusters : " << k << std::endl;
                for (int i = 0; i < k; i++) {
                    s << iclustr.get()[2 * i] << " : " << iclustr.get()[2 * i + 1] << std::endl;
                }
                WARNING(s);
            }

            std::stringstream s;
            s << "pdsygvx returned " << info;
            WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
              double* eval__, dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_scalapack|pzhegvx");

        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descb[9];
        linalg_base::descinit(descb, matrix_size__, matrix_size__, B__.bs_row(), B__.bs_col(), 0, 0,
                              B__.blacs_grid().context(), B__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        auto ifail   = mp_h_.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mp_h_.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mp_h_.get_unique_ptr<double>(A__.blacs_grid().comm().size());
        auto w       = mp_h_.get_unique_ptr<double>(matrix_size__);

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        double d1;
        ftn_int info{-1};

        ftn_int lwork  = -1;
        ftn_int lrwork = -1;
        ftn_int liwork = -1;

        double_complex work1;
        double rwork3[3];
        ftn_int iwork1;

        /* work size query */
        FORTRAN(pzhegvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, B__.at(memory_t::host),
         &ione, &ione, descb, &d1, &d1, &ione, &nev__, &abstol_, &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host),
         &ione, &ione, descz, &work1, &lwork, rwork3, &lrwork, &iwork1, &liwork, ifail.get(), iclustr.get(), gap.get(),
         &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        lwork  = static_cast<int32_t>(work1.real()) + 4096;
        lrwork = static_cast<int32_t>(rwork3[0]) + 4096;
        liwork = iwork1 + 4096;

        auto work  = mp_h_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_h_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<ftn_int>(liwork);

        FORTRAN(pzhegvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, B__.at(memory_t::host),
         &ione, &ione, descb, &d1, &d1, &ione, &nev__, &abstol_, &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host),
         &ione, &ione, descz, work.get(), &lwork, rwork.get(), &lrwork, iwork.get(), &liwork, ifail.get(),
         iclustr.get(), gap.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

        if ((m != nev__) || (nz != nev__)) {
            WARNING("Not all eigen-vectors or eigen-values are found.");
            return 1;
        }

        if (info) {
            if ((info / 2) % 2) {
                std::stringstream s;
                s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
                  << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                int k = A__.blacs_grid().comm().size();
                for (int i = 0; i < A__.blacs_grid().comm().size() - 1; i++) {
                    if ((iclustr.get()[2 * i + 1] != 0) && (iclustr.get()[2 * (i + 1)] == 0)) {
                        k = i + 1;
                        break;
                    }
                }

                s << "number of eigenvalue clusters : " << k << std::endl;
                for (int i = 0; i < k; i++) {
                    s << iclustr.get()[2 * i] << " : " << iclustr.get()[2 * i + 1] << std::endl;
                }
                WARNING(s);
            }

            std::stringstream s;
            s << "pzhegvx returned " << info;
            WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }
};
#else
class Eigensolver_scalapack : public Eigensolver
{
  public:
    inline bool is_parallel()
    {
        return true;
    }
};
#endif

#ifdef __MAGMA
class Eigensolver_magma: public Eigensolver
{
  public:

    inline bool is_parallel()
    {
        return false;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
              dmatrix<double>& Z__)
    {
        int nt = omp_get_max_threads();
        int lda = A__.ld();
        int ldb = B__.ld();

        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        int m;
        int info;

        int lwork;
        int liwork;
        magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

        auto h_work = mp_hp_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

        magma_dsygvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size__, A__.at(memory_t::host), lda,
                             B__.at(memory_t::host), ldb, 0.0, 0.0, 1, nev__, &m, w.get(), h_work.get(), lwork,
                             iwork.get(), liwork, &info);


        if (nt != omp_get_max_threads()) {
            TERMINATE("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nev__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }

        return info;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
              double* eval__, dmatrix<double_complex>& Z__)
    {
        int nt = omp_get_max_threads();
        int lda = A__.ld();
        int ldb = B__.ld();

        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        int m;
        int info;

        int lwork;
        int lrwork;
        int liwork;
        magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

        auto h_work = mp_hp_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_hp_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

        magma_zhegvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
                             reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::host)), lda,
                             reinterpret_cast<magmaDoubleComplex*>(B__.at(memory_t::host)), ldb, 0.0, 0.0,
                             1, nev__, &m, w.get(), reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork,
                             rwork.get(), lrwork, iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            TERMINATE("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nev__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }

        return info;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_magma|dsygvdx");

        int nt = omp_get_max_threads();
        int lda = A__.ld();
        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        int lwork;
        int liwork;
        magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

        auto h_work = mp_hp_.get_unique_ptr<double>(lwork);
        auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

        int info;
        int m;

        magma_dsyevdx(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__, A__.at(memory_t::host), lda, 0.0, 0.0, 1,
                      nev__, &m, w.get(), h_work.get(), lwork, iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            TERMINATE("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nev__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }

        return info;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_magma|zheevdx");

        int nt = omp_get_max_threads();
        int lda = A__.ld();
        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        int info, m;

        int lwork;
        int lrwork;
        int liwork;
        magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

        auto h_work = mp_hp_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_hp_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

        magma_zheevdx_2stage(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
                      reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::host)), lda, 0.0, 0.0, 1,
                      nev__, &m, w.get(), reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork, rwork.get(),
                      lrwork, iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            TERMINATE("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nev__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }

        return info;
    }
};

class Eigensolver_magma_gpu: public Eigensolver
{
  public:

    inline bool is_parallel()
    {
        return false;
    }

    ///// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    //int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
    //          dmatrix<double>& Z__)
    //{
    //    int nt = omp_get_max_threads();
    //    int result{-1};
    //    int lda = A__.ld();
    //    int ldb = B__.ld();

    //    auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

    //    int m;
    //    int info;

    //    int lwork;
    //    int liwork;
    //    magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

    //    auto h_work = mp_hp_.get_unique_ptr<double>(lwork);
    //    auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

    //    magma_dsygvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size__, A__.at(memory_t::host), lda,
    //                         B__.at(memory_t::host), ldb, 0.0, 0.0, 1, nev__, &m, w.get(), h_work.get(), lwork,
    //                         iwork.get(), liwork, &info);


    //    if (nt != omp_get_max_threads()) {
    //        TERMINATE("magma has changed the number of threads");
    //    }

    //    if (m < nev__) {
    //        return 1;
    //    }

    //    if (!info) {
    //        std::copy(w.get(), w.get() + nev__, eval__);
    //        #pragma omp parallel for schedule(static)
    //        for (int i = 0; i < nev__; i++) {
    //            std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
    //                      Z__.at(memory_t::host, 0, i));
    //        }
    //    }

    //    return info;
    //}

    ///// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    //int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
    //          double* eval__, dmatrix<double_complex>& Z__)
    //{
    //    int nt = omp_get_max_threads();
    //    int result{-1};
    //    int lda = A__.ld();
    //    int ldb = B__.ld();

    //    auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

    //    int m;
    //    int info;

    //    int lwork;
    //    int lrwork;
    //    int liwork;
    //    magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

    //    auto h_work = mp_hp_.get_unique_ptr<double_complex>(lwork);
    //    auto rwork = mp_hp_.get_unique_ptr<double>(lrwork);
    //    auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

    //    magma_zhegvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
    //                         reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::host)), lda,
    //                         reinterpret_cast<magmaDoubleComplex*>(B__.at(memory_t::host)), ldb, 0.0, 0.0,
    //                         1, nev__, &m, w.get(), reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork,
    //                         rwork.get(), lrwork, iwork.get(), liwork, &info);

    //    if (nt != omp_get_max_threads()) {
    //        TERMINATE("magma has changed the number of threads");
    //    }

    //    if (m < nev__) {
    //        return 1;
    //    }

    //    if (!info) {
    //        std::copy(w.get(), w.get() + nev__, eval__);
    //        #pragma omp parallel for schedule(static)
    //        for (int i = 0; i < nev__; i++) {
    //            std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
    //                      Z__.at(memory_t::host, 0, i));
    //        }
    //    }

    //    return info;
    //}

    ///// Solve a standard eigen-value problem for N lowest eigen-pairs.
    //int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    //{
    //    PROFILE("Eigensolver_magma|dsygvdx");

    //    int nt = omp_get_max_threads();
    //    int lda = A__.ld();
    //    auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

    //    int lwork;
    //    int liwork;
    //    magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

    //    auto h_work = mp_hp_.get_unique_ptr<double>(lwork);
    //    auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

    //    int info;
    //    int m;

    //    magma_dsyevdx(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__, A__.at(memory_t::host), lda, 0.0, 0.0, 1,
    //                  nev__, &m, w.get(), h_work.get(), lwork, iwork.get(), liwork, &info);
    //
    //    if (nt != omp_get_max_threads()) {
    //        TERMINATE("magma has changed the number of threads");
    //    }

    //    if (m < nev__) {
    //        return 1;
    //    }

    //    if (!info) {
    //        std::copy(w.get(), w.get() + nev__, eval__);
    //        #pragma omp parallel for schedule(static)
    //        for (int i = 0; i < nev__; i++) {
    //            std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
    //                      Z__.at(memory_t::host, 0, i));
    //        }
    //    }

    //    return info;
    //}

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_magma_gpu|zheevdx");

        int nt = omp_get_max_threads();
        int lda = A__.ld();
        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        int info, m;

        int lwork;
        int lrwork;
        int liwork;
        magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

        int llda = matrix_size__ + 32;
        auto z_work = mp_hp_.get_unique_ptr<double_complex>(llda * matrix_size__);

        auto h_work = mp_hp_.get_unique_ptr<double_complex>(lwork);
        auto rwork = mp_hp_.get_unique_ptr<double>(lrwork);
        auto iwork = mp_h_.get_unique_ptr<magma_int_t>(liwork);

        magma_zheevdx_gpu(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
                      reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::device)), lda, 0.0, 0.0, 1,
                      nev__, &m, w.get(),
                      reinterpret_cast<magmaDoubleComplex*>(z_work.get()), llda,
                      reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork,
                      rwork.get(), lrwork, iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            TERMINATE("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            //#pragma omp parallel for schedule(static)
            //for (int i = 0; i < nev__; i++) {
            //    std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
            //              Z__.at(memory_t::host, 0, i));
            //}
            acc::copyout(Z__.at(memory_t::host, 0, 0), Z__.ld(), A__.at(memory_t::device, 0, 0), A__.ld(),
                         matrix_size__, nev__);
        }

        return info;
    }
};
#else
class Eigensolver_magma: public Eigensolver
{
  public:
    inline bool is_parallel()
    {
        return false;
    }
};
class Eigensolver_magma_gpu: public Eigensolver
{
  public:
    inline bool is_parallel()
    {
        return false;
    }
};
#endif

#if defined(__CUDA)
class Eigensolver_cuda: public Eigensolver
{
  public:
    inline bool is_parallel()
    {
        return false;
    }

    int solve(ftn_int matrix_size__, int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_cuda|zheevd");

        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        auto w = mp_d_.get_unique_ptr<double>(matrix_size__);
        acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);

        int lwork;
        CALL_CUSOLVER(cusolverDnZheevd_bufferSize, (cusolver::cusolver_handle(), jobz, uplo, matrix_size__,
                                                    reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                                    w.get(), &lwork));

        auto work = mp_d_.get_unique_ptr<double_complex>(lwork);

        int info;
        auto dinfo = mp_d_.get_unique_ptr<int>(1);
        CALL_CUSOLVER(cusolverDnZheevd, (cusolver::cusolver_handle(), jobz, uplo, matrix_size__,
                                         reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                         w.get(), reinterpret_cast<cuDoubleComplex*>(work.get()), lwork, dinfo.get()));
        acc::copyout(&info, dinfo.get(), 1);
        if (!info) {
            acc::copyout(eval__, w.get(), nev__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }
        return info;
    }

    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__, dmatrix<double_complex>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
    }

    int solve(ftn_int matrix_size__, int nev__, dmatrix<double>& A__, double* eval__,
              dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_cuda|dsyevd");

        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        auto w = mp_d_.get_unique_ptr<double>(matrix_size__);
        acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);

        int lwork;
        CALL_CUSOLVER(cusolverDnDsyevd_bufferSize, (cusolver::cusolver_handle(), jobz, uplo, matrix_size__,
                                                    A__.at(memory_t::device), A__.ld(),
                                                    w.get(), &lwork));

        auto work = mp_d_.get_unique_ptr<double>(lwork);

        int info;
        auto dinfo = mp_d_.get_unique_ptr<int>(1);
        CALL_CUSOLVER(cusolverDnDsyevd, (cusolver::cusolver_handle(), jobz, uplo, matrix_size__,
                                         A__.at(memory_t::device), A__.ld(),
                                         w.get(), work.get(), lwork, dinfo.get()));
        acc::copyout(&info, dinfo.get(), 1);
        if (!info) {
            acc::copyout(eval__, w.get(), nev__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }
        return info;
    }

    int solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
    }

    int solve(ftn_int matrix_size__, int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        PROFILE("Eigensolver_cuda|zhegvd");

        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        auto w = mp_d_.get_unique_ptr<double>(matrix_size__);
        acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);
        acc::copyin(B__.at(memory_t::device), B__.ld(), B__.at(memory_t::host), B__.ld(), matrix_size__, matrix_size__);

        int lwork;
        CALL_CUSOLVER(cusolverDnZhegvd_bufferSize, (cusolver::cusolver_handle(), itype, jobz, uplo, matrix_size__,
                                                    reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                                    reinterpret_cast<cuDoubleComplex*>(B__.at(memory_t::device)), B__.ld(),
                                                    w.get(), &lwork));

        auto work = mp_d_.get_unique_ptr<double_complex>(lwork);


        int info;
        auto dinfo = mp_d_.get_unique_ptr<int>(1);
        CALL_CUSOLVER(cusolverDnZhegvd, (cusolver::cusolver_handle(), itype, jobz, uplo, matrix_size__,
                                         reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                         reinterpret_cast<cuDoubleComplex*>(B__.at(memory_t::device)), B__.ld(),
                                         w.get(), reinterpret_cast<cuDoubleComplex*>(work.get()), lwork, dinfo.get()));
        acc::copyout(&info, dinfo.get(), 1);
        if (!info) {
            acc::copyout(eval__, w.get(), nev__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }
        return info;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    int solve(ftn_int matrix_size__, int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
              dmatrix<double>& Z__)
    {
        PROFILE("Eigensolver_cuda|dsygvd");

        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        auto w = mp_d_.get_unique_ptr<double>(matrix_size__);
        acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);
        acc::copyin(B__.at(memory_t::device), B__.ld(), B__.at(memory_t::host), B__.ld(), matrix_size__, matrix_size__);

        int lwork;
        CALL_CUSOLVER(cusolverDnDsygvd_bufferSize, (cusolver::cusolver_handle(), itype, jobz, uplo, matrix_size__,
                                                    A__.at(memory_t::device), A__.ld(),
                                                    B__.at(memory_t::device), B__.ld(),
                                                    w.get(), &lwork));

        auto work = mp_d_.get_unique_ptr<double>(lwork);

        int info;
        auto dinfo = mp_d_.get_unique_ptr<int>(1);
        CALL_CUSOLVER(cusolverDnDsygvd, (cusolver::cusolver_handle(), itype, jobz, uplo, matrix_size__,
                                         A__.at(memory_t::device), A__.ld(),
                                         B__.at(memory_t::device), B__.ld(),
                                         w.get(), work.get(), lwork, dinfo.get()));
        acc::copyout(&info, dinfo.get(), 1);
        if (!info) {
            acc::copyout(eval__, w.get(), nev__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }
        return info;
    }

    int solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__, dmatrix<double>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }
};
#else
class Eigensolver_cuda: public Eigensolver
{
  public:
    inline bool is_parallel()
    {
        return false;
    }
};
#endif

inline std::unique_ptr<Eigensolver> Eigensolver_factory(ev_solver_t ev_solver_type__)
{
    Eigensolver* ptr;
    switch (ev_solver_type__) {
        case ev_solver_t::lapack: {
            ptr = new Eigensolver_lapack();
            break;
        }
        case ev_solver_t::scalapack: {
            ptr = new Eigensolver_scalapack();
            break;
        }
        case ev_solver_t::elpa1: {
            ptr = new Eigensolver_elpa(1);
            break;
        }
        case ev_solver_t::elpa2: {
            ptr = new Eigensolver_elpa(2);
            break;
        }
        case ev_solver_t::magma: {
            ptr = new Eigensolver_magma();
            break;
        }
        case ev_solver_t::magma_gpu: {
            ptr = new Eigensolver_magma_gpu();
            break;
        }
        case ev_solver_t::cusolver: {
            ptr = new Eigensolver_cuda();
            break;
        }
        default: {
            TERMINATE("not implemented");
        }
    }
    return std::unique_ptr<Eigensolver>(ptr);
}

//== #ifdef __PLASMA
//== extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
//==                                       int32_t ldz, double* eval);
//== #endif
//==
//== /// Interface for PLASMA eigen-value solvers.
//== class Eigenproblem_plasma: public Eigenproblem
//== {
//==     public:
//==
//==         Eigenproblem_plasma()
//==         {
//==         }
//==
//==         #ifdef __PLASMA
//==         void solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t
//ldz) const
//==         {
//==             //plasma_set_num_threads(1);
//==             //omp_set_num_threads(1);
//==             //printf("before call to plasma_zheevd_wrapper\n");
//==             plasma_zheevd_wrapper(matrix_size, a, lda, z, lda, eval);
//==             //printf("after call to plasma_zheevd_wrapper\n");
//==             //plasma_set_num_threads(8);
//==             //omp_set_num_threads(8);
//==         }
//==         #endif
//==
//==         bool parallel() const
//==         {
//==             return false;
//==         }
//==
//==         ev_solver_t type() const
//==         {
//==             return ev_plasma;
//==         }
//== };

#endif // __EIGENPROBLEM_HPP__

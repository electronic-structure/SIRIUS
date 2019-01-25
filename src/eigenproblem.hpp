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

#include "utils/timer.hpp"
#include "linalg.hpp"

#ifdef __ELPA
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

    /// MAGMA
    magma,

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
        {"elpa2", ev_solver_t::elpa2},   {"magma", ev_solver_t::magma},         {"plasma", ev_solver_t::plasma},
        {"cusolver", ev_solver_t::cusolver}};

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of eigen-solver : " << name__;
        TERMINATE(s);
    }

    return map_to_type.at(name__);
}

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
        TERMINATE("solver is not implemented");
        return -1;
    }
    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__, dmatrix<double_complex>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
                      dmatrix<double_complex>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
                      dmatrix<double>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
                      dmatrix<double_complex>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
                      dmatrix<double>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
                      double* eval__, dmatrix<double_complex>& Z__)
    {
        TERMINATE("solver is not implemented");
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
        utils::timer t0("Eigensolver_lapack|dsyevd");

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
        utils::timer t0("Eigensolver_lapack|zheevd");

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
        utils::timer t0("Eigensolver_lapack|dsyevr");

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
        utils::timer t0("Eigensolver_lapack|zheevr");

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

        int nb        = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size__, -1, -1, -1);
        ftn_int lwork = (nb + 1) * matrix_size__;
        auto work     = mp_h_.get_unique_ptr<double_complex>(lwork);

        ftn_int lrwork = 24 * matrix_size__;
        auto rwork     = mp_h_.get_unique_ptr<double>(lrwork);

        FORTRAN(zheevr)
        ("V", "I", "U", &matrix_size__, A__.at(memory_t::host), &lda, &vl, &vu, &il, &nev__, &abs_tol, &m, w.get(),
         Z__.at(memory_t::host), &ldz, isuppz.get(), work.get(), &lwork, rwork.get(), &lrwork, iwork.get(), &liwork,
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
        utils::timer t0("Eigensolver_lapack|dsygvx");

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
        utils::timer t0("Eigensolver_lapack|zhegvx");

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
        if (A__.num_cols_local() != Z__.num_cols_local()) {
            std::stringstream s;
            s << "number of columns in A and Z don't match" << std::endl
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

        utils::timer t1("Eigensolver_elpa|to_std");
        /* Cholesky factorization B = U^{H}*U */
        linalg<CPU>::potrf(matrix_size__, B__);
        /* inversion of the triangular matrix */
        linalg<CPU>::trtri(matrix_size__, B__);
        /* U^{-1} is upper triangular matrix */
        for (int i = 0; i < matrix_size__; i++) {
            for (int j = i + 1; j < matrix_size__; j++) {
                B__.set(j, i, 0);
            }
        }
        /* transform to standard eigen-problem */
        /* A * U{-1} -> Z */
        linalg<CPU>::gemm(0, 0, matrix_size__, matrix_size__, matrix_size__, linalg_const<double>::one(), A__, B__,
                          linalg_const<double>::zero(), Z__);
        /* U^{-H} * Z = U{-H} * A * U^{-1} -> A */
        linalg<CPU>::gemm(2, 0, matrix_size__, matrix_size__, matrix_size__, linalg_const<double>::one(), B__, Z__,
                          linalg_const<double>::zero(), A__);
        t1.stop();

        /* solve a standard problem */
        int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
        if (result) {
            return result;
        }

        utils::timer t3("Eigensolver_elpa|bt");
        /* back-transform of eigen-vectors */
        linalg<CPU>::gemm(0, 0, matrix_size__, nev__, matrix_size__, linalg_const<double>::one(), B__, Z__,
                          linalg_const<double>::zero(), A__);
        A__ >> Z__;
        t3.stop();

        return 0;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__,
              double* eval__, dmatrix<double_complex>& Z__)
    {
        if (A__.num_cols_local() != Z__.num_cols_local()) {
            std::stringstream s;
            s << "number of columns in A and Z don't match" << std::endl
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

        utils::timer t1("Eigensolver_elpa|to_std");
        /* Cholesky factorization B = U^{H}*U */
        linalg<CPU>::potrf(matrix_size__, B__);
        /* inversion of the triangular matrix */
        linalg<CPU>::trtri(matrix_size__, B__);
        /* U^{-1} is upper triangular matrix */
        for (int i = 0; i < matrix_size__; i++) {
            for (int j = i + 1; j < matrix_size__; j++) {
                B__.set(j, i, 0);
            }
        }
        /* transform to standard eigen-problem */
        /* A * U{-1} -> Z */
        linalg<CPU>::gemm(0, 0, matrix_size__, matrix_size__, matrix_size__, linalg_const<double_complex>::one(), A__,
                          B__, linalg_const<double_complex>::zero(), Z__);
        /* U^{-H} * Z = U{-H} * A * U^{-1} -> A */
        linalg<CPU>::gemm(2, 0, matrix_size__, matrix_size__, matrix_size__, linalg_const<double_complex>::one(), B__,
                          Z__, linalg_const<double_complex>::zero(), A__);
        t1.stop();

        /* solve a standard problem */
        int result = this->solve(matrix_size__, nev__, A__, eval__, Z__);
        if (result) {
            return result;
        }

        utils::timer t3("Eigensolver_elpa|bt");
        /* back-transform of eigen-vectors */
        linalg<CPU>::gemm(0, 0, matrix_size__, nev__, matrix_size__, linalg_const<double_complex>::one(), B__, Z__,
                          linalg_const<double_complex>::zero(), A__);
        A__ >> Z__;
        t3.stop();

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
        utils::timer t0("Eigensolver_elpa|solve_std");

        if (A__.num_cols_local() != Z__.num_cols_local()) {
            TERMINATE("number of columns in A and Z don't match");
        }

        int num_cols_loc = A__.num_cols_local();
        int bs           = A__.bs_row();
        int lda          = A__.ld();
        int ldz          = Z__.ld();
        int mpi_comm_row = MPI_Comm_c2f(A__.blacs_grid().comm_row().mpi_comm());
        int mpi_comm_col = MPI_Comm_c2f(A__.blacs_grid().comm_col().mpi_comm());
        int mpi_comm_all = MPI_Comm_c2f(A__.blacs_grid().comm().mpi_comm());

        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);

        int success{-1};
        if (stage_ == 1) {
            success = elpa_solve_evp_real_1stage_double_precision(
                matrix_size__, nev__, A__.at(memory_t::host), lda, w.get(), Z__.at(memory_t::host), ldz, bs,
                num_cols_loc, mpi_comm_row, mpi_comm_col, mpi_comm_all, 0);
        } else {
            success = elpa_solve_evp_real_2stage_double_precision(
                matrix_size__, nev__, A__.at(memory_t::host), lda, w.get(), Z__.at(memory_t::host), ldz, bs,
                num_cols_loc, mpi_comm_row, mpi_comm_col, mpi_comm_all, ELPA_2STAGE_REAL_DEFAULT, 0, 0);
        }

        if (success != 1) {
            return 1;
        }

        std::copy(w.get(), w.get() + nev__, eval__);

        return 0;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        utils::timer t0("Eigensolver_elpa|solve_std");

        if (A__.num_cols_local() != Z__.num_cols_local()) {
            TERMINATE("number of columns in A and Z don't match");
        }

        int num_cols_loc = A__.num_cols_local();
        int bs           = A__.bs_row();
        int lda          = A__.ld();
        int ldz          = Z__.ld();
        int mpi_comm_row = MPI_Comm_c2f(A__.blacs_grid().comm_row().mpi_comm());
        int mpi_comm_col = MPI_Comm_c2f(A__.blacs_grid().comm_col().mpi_comm());
        int mpi_comm_all = MPI_Comm_c2f(A__.blacs_grid().comm().mpi_comm());

        auto w = mp_h_.get_unique_ptr<double>(matrix_size__);
        int success{-1};
        if (stage_ == 1) {
            success = elpa_solve_evp_complex_1stage_double_precision(
                matrix_size__, nev__, A__.at(memory_t::host), lda, w.get(), Z__.at(memory_t::host), ldz, bs,
                num_cols_loc, mpi_comm_row, mpi_comm_col, mpi_comm_all, 0);
        } else {
            success = elpa_solve_evp_complex_2stage_double_precision(
                matrix_size__, nev__, A__.at(memory_t::host), lda, w.get(), Z__.at(memory_t::host), ldz, bs,
                num_cols_loc, mpi_comm_row, mpi_comm_col, mpi_comm_all, ELPA_2STAGE_COMPLEX_DEFAULT, 0);
        }

        if (success != 1) {
            return 1;
        }

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
        utils::timer t0("Eigensolver_scalapack|pzheevd");
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

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__)
    {
        utils::timer t0("Eigensolver_scalapack|pdsyevx");

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
        utils::timer t0("Eigensolver_scalapack|pzheevx");

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
        utils::timer t0("Eigensolver_scalapack|pdsygvx");

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

        double work1;
        ftn_int lwork  = -1;
        ftn_int liwork = -1;
        /* work size query */
        FORTRAN(pdsygvx)
        (&ione, "V", "I", "U", &matrix_size__, A__.at(memory_t::host), &ione, &ione, desca, B__.at(memory_t::host),
         &ione, &ione, descb, &d1, &d1, &ione, &nev__, &abstol_, &m, &nz, w.get(), &ortfac_, Z__.at(memory_t::host),
         &ione, &ione, descz, &work1, &lwork, &liwork, &lwork, ifail.get(), iclustr.get(), gap.get(), &info, (ftn_int)1,
         (ftn_int)1, (ftn_int)1);

        lwork = static_cast<int32_t>(work1) + 4 * (1 << 20);

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
        utils::timer t0("Eigensolver_scalapack|pzhegvx");

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
        int result{-1};
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
        int result{-1};
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
        utils::timer t0("Eigensolver_magma|dsygvdx");

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
        utils::timer t0("Eigensolver_magma|zheevdx");

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

        magma_zheevdx(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
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
#else
class Eigensolver_magma: public Eigensolver
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
        utils::timer t0("Eigensolver_cuda|zheevd");

        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        auto w = mp_d_.get_unique_ptr<double>(matrix_size__);
        A__.copy_to(memory_t::device);

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
            acc::copy(Z__.at(memory_t::device), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
            Z__.copy_to(memory_t::host);
        }
        return info;
    }

    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
    }

    int solve(ftn_int matrix_size__, int nev__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
              dmatrix<double_complex>& Z__)
    {
        utils::timer t0("Eigensolver_cuda|zhegvd");

        cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

        auto w = mp_d_.get_unique_ptr<double>(matrix_size__);
        A__.copy_to(memory_t::device);
        B__.copy_to(memory_t::device);

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
            acc::copy(Z__.at(memory_t::device), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
            Z__.copy_to(memory_t::host);
        }
        return info;
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<double_complex>& A__, dmatrix<double_complex>& B__, double* eval__,
              dmatrix<double_complex>& Z__)
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

//#ifdef __PLASMA
//template <typename T>
//class Eigensolver_plasma: public Eigensolver<T>
//{
//  public:
//    inline bool is_parallel()
//    {
//        return false;
//    }
//};
//#else
//template <typename T>
//class Eigensolver_plasma: public Eigensolver<T>
//{
//  public:
//    inline bool is_parallel()
//    {
//        return false;
//    }
//};

std::unique_ptr<Eigensolver> Eigensolver_factory(ev_solver_t ev_solver_type__)
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
        case ev_solver_t::cusolver: {
            ptr = new Eigensolver_cuda();
            break;
        }
        default: {
            TERMINATE("not implemented");
        }
    }
    return std::move(std::unique_ptr<Eigensolver>(ptr));
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
//==
//== //#ifdef __MAGMA
//== //extern "C" int magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda,
//== //                                             void* b, int32_t ldb, double* eval);
//== //
//== //extern "C" int magma_dsygvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b,
//== //                                             int32_t ldb, double* eval);
//== //
//== //extern "C" int magma_dsyevdx_wrapper(int32_t matrix_size, int32_t nv, double* a, int32_t lda, double* eval);
//== //
//== //extern "C" int magma_zheevdx_wrapper(int32_t matrix_size, int32_t nv, double_complex* a, int32_t lda, double*
//eval);
//== //
//== //extern "C" int magma_zheevdx_2stage_wrapper(int32_t matrix_size, int32_t nv, cuDoubleComplex* a, int32_t lda,
//double* eval);
//== //#endif
//==
//== /// Interface for ScaLAPACK eigen-value solvers.
//== class Eigenproblem_scalapack: public Eigenproblem
//== {
//==     private:
//==
//==         int32_t bs_row_;
//==         int32_t bs_col_;
//==         int num_ranks_row_;
//==         int num_ranks_col_;
//==         int blacs_context_;
//==         double abstol_;
//==
//==         //== #ifdef __SCALAPACK
//==         //== std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol,
//==         //==                                     int blacs_context) const
//==         //== {
//==         //==     std::vector<int32_t> work_sizes(3);
//==         //==
//==         //==     int32_t nn = std::max(matrix_size, std::max(nb, 2));
//==         //==
//==         //==     int32_t np0 = linalg_base::numroc(nn, nb, 0, 0, nprow);
//==         //==     int32_t mq0 = linalg_base::numroc(nn, nb, 0, 0, npcol);
//==         //==
//==         //==     work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;
//==         //==
//==         //==     work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;
//==         //==
//==         //==     work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;
//==         //==
//==         //==     return work_sizes;
//==         //== }
//==
//==         //== std::vector<int32_t> get_work_sizes_gevp(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t
//npcol,
//==         //==                                          int blacs_context) const
//==         //== {
//==         //==     std::vector<int32_t> work_sizes(3);
//==         //==
//==         //==     int32_t nn = std::max(matrix_size, std::max(nb, 2));
//==         //==
//==         //==     int32_t neig = std::max(1024, nb);
//==
//==         //==     int32_t nmax3 = std::max(neig, std::max(nb, 2));
//==         //==
//==         //==     int32_t np = nprow * npcol;
//==
//==         //==     // due to the mess in the documentation, take the maximum of np0, nq0, mq0
//==         //==     int32_t nmpq0 = std::max(linalg_base::numroc(nn, nb, 0, 0, nprow),
//==         //==                           std::max(linalg_base::numroc(nn, nb, 0, 0, npcol),
//==         //==                                    linalg_base::numroc(nmax3, nb, 0, 0, npcol)));
//==
//==         //==     int32_t anb = linalg_base::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
//==         //==     int32_t sqnpc = (int32_t)pow(double(np), 0.5);
//==         //==     int32_t nps = std::max(linalg_base::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);
//==
//==         //==     work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
//==         //==     work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) *
//nps);
//==         //==     work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);
//==
//==         //==     work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) +
//==         //==                     linalg_base::iceil(neig, np) * nn + neig * matrix_size;
//==
//==         //==     int32_t nnp = std::max(matrix_size, std::max(np + 1, 4));
//==         //==     work_sizes[2] = 6 * nnp;
//==
//==         //==     return work_sizes;
//==         //== }
//==         //== #endif
//==
//==     public:
//==
//==         Eigenproblem_scalapack(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__, double abstol__
//= 1e-12)
//==             : bs_row_(bs_row__),
//==               bs_col_(bs_col__),
//==               num_ranks_row_(blacs_grid__.num_ranks_row()),
//==               num_ranks_col_(blacs_grid__.num_ranks_col()),
//==               blacs_context_(blacs_grid__.context()),
//==               abstol_(abstol__)
//==         {
//==         }
//==
//==         #ifdef __SCALAPACK
//==         int solve(int32_t         matrix_size,
//==                   double_complex* A,
//==                   int32_t         lda,
//==                   double*         eval,
//==                   double_complex* Z,
//==                   int32_t         ldz) const
//==         {
//==             int desca[9];
//==             linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
//==
//==             int descz[9];
//==             linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
//==
//==             int32_t info;
//==             int32_t ione = 1;
//==
//==             int32_t lwork = -1;
//==             int32_t lrwork = -1;
//==             int32_t liwork = -1;
//==             std::vector<double_complex> work(1);
//==             std::vector<double> rwork(1);
//==             std::vector<int32_t> iwork(1);
//==
//==             /* work size query */
//==             FORTRAN(pzheevd)("V", "U", &matrix_size, A, &ione, &ione, desca, eval, Z, &ione, &ione, descz,
//&work[0],
//==                              &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (int32_t)1,
//==                              (int32_t)1);
//==
//==             lwork = static_cast<int32_t>(work[0].real()) + 1;
//==             lrwork = static_cast<int32_t>(rwork[0]) + 1;
//==             liwork = iwork[0];
//==
//==             work = std::vector<double_complex>(lwork);
//==             rwork = std::vector<double>(lrwork);
//==             iwork = std::vector<int32_t>(liwork);
//==
//==             FORTRAN(pzheevd)("V", "U", &matrix_size, A, &ione, &ione, desca, eval, Z, &ione, &ione, descz,
//&work[0],
//==                              &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (int32_t)1,
//==                              (int32_t)1);
//==
//==             if (info)
//==             {
//==                 std::stringstream s;
//==                 s << "pzheevd returned " << info;
//==                 TERMINATE(s);
//==             }
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size,  int32_t nevec,
//==                   double_complex* A, int32_t lda,
//==                   double_complex* B, int32_t ldb,
//==                   double* eval,
//==                   double_complex* Z, int32_t ldz,
//==                   int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             int32_t desca[9];
//==             linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
//==
//==             int32_t descb[9];
//==             linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldb);
//==
//==             int32_t descz[9];
//==             linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
//==
//==             //std::vector<int32_t> work_sizes = get_work_sizes_gevp(matrix_size, std::max(bs_row_, bs_col_),
//==             //                                                      num_ranks_row_, num_ranks_col_,
//blacs_context_);
//==             //
//==
//==             std::vector<int32_t> ifail(matrix_size);
//==             std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> w(matrix_size);
//==
//==             double orfac = 1e-6;
//==             int32_t ione = 1;
//==
//==             int32_t m;
//==             int32_t nz;
//==             double d1;
//==             int32_t info;
//==
//==             int32_t lwork = -1;
//==             int32_t lrwork = -1;
//==             int32_t liwork = -1;
//==             std::vector<double_complex> work(1);
//==             std::vector<double> rwork(3);
//==             std::vector<int32_t> iwork(1);
//==             /* work size query */
//==             FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb,
//&d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info,
//==                              (int32_t)1, (int32_t)1, (int32_t)1);
//==             lwork = static_cast<int32_t>(work[0].real()) + 4096;
//==             lrwork = static_cast<int32_t>(rwork[0]) + 4096;
//==             liwork = iwork[0] + 4096;
//==
//==             work = std::vector<double_complex>(lwork);
//==             rwork = std::vector<double>(lrwork);
//==             iwork = std::vector<int32_t>(liwork);
//==
//==             FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb,
//&d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info,
//==                              (int32_t)1, (int32_t)1, (int32_t)1);
//==
//==             if (info)
//==             {
//==                 if ((info / 2) % 2)
//==                 {
//==                     std::stringstream s;
//==                     s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
//==                       << "could not be reorthogonalized because of insufficient workspace" << std::endl;
//==
//==                     int k = num_ranks_row_ * num_ranks_col_;
//==                     for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
//==                     {
//==                         if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
//==                         {
//==                             k = i + 1;
//==                             break;
//==                         }
//==                     }
//==
//==                     s << "number of eigenvalue clusters : " << k << std::endl;
//==                     for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl;
//==                     TERMINATE(s);
//==                 }
//==
//==                 std::stringstream s;
//==                 s << "pzhegvx returned " << info;
//==                 TERMINATE(s);
//==             }
//==
//==             if ((m != nevec) || (nz != nevec))
//==                 TERMINATE("Not all eigen-vectors or eigen-values are found.");
//==
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size,  int32_t nevec,
//==                   double* A, int32_t lda,
//==                   double* B, int32_t ldb,
//==                   double* eval,
//==                   double* Z, int32_t ldz,
//==                   int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             int32_t desca[9];
//==             linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
//==
//==             int32_t descb[9];
//==             linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldb);
//==
//==             int32_t descz[9];
//==             linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
//==
//==             int32_t lwork = -1;
//==             int32_t liwork;
//==
//==             double work1;
//==
//==             double orfac = 1e-6;
//==             int32_t ione = 1;
//==
//==             int32_t m;
//==             int32_t nz;
//==             double d1;
//==             int32_t info;
//==
//==             std::vector<int32_t> ifail(matrix_size);
//==             std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> w(matrix_size);
//==
//==             /* work size query */
//==             FORTRAN(pdsygvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb,
//&d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work1, &lwork,
//==                              &liwork, &lwork, &ifail[0], &iclustr[0], &gap[0], &info, (int32_t)1, (int32_t)1,
//(int32_t)1);
//==
//==             lwork = static_cast<int32_t>(work1) + 4 * (1 << 20);
//==
//==             std::vector<double> work(lwork);
//==             std::vector<int32_t> iwork(liwork);
//==
//==             FORTRAN(pdsygvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb,
//&d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info,
//==                              (int32_t)1, (int32_t)1, (int32_t)1);
//==
//==             if (info)
//==             {
//==                 if ((info / 2) % 2)
//==                 {
//==                     std::stringstream s;
//==                     s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
//==                       << "could not be reorthogonalized because of insufficient workspace" << std::endl;
//==
//==                     int k = num_ranks_row_ * num_ranks_col_;
//==                     for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
//==                     {
//==                         if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
//==                         {
//==                             k = i + 1;
//==                             break;
//==                         }
//==                     }
//==
//==                     s << "number of eigenvalue clusters : " << k << std::endl;
//==                     for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl;
//==                     TERMINATE(s);
//==                 }
//==
//==                 std::stringstream s;
//==                 s << "pzhegvx returned " << info;
//==                 TERMINATE(s);
//==             }
//==
//==             if ((m != nevec) || (nz != nevec))
//==                 TERMINATE("Not all eigen-vectors or eigen-values are found.");
//==
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size,
//==                   int32_t nevec,
//==                   double* A,
//==                   int32_t lda,
//==                   double* eval,
//==                   double* Z,
//==                   int32_t ldz,
//==                   int32_t num_rows_loc = 0,
//==                   int32_t num_cols_loc = 0) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             int32_t desca[9];
//==             linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
//==
//==             int32_t descz[9];
//==             linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
//==
//==             double orfac = 1e-6;
//==             int32_t ione = 1;
//==
//==             int32_t m;
//==             int32_t nz;
//==             double d1;
//==             int32_t info;
//==
//==             std::vector<int32_t> ifail(matrix_size);
//==             std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> w(matrix_size);
//==
//==             /* work size query */
//==             std::vector<double> work(3);
//==             std::vector<int32_t> iwork(1);
//==             int32_t lwork = -1;
//==             int32_t liwork = -1;
//==             FORTRAN(pdsyevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, (int32_t)1, (int32_t)1,
//(int32_t)1);
//==
//==             lwork = static_cast<int32_t>(work[0]) + 4 * (1 << 20);
//==             liwork = iwork[0];
//==
//==             work = std::vector<double>(lwork);
//==             iwork = std::vector<int32_t>(liwork);
//==
//==             FORTRAN(pdsyevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info,
//==                              (int32_t)1, (int32_t)1, (int32_t)1);
//==
//==             if (info)
//==             {
//==                 if ((info / 2) % 2)
//==                 {
//==                     std::stringstream s;
//==                     s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
//==                       << "could not be reorthogonalized because of insufficient workspace" << std::endl;
//==
//==                     int k = num_ranks_row_ * num_ranks_col_;
//==                     for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
//==                     {
//==                         if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
//==                         {
//==                             k = i + 1;
//==                             break;
//==                         }
//==                     }
//==
//==                     s << "number of eigenvalue clusters : " << k << std::endl;
//==                     for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl;
//==                     TERMINATE(s);
//==                 }
//==
//==                 std::stringstream s;
//==                 s << "pdsyevx returned " << info;
//==                 TERMINATE(s);
//==             }
//==
//==             if ((m != nevec) || (nz != nevec))
//==                 TERMINATE("Not all eigen-vectors or eigen-values are found.");
//==
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t         matrix_size,
//==                   int32_t         nevec,
//==                   double_complex* A,
//==                   int32_t         lda,
//==                   double*         eval,
//==                   double_complex* Z,
//==                   int32_t         ldz,
//==                   int32_t         num_rows_loc = 0,
//==                   int32_t         num_cols_loc = 0) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             int32_t desca[9];
//==             linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
//==
//==             int32_t descz[9];
//==             linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
//==
//==             double orfac = 1e-6;
//==             int32_t ione = 1;
//==
//==             int32_t m;
//==             int32_t nz;
//==             double d1;
//==             int32_t info;
//==
//==             std::vector<int32_t> ifail(matrix_size);
//==             std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
//==             std::vector<double> w(matrix_size);
//==
//==             /* work size query */
//==             std::vector<double_complex> work(3);
//==             std::vector<double> rwork(3);
//==             std::vector<int32_t> iwork(1);
//==             int32_t lwork = -1;
//==             int32_t lrwork = -1;
//==             int32_t liwork = -1;
//==             FORTRAN(pzheevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info,
//==                              (int32_t)1, (int32_t)1, (int32_t)1);
//==
//==             lwork = static_cast<int32_t>(work[0].real()) + (1 << 16);
//==             lrwork = static_cast<int32_t>(rwork[0]) + (1 << 16);
//==             liwork = iwork[0];
//==
//==             work = std::vector<double_complex>(lwork);
//==             rwork = std::vector<double>(lrwork);
//==             iwork = std::vector<int32_t>(liwork);
//==
//==             FORTRAN(pzheevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1,
//==                              &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione,
//descz, &work[0], &lwork,
//==                              &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info,
//==                              (int32_t)1, (int32_t)1, (int32_t)1);
//==
//==             if (info)
//==             {
//==                 if ((info / 2) % 2)
//==                 {
//==                     std::stringstream s;
//==                     s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl
//==                       << "could not be reorthogonalized because of insufficient workspace" << std::endl;
//==
//==                     int k = num_ranks_row_ * num_ranks_col_;
//==                     for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
//==                     {
//==                         if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
//==                         {
//==                             k = i + 1;
//==                             break;
//==                         }
//==                     }
//==
//==                     s << "number of eigenvalue clusters : " << k << std::endl;
//==                     for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl;
//==                     TERMINATE(s);
//==                 }
//==
//==                 std::stringstream s;
//==                 s << "pzheevx returned " << info;
//==                 TERMINATE(s);
//==             }
//==
//==             if ((m != nevec) || (nz != nevec))
//==                 TERMINATE("Not all eigen-vectors or eigen-values are found.");
//==
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==         #endif
//==
//==         bool parallel() const
//==         {
//==             return true;
//==         }
//==
//==         ev_solver_t type() const
//==         {
//==             return ev_scalapack;
//==         }
//== };
//==
//== class Eigenproblem_elpa: public Eigenproblem
//== {
//==     protected:
//==
//==         int32_t block_size_;
//==         int32_t num_ranks_row_;
//==         int32_t rank_row_;
//==         int32_t num_ranks_col_;
//==         int32_t rank_col_;
//==         int blacs_context_;
//==         Communicator const& comm_row_;
//==         Communicator const& comm_col_;
//==         Communicator const& comm_all_;
//==         int32_t mpi_comm_rows_;
//==         int32_t mpi_comm_cols_;
//==         int32_t mpi_comm_all_;
//==
//==     public:
//==
//==         Eigenproblem_elpa(BLACS_grid const& blacs_grid__, int32_t block_size__)
//==             : block_size_(block_size__),
//==               num_ranks_row_(blacs_grid__.num_ranks_row()),
//==               rank_row_(blacs_grid__.rank_row()),
//==               num_ranks_col_(blacs_grid__.num_ranks_col()),
//==               rank_col_(blacs_grid__.rank_col()),
//==               blacs_context_(blacs_grid__.context()),
//==               comm_row_(blacs_grid__.comm_row()),
//==               comm_col_(blacs_grid__.comm_col()),
//==               comm_all_(blacs_grid__.comm())
//==         {
//==             mpi_comm_rows_ = MPI_Comm_c2f(comm_row_.mpi_comm());
//==             mpi_comm_cols_ = MPI_Comm_c2f(comm_col_.mpi_comm());
//==             mpi_comm_all_  = MPI_Comm_c2f(comm_all_.mpi_comm());
//==         }
//==
//==         #ifdef __ELPA
//==         void transform_to_standard(int32_t matrix_size__,
//==                                    double_complex* A__, int32_t lda__,
//==                                    double_complex* B__, int32_t ldb__,
//==                                    int32_t num_rows_loc__, int32_t num_cols_loc__,
//==                                    matrix<double_complex>& tmp1__,
//==                                    matrix<double_complex>& tmp2__) const
//==         {
//==             PROFILE("Eigenproblem_elpa:transform_to_standard");
//==
//==             /* compute Cholesky decomposition of B: B=L*L^H; overwrite B with L */
//==             FORTRAN(elpa_cholesky_complex_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__,
//&mpi_comm_rows_, &mpi_comm_cols_);
//==             /* invert L */
//==             FORTRAN(elpa_invert_trm_complex_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__,
//&mpi_comm_rows_, &mpi_comm_cols_);
//==
//==             FORTRAN(elpa_mult_ah_b_complex_wrapper)("U", "L", &matrix_size__, &matrix_size__, B__, &ldb__,
//&num_cols_loc__, A__, &lda__, &num_cols_loc__, &block_size_,
//==                                                     &mpi_comm_rows_, &mpi_comm_cols_, tmp1__.at(memory_t::host),
//&num_rows_loc__, &num_cols_loc__,
//==                                                     (int32_t)1, (int32_t)1);
//==
//==             int32_t descc[9];
//==             linalg_base::descinit(descc, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0,
//blacs_context_, lda__);
//==
//==             linalg_base::pztranc(matrix_size__, matrix_size__, linalg_const<double_complex>::one(),
//tmp1__.at(memory_t::host), 1, 1, descc,
//==                                  linalg_const<double_complex>::zero(), tmp2__.at(memory_t::host), 1, 1, descc);
//==
//==             FORTRAN(elpa_mult_ah_b_complex_wrapper)("U", "U", &matrix_size__, &matrix_size__, B__, &ldb__,
//&num_cols_loc__,
//==                                                     tmp2__.at(memory_t::host), &num_rows_loc__, &num_cols_loc__,
//==                                                     &block_size_, &mpi_comm_rows_, &mpi_comm_cols_, A__, &lda__,
//&num_cols_loc__,
//==                                                     (int32_t)1, (int32_t)1);
//==
//==             linalg_base::pztranc(matrix_size__, matrix_size__, linalg_const<double_complex>::one(), A__, 1, 1,
//descc, linalg_const<double_complex>::zero(),
//==                                  tmp1__.at(memory_t::host), 1, 1, descc);
//==
//==             for (int i = 0; i < num_cols_loc__; i++)
//==             {
//==                 int32_t n_col = linalg_base::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
//==                 int32_t n_row = linalg_base::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
//==                 for (int j = n_row; j < num_rows_loc__; j++)
//==                 {
//==                     A__[j + i * lda__] = tmp1__(j, i);
//==                 }
//==             }
//==         }
//==
//==         void transform_to_standard(int32_t matrix_size__,
//==                                    double* A__, int32_t lda__,
//==                                    double* B__, int32_t ldb__,
//==                                    int32_t num_rows_loc__, int32_t num_cols_loc__,
//==                                    matrix<double>& tmp1__,
//==                                    matrix<double>& tmp2__) const
//==         {
//==             PROFILE("Eigenproblem_elpa:transform_to_standard");
//==
//==             /* compute Cholesky decomposition of B: B=L*L^H; overwrite B with L */
//==             FORTRAN(elpa_cholesky_real_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__,
//&mpi_comm_rows_, &mpi_comm_cols_);
//==             /* invert L */
//==             FORTRAN(elpa_invert_trm_real_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__,
//&mpi_comm_rows_, &mpi_comm_cols_);
//==
//==             FORTRAN(elpa_mult_at_b_real_wrapper)("U", "L", &matrix_size__, &matrix_size__, B__, &ldb__,
//&num_cols_loc__, A__, &lda__, &num_cols_loc__, &block_size_,
//==                                                  &mpi_comm_rows_, &mpi_comm_cols_, tmp1__.at(memory_t::host),
//&num_rows_loc__, &num_cols_loc__,
//==                                                  (int32_t)1, (int32_t)1);
//==
//==             int32_t descc[9];
//==             linalg_base::descinit(descc, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0,
//blacs_context_, lda__);
//==
//==             linalg_base::pdtran(matrix_size__, matrix_size__, 1.0, tmp1__.at(memory_t::host), 1, 1, descc, 0.0,
//==                                 tmp2__.at(memory_t::host), 1, 1, descc);
//==
//==             FORTRAN(elpa_mult_at_b_real_wrapper)("U", "U", &matrix_size__, &matrix_size__, B__, &ldb__,
//&num_cols_loc__, tmp2__.at(memory_t::host), &num_rows_loc__,
//==                                                 &num_cols_loc__, &block_size_, &mpi_comm_rows_, &mpi_comm_cols_,
//A__, &lda__, &num_cols_loc__,
//==                                                 (int32_t)1, (int32_t)1);
//==
//==             linalg_base::pdtran(matrix_size__, matrix_size__, 1.0, A__, 1, 1, descc, 0.0,
//tmp1__.at(memory_t::host), 1, 1, descc);
//==
//==             for (int i = 0; i < num_cols_loc__; i++)
//==             {
//==                 int32_t n_col = linalg_base::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
//==                 int32_t n_row = linalg_base::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
//==                 for (int j = n_row; j < num_rows_loc__; j++)
//==                 {
//==                     A__[j + i * lda__] = tmp1__(j, i);
//==                 }
//==             }
//==         }
//==
//==         void transform_back(int32_t matrix_size__, int32_t nevec__,
//==                             double_complex* B__, int32_t ldb__,
//==                             double_complex* Z__, int32_t ldz__,
//==                             int32_t num_rows_loc__, int32_t num_cols_loc__,
//==                             matrix<double_complex>& tmp1__,
//==                             matrix<double_complex>& tmp2__) const
//==         {
//==             PROFILE("Eigenproblem_elpa:transform_back");
//==
//==             int32_t descb[9];
//==             linalg_base::descinit(descb, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0,
//blacs_context_, ldb__);
//==
//==             linalg_base::pztranc(matrix_size__, matrix_size__, linalg_const<double_complex>::one(), B__, 1, 1,
//descb, linalg_const<double_complex>::zero(),
//==                                  tmp2__.at(memory_t::host), 1, 1, descb);
//==
//==             FORTRAN(elpa_mult_ah_b_complex_wrapper)("L", "N", &matrix_size__, &nevec__, tmp2__.at(memory_t::host),
//&num_rows_loc__, &num_cols_loc__, tmp1__.at(memory_t::host),
//==                                                     &num_rows_loc__, &num_cols_loc__, &block_size_,
//&mpi_comm_rows_, &mpi_comm_cols_, Z__, &ldz__, &num_cols_loc__,
//==                                                     (int32_t)1, (int32_t)1);
//==         }
//==
//==         void transform_back(int32_t matrix_size__, int32_t nevec__,
//==                             double* B__, int32_t ldb__,
//==                             double* Z__, int32_t ldz__,
//==                             int32_t num_rows_loc__, int32_t num_cols_loc__,
//==                             matrix<double>& tmp1__,
//==                             matrix<double>& tmp2__) const
//==         {
//==             PROFILE("Eigenproblem_elpa:transform_back");
//==
//==             int32_t descb[9];
//==             linalg_base::descinit(descb, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0,
//blacs_context_, ldb__);
//==
//==             linalg_base::pdtran(matrix_size__, matrix_size__, 1.0, B__, 1, 1, descb, 0.0,
//tmp2__.at(memory_t::host), 1, 1, descb);
//==
//==             FORTRAN(elpa_mult_at_b_real_wrapper)("L", "N", &matrix_size__, &nevec__, tmp2__.at(memory_t::host),
//&num_rows_loc__, &num_cols_loc__, tmp1__.at(memory_t::host),
//==                                                  &num_rows_loc__, &num_cols_loc__, &block_size_, &mpi_comm_rows_,
//&mpi_comm_cols_, Z__, &ldz__, &num_cols_loc__,
//==                                                  (int32_t)1, (int32_t)1);
//==         }
//==         #endif
//==
//== };
//==
//==
//== class Eigenproblem_elpa1: public Eigenproblem_elpa
//== {
//==     public:
//==
//==         Eigenproblem_elpa1(BLACS_grid const& blacs_grid__, int32_t block_size__)
//==             : Eigenproblem_elpa(blacs_grid__, block_size__)
//==         {
//==         }
//==
//==         #ifdef __ELPA
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double_complex* A, int32_t lda,
//==                   double_complex* B, int32_t ldb,
//==                   double* eval,
//==                   double_complex* Z, int32_t ldz,
//==                   int32_t num_rows_loc,
//==                   int32_t num_cols_loc) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             matrix<double_complex> tmp1(num_rows_loc, num_cols_loc);
//==             matrix<double_complex> tmp2(num_rows_loc, num_cols_loc);
//==
//==             transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa1|diag");
//==             FORTRAN(elpa_solve_evp_complex)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at(memory_t::host),
//&num_rows_loc,
//==                                             &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             t.stop();
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double* A, int32_t lda,
//==                   double* B, int32_t ldb,
//==                   double* eval,
//==                   double* Z, int32_t ldz,
//==                   int32_t num_rows_loc,
//==                   int32_t num_cols_loc) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             matrix<double> tmp1(num_rows_loc, num_cols_loc);
//==             matrix<double> tmp2(num_rows_loc, num_cols_loc);
//==
//==             transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa1|diag");
//==             FORTRAN(elpa_solve_evp_real)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at(memory_t::host),
//&num_rows_loc,
//==                                          &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double* A, int32_t lda,
//==                   double* eval,
//==                   double* Z, int32_t ldz,
//==                   int32_t num_rows_loc,
//==                   int32_t num_cols_loc) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa1|diag");
//==             FORTRAN(elpa_solve_evp_real)(&matrix_size, &nevec, A, &lda, &w[0], Z, &ldz,
//==                                          &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double_complex* A, int32_t lda,
//==                   double* eval,
//==                   double_complex* Z, int32_t ldz,
//==                   int32_t num_rows_loc, int32_t num_cols_loc) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa1|diag");
//==             FORTRAN(elpa_solve_evp_complex)(&matrix_size, &nevec, A, &lda, &w[0], Z, &ldz,
//==                                             &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==         #endif
//==
//==         bool parallel() const
//==         {
//==             return true;
//==         }
//==
//==         ev_solver_t type() const
//==         {
//==             return ev_elpa1;
//==         }
//== };
//==
//== class Eigenproblem_elpa2: public Eigenproblem_elpa
//== {
//==     public:
//==
//==         Eigenproblem_elpa2(BLACS_grid const& blacs_grid__, int32_t block_size__)
//==             : Eigenproblem_elpa(blacs_grid__, block_size__)
//==         {
//==         }
//==
//==         #ifdef __ELPA
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double_complex* A, int32_t lda,
//==                   double_complex* B, int32_t ldb,
//==                   double* eval,
//==                   double_complex* Z, int32_t ldz,
//==                   int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             matrix<double_complex> tmp1(num_rows_loc, num_cols_loc);
//==             matrix<double_complex> tmp2(num_rows_loc, num_cols_loc);
//==
//==             transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa2|diag");
//==             FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at(memory_t::host),
//&num_rows_loc,
//==                                                    &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double* A, int32_t lda,
//==                   double* B, int32_t ldb,
//==                   double* eval,
//==                   double* Z, int32_t ldz,
//==                   int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             matrix<double> tmp1(num_rows_loc, num_cols_loc);
//==             matrix<double> tmp2(num_rows_loc, num_cols_loc);
//==
//==             transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa2|diag");
//==             FORTRAN(elpa_solve_evp_real_2stage)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at(memory_t::host),
//&num_rows_loc,
//==                                                 &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);
//==
//==             return 0;
//==         }
//==
//==         int solve(int32_t matrix_size, int32_t nevec,
//==                   double* A, int32_t lda,
//==                   double* eval,
//==                   double* Z, int32_t ldz,
//==                   int32_t num_rows_loc, int32_t num_cols_loc) const
//==         {
//==             assert(nevec <= matrix_size);
//==
//==             std::vector<double> w(matrix_size);
//==             utils::timer t("Eigenproblem_elpa2|diag");
//==             FORTRAN(elpa_solve_evp_real_2stage)(&matrix_size, &nevec, A, &lda, &w[0], Z, &ldz,
//==                                                 &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_,
//&mpi_comm_all_);
//==             t.stop();
//==             std::memcpy(eval, &w[0], nevec * sizeof(double));
//==
//==             return 0;
//==         }
//==         #endif
//==
//==         bool parallel() const
//==         {
//==             return true;
//==         }
//==
//==         ev_solver_t type() const
//==         {
//==             return ev_elpa2;
//==         }
//== };
//==

#endif // __EIGENPROBLEM_HPP__

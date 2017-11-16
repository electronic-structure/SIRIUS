// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file eigenproblem.h
 *
 *  \brief Contains definition and implementaiton of various eigenvalue solver interfaces.
 */

#ifndef __EIGENPROBLEM_H__
#define __EIGENPROBLEM_H__

#include "constants.h"
#include "linalg.hpp"

#ifdef __ELPA
extern "C" {
void FORTRAN(elpa_cholesky_complex_wrapper)(ftn_int const*      na,
                                            ftn_double_complex* a,
                                            ftn_int const*      lda,
                                            ftn_int const*      nblk,
                                            ftn_int const*      matrixCols,
                                            ftn_int const*      mpi_comm_rows,
                                            ftn_int const*      mpi_comm_cols);

void FORTRAN(elpa_cholesky_real_wrapper)(ftn_int const* na,
                                         ftn_double*    a,
                                         ftn_int const* lda,
                                         ftn_int const* nblk,
                                         ftn_int const* matrixCols,
                                         ftn_int const* mpi_comm_rows,
                                         ftn_int const* mpi_comm_cols);

void FORTRAN(elpa_invert_trm_complex_wrapper)(ftn_int const*      na,
                                              ftn_double_complex* a,
                                              ftn_int const*      lda,
                                              ftn_int const*      nblk,
                                              ftn_int const*      matrixCols,
                                              ftn_int const*      mpi_comm_rows,
                                              ftn_int const*      mpi_comm_cols);

void FORTRAN(elpa_invert_trm_real_wrapper)(ftn_int const* na,
                                           ftn_double*    a,
                                           ftn_int const* lda,
                                           ftn_int const* nblk,
                                           ftn_int const* matrixCols,
                                           ftn_int const* mpi_comm_rows,
                                           ftn_int const* mpi_comm_cols);

void FORTRAN(elpa_mult_ah_b_complex_wrapper)(ftn_char            uplo_a,
                                             ftn_char            uplo_c,
                                             ftn_int const*      na,
                                             ftn_int const*      ncb, 
                                             ftn_double_complex* a,
                                             ftn_int const*      lda,
                                             ftn_int const*      ldaCols,
                                             ftn_double_complex* b,
                                             ftn_int const*      ldb,
                                             ftn_int const*      ldbCols,
                                             ftn_int const*      nblk,
                                             ftn_int const*      mpi_comm_rows,
                                             ftn_int const*      mpi_comm_cols,
                                             ftn_double_complex* c,
                                             ftn_int const*      ldc,
                                             ftn_int const*      ldcCols,
                                             ftn_len             uplo_a_len,
                                             ftn_len             uplo_c_len);

void FORTRAN(elpa_mult_at_b_real_wrapper)(ftn_char       uplo_a,
                                          ftn_char       uplo_c,
                                          ftn_int const* na,
                                          ftn_int const* ncb, 
                                          ftn_double*    a,
                                          ftn_int const* lda,
                                          ftn_int const* ldaCols,
                                          ftn_double*    b,
                                          ftn_int const* ldb,
                                          ftn_int const* ldbCols,
                                          ftn_int const* nblk,
                                          ftn_int const* mpi_comm_rows,
                                          ftn_int const* mpi_comm_cols,
                                          ftn_double*    c,
                                          ftn_int const* ldc,
                                          ftn_int const* ldcCols,
                                          ftn_len        uplo_a_len,
                                          ftn_len        uplo_c_len);

void FORTRAN(elpa_solve_evp_complex)(ftn_int const* na,
                                     ftn_int const* nev,
                                     ftn_double_complex* a,
                                     ftn_int const* lda,
                                     ftn_double* ev, 
                                     ftn_double_complex* q,
                                     ftn_int const* ldq,
                                     ftn_int const* nblk,
                                     ftn_int const* matrixCols,
                                     ftn_int const* mpi_comm_rows,
                                     ftn_int const* mpi_comm_cols,
                                     ftn_int const* mpi_comm_all);

void FORTRAN(elpa_solve_evp_real)(ftn_int const* na,
                                  ftn_int const* nev,
                                  ftn_double* a,
                                  ftn_int const* lda,
                                  ftn_double* ev, 
                                  ftn_double* q,
                                  ftn_int const* ldq,
                                  ftn_int const* nblk,
                                  ftn_int const* matrixCols,
                                  ftn_int const* mpi_comm_rows,
                                  ftn_int const* mpi_comm_cols,
                                  ftn_int const* mpi_comm_all);

void FORTRAN(elpa_solve_evp_complex_2stage)(ftn_int const* na,
                                            ftn_int const* nev,
                                            ftn_double_complex* a,
                                            ftn_int const* lda,
                                            ftn_double* ev,
                                            ftn_double_complex* q,
                                            ftn_int const* ldq,
                                            ftn_int const* nblk,
                                            ftn_int const* matrixCols,
                                            ftn_int const* mpi_comm_rows,
                                            ftn_int const* mpi_comm_cols,
                                            ftn_int const* mpi_comm_all);

void FORTRAN(elpa_solve_evp_real_2stage)(ftn_int const* na,
                                         ftn_int const* nev,
                                         ftn_double* a,
                                         ftn_int const* lda,
                                         ftn_double* ev,
                                         ftn_double* q,
                                         ftn_int const* ldq,
                                         ftn_int const* nblk,
                                         ftn_int const* matrixCols,
                                         ftn_int const* mpi_comm_rows,
                                         ftn_int const* mpi_comm_cols,
                                         ftn_int const* mpi_comm_all);
}
#endif

namespace experimental {

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
    plasma
};

template <typename T>
class Eigensolver
{
  public:

    virtual ~Eigensolver()
    {
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    virtual int solve(ftn_int matrix_size__, dmatrix<T>& A__, dmatrix<T>& B__, double* eval__, dmatrix<T>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    virtual int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, double* eval__, dmatrix<T>& Z__)
    {
        TERMINATE("solver is not implemented");
        return -1;
    }
};

template <typename T>
class Eigensolver_lapack: public Eigensolver<T>
{
  private:

    std::array<ftn_int, 3> get_work_sizes(ftn_int matrix_size) const
    {
        std::array<ftn_int, 3> work_sizes;
        
        work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
        work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
        work_sizes[2] = 3 + 5 * matrix_size;
        return work_sizes;
    }
    
  public:

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        auto work_sizes = get_work_sizes(matrix_size__);

        ftn_int info;
        ftn_int lda = A__.ld();

        if (std::is_same<T, double_complex>::value) {
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<ftn_int> iwork(work_sizes[2]);

            FORTRAN(zheevd)("V", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                            &lda, eval__, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                            &iwork[0], &work_sizes[2], &info, (ftn_int)1, (ftn_int)1);
            
            if (info) {
                std::stringstream s;
                s << "zheevd returned " << info; 
                TERMINATE(s);
            }
        }

        if (std::is_same<T, double>::value) {

            int32_t lwork = 1 + 6 * matrix_size__ + 2 * matrix_size__ * matrix_size__;
            int32_t liwork = 3 + 5 * matrix_size__; 
            
            std::vector<double> work(lwork);
            std::vector<int32_t> iwork(liwork);

            FORTRAN(dsyevd)("V", "U", &matrix_size__, reinterpret_cast<double*>(A__.template at<CPU>()), &lda,
                            eval__, &work[0], &lwork, 
                            &iwork[0], &liwork, &info, (ftn_int)1, (ftn_int)1);
            
            if (info) {
                std::stringstream s;
                s << "dsyevd returned " << info; 
                TERMINATE(s);
            }
        }

        for (int i = 0; i < matrix_size__; i++) {
            std::copy(A__.template at<CPU>(0, i), A__.template at<CPU>(0, i) + matrix_size__, Z__.template at<CPU>(0, i));
        }

        return 0;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        int32_t lwork{-1};
        double lwork1, vl, vu;
        int32_t il{1};
        int32_t m{-1};
        int32_t info;

        std::vector<double> w(matrix_size__);
        std::vector<int32_t> iwork(5 * matrix_size__);
        std::vector<int32_t> ifail(matrix_size__);

        ftn_int lda = A__.ld();
        ftn_int ldz = Z__.ld();

        double abs_tol{1e-12};

        if (std::is_same<T, double>::value) {

            FORTRAN(dsyevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.template at<CPU>()), &lda, 
                            &vl, &vu, &il, &nev__, &abs_tol, &m, &w[0], reinterpret_cast<double*>(Z__.template at<CPU>()),
                            &ldz, &lwork1, &lwork, &iwork[0], &ifail[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

            lwork = static_cast<int32_t>(lwork1 + 1);
            std::vector<double> work(lwork);

            FORTRAN(dsyevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.template at<CPU>()), &lda,
                            &vl, &vu, &il, &nev__, &abs_tol, &m, &w[0], reinterpret_cast<double*>(Z__.template at<CPU>()),
                            &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
            
            if (info) {
                std::stringstream s;
                s << "dsyevx returned " << info; 
                TERMINATE(s);
            }
        }

        if (std::is_same<T, double_complex>::value) {
            std::vector<double> rwork(7 * matrix_size__);
            std::vector<double_complex> work(3);

            FORTRAN(zheevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()), &lda,
                            &vl, &vu, &il, &nev__, &abs_tol, &m, 
                            &w[0], reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz, &work[0], &lwork,
                            &rwork[0], &iwork[0], &ifail[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

            lwork = static_cast<int32_t>(work[0].real()) + 1;
            work.resize(lwork);

            FORTRAN(zheevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()), &lda,
                            &vl, &vu, &il, &nev__, &abs_tol, &m, 
                            &w[0], reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz, &work[0], &lwork,
                            &rwork[0], &iwork[0], &ifail[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

            if (info) {
                std::stringstream s;
                s << "zheevx returned " << info; 
                TERMINATE(s);
            }
        }

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eign-values: " << nev__ << std::endl
              << "number of eign-values found: " << m;
            WARNING(s);
            return 1;
        }

        std::memcpy(eval__, &w[0], nev__ * sizeof(double));
            
        return 0;
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, double* eval__, dmatrix<T>& Z__)
    {
        ftn_int info;

        ftn_int lda = A__.ld();
        ftn_int ldb = B__.ld();
        ftn_int ldz = Z__.ld();

        double abs_tol{0};
        double vl{0};
        double vu{0};
        ftn_int ione{1};
        ftn_int m{0};
        std::vector<double> w(matrix_size__);
        std::vector<int32_t> ifail(matrix_size__);

        if (std::is_same<T, double_complex>::value) {

            int nb = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size__, 0, 0, 0);
            int lwork = (nb + 1) * matrix_size__;
            int lrwork = 7 * matrix_size__;
            int liwork = 5 * matrix_size__;
            
            std::vector<double_complex> work(lwork);
            std::vector<double> rwork(lrwork);
            std::vector<int32_t> iwork(liwork);
       
            FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                            &lda, reinterpret_cast<double_complex*>(B__.template at<CPU>()), &ldb, 
                            &vl, &vu, &ione, &nev__, &abs_tol, &m, &w[0], reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz, &work[0], 
                            &lwork, &rwork[0], &iwork[0], &ifail[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

            if (info) {
                std::stringstream s;
                s << "zhegvx returned " << info; 
                TERMINATE(s);
            }

        }

        if (std::is_same<T, double>::value) {

            int nb = linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size__, 0, 0, 0);
            int lwork = (nb + 3) * matrix_size__ + 1024;
            int liwork = 5 * matrix_size__;
            
            std::vector<double> work(lwork);
            std::vector<int32_t> iwork(liwork);
       
            FORTRAN(dsygvx)(&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.template at<CPU>()), &lda,
                            reinterpret_cast<double*>(B__.template at<CPU>()), &ldb, 
                            &vl, &vu, &ione, &nev__, &abs_tol, &m, &w[0], reinterpret_cast<double*>(Z__.template at<CPU>()),
                            &ldz, &work[0], &lwork,
                            &iwork[0], &ifail[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);

            if (info) {
                std::stringstream s;
                s << "dsygvx returned " << info; 
                TERMINATE(s);
            }
        }

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eign-values: " << nev__ << std::endl
              << "number of eign-values found: " << m << std::endl
              << "matrix size: " << matrix_size__;
            WARNING(s);
            return 1;
        }
        
        std::memcpy(eval__, &w[0], nev__ * sizeof(double));

        return 0;
    }
};

#ifdef __ELPA
template <typename T>
class Eigensolver_elpa: public Eigensolver<T>
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

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, double* eval__, dmatrix<T>& Z__)
    {
        if (A__.num_cols_local() != Z__.num_cols_local()) {
            TERMINATE("number of columns in A and Z don't match");
        }
        if (A__.bs_row() != A__.bs_col()) {
            TERMINATE("wrong block size");
        }
        
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
        linalg<CPU>::gemm(0, 0, matrix_size__, matrix_size__, matrix_size__, linalg_const<T>::one(), A__, B__,
                          linalg_const<T>::zero(), Z__);
        /* U^{-H} * Z = U{-H} * A * U^{-1} -> A */
        linalg<CPU>::gemm(2, 0, matrix_size__, matrix_size__, matrix_size__, linalg_const<T>::one(), B__, Z__,
                          linalg_const<T>::zero(), A__);

        int num_cols_loc = A__.num_cols_local();
        int bs = A__.bs_row();
        int lda = A__.ld();
        int ldz = Z__.ld();
        int mpi_comm_row = MPI_Comm_c2f(A__.blacs_grid().comm_row().mpi_comm());
        int mpi_comm_col = MPI_Comm_c2f(A__.blacs_grid().comm_col().mpi_comm());
        int mpi_comm_all = MPI_Comm_c2f(A__.blacs_grid().comm().mpi_comm());
        /* solve standard eigen-value problem with ELPA1 */
        if (std::is_same<T, double_complex>::value) {
            if (stage_ == 1) {
                FORTRAN(elpa_solve_evp_complex)(&matrix_size__, &nev__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                                                &lda, eval__, reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz,
                                                &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            } else {
                FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size__, &nev__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                                                       &lda, eval__, reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz,
                                                       &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            }
        }
        if (std::is_same<T, double>::value) {
            if (stage_ == 1) {
                FORTRAN(elpa_solve_evp_real)(&matrix_size__, &nev__, reinterpret_cast<double*>(A__.template at<CPU>()),
                                             &lda, eval__, reinterpret_cast<double*>(Z__.template at<CPU>()), &ldz,
                                             &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            } else {
                FORTRAN(elpa_solve_evp_real_2stage)(&matrix_size__, &nev__, reinterpret_cast<double*>(A__.template at<CPU>()),
                                                    &lda, eval__, reinterpret_cast<double*>(Z__.template at<CPU>()), &ldz,
                                                    &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            }
        }

        /* back-transform of eigen-vectors */
        linalg<CPU>::gemm(0, 0, matrix_size__, nev__, matrix_size__, linalg_const<T>::one(), B__, Z__,
                          linalg_const<T>::zero(), A__);
        A__ >> Z__;
        return 0;
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<T>& A__, dmatrix<T>& B__, double* eval__, dmatrix<T>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        if (A__.num_cols_local() != Z__.num_cols_local()) {
            TERMINATE("number of columns in A and Z don't match");
        }
        
        int num_cols_loc = A__.num_cols_local();
        int bs = A__.bs_row();
        int lda = A__.ld();
        int ldz = Z__.ld();
        int mpi_comm_row = MPI_Comm_c2f(A__.blacs_grid().comm_row().mpi_comm());
        int mpi_comm_col = MPI_Comm_c2f(A__.blacs_grid().comm_col().mpi_comm());
        int mpi_comm_all = MPI_Comm_c2f(A__.blacs_grid().comm().mpi_comm());
        /* solve standard eigen-value problem with ELPA1 */
        if (std::is_same<T, double_complex>::value) {
            if (stage_ == 1) {
                FORTRAN(elpa_solve_evp_complex)(&matrix_size__, &nev__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                                                &lda, eval__, reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz,
                                                &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            } else {
                FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size__, &nev__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                                                       &lda, eval__, reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ldz,
                                                       &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            }
        }
        if (std::is_same<T, double>::value) {
            if (stage_ == 1) {
                FORTRAN(elpa_solve_evp_real)(&matrix_size__, &nev__, reinterpret_cast<double*>(A__.template at<CPU>()),
                                             &lda, eval__, reinterpret_cast<double*>(Z__.template at<CPU>()), &ldz,
                                             &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            } else {
                FORTRAN(elpa_solve_evp_real_2stage)(&matrix_size__, &nev__, reinterpret_cast<double*>(A__.template at<CPU>()),
                                                    &lda, eval__, reinterpret_cast<double*>(Z__.template at<CPU>()), &ldz,
                                                    &bs, &num_cols_loc, &mpi_comm_row, &mpi_comm_col, &mpi_comm_all);
            }
        }

        return 0;
    }
    
    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        return solve(matrix_size__, matrix_size__, A__, eval__, Z__);
    }
};
#endif

template <typename T>
class Eigensolver_scalapack: public Eigensolver<T>
{
  public:
    /// Solve a standard eigen-value problem for all eigen-pairs.
    int solve(ftn_int matrix_size__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0, A__.blacs_grid().context(), A__.ld());
        
        int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0, Z__.blacs_grid().context(), Z__.ld());

        ftn_int info;
        ftn_int ione{1};

        ftn_int lwork{-1};
        ftn_int lrwork{-1};
        ftn_int liwork{-1};
        std::vector<double_complex> work(1);
        std::vector<double> rwork(1);
        std::vector<int32_t> iwork(1);

        if (std::is_same<T, double_complex>::value) {
            /* work size query */
            FORTRAN(pzheevd)("V", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                             &ione, &ione, desca, eval__, reinterpret_cast<double_complex*>(Z__.template at<CPU>()),
                             &ione, &ione, descz, &work[0], 
                             &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (ftn_int)1, (ftn_int)1);
            
            lwork = static_cast<int32_t>(work[0].real()) + 1;
            lrwork = static_cast<int32_t>(rwork[0]) + 1;
            liwork = iwork[0];

            work = std::vector<double_complex>(lwork);
            rwork = std::vector<double>(lrwork);
            iwork = std::vector<int32_t>(liwork);

            FORTRAN(pzheevd)("V", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                             &ione, &ione, desca, eval__, reinterpret_cast<double_complex*>(Z__.template at<CPU>()),
                             &ione, &ione, descz, &work[0], 
                             &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (ftn_int)1, (ftn_int)1);
            if (info) {
                std::stringstream s;
                s << "pzheevd returned " << info; 
                TERMINATE(s);
            }
        }
        if (std::is_same<T, double>::value) {
            TERMINATE("not implemented");
        }

        return 0;
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, double* eval__, dmatrix<T>& Z__)
    {
        int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0, A__.blacs_grid().context(), A__.ld());
        
        int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0, Z__.blacs_grid().context(), Z__.ld());
        
        double orfac{1e-6};
        double abs_tol{1e-12};
        int32_t ione{1};
        
        int32_t m{-1};
        int32_t nz{-1};
        double d1;
        int32_t info{-1};

        std::vector<int32_t> ifail(matrix_size__);
        std::vector<int32_t> iclustr(2 * A__.blacs_grid().comm().size());
        std::vector<double> gap(A__.blacs_grid().comm().size());
        std::vector<double> w(matrix_size__);

        if (std::is_same<T, double_complex>::value) {

            /* work size query */
            std::vector<double_complex> work(3);
            std::vector<double> rwork(3);
            std::vector<int32_t> iwork(1);
            int32_t lwork = -1;
            int32_t lrwork = -1;
            int32_t liwork = -1;
            FORTRAN(pzheevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()), 
                             &ione, &ione, desca, &d1, &d1, &ione, &nev__, &abs_tol, &m, &nz, &w[0], &orfac,
                             reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ione, &ione, descz, &work[0], &lwork,
                             &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (ftn_int)1, (ftn_int)1, (ftn_int)1); 
            
            lwork = static_cast<int32_t>(work[0].real()) + (1 << 16);
            lrwork = static_cast<int32_t>(rwork[0]) + (1 << 16);
            liwork = iwork[0];

            work = std::vector<double_complex>(lwork);
            rwork = std::vector<double>(lrwork);
            iwork = std::vector<int32_t>(liwork);

            FORTRAN(pzheevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double_complex*>(A__.template at<CPU>()),
                             &ione, &ione, desca, &d1, &d1, &ione, &nev__, &abs_tol, &m, &nz, &w[0], &orfac,
                             reinterpret_cast<double_complex*>(Z__.template at<CPU>()), &ione, &ione, descz, &work[0],
                             &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (ftn_int)1, (ftn_int)1, (ftn_int)1); 
        }

        if (std::is_same<T, double>::value) {

            /* work size query */
            std::vector<double> work(3);
            std::vector<int32_t> iwork(1);
            int32_t lwork{-1};
            int32_t liwork{-1};


            FORTRAN(pdsyevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.template at<CPU>()), &ione, &ione, desca, &d1, &d1, 
                             &ione, &nev__, &abs_tol, &m, &nz, &w[0], &orfac, reinterpret_cast<double*>(Z__.template at<CPU>()), &ione, &ione, descz, &work[0], &lwork, 
                             &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, (ftn_int)1, (ftn_int)1, (ftn_int)1); 
            
            lwork = static_cast<int32_t>(work[0]) + 4 * (1 << 20);
            liwork = iwork[0];
            
            work = std::vector<double>(lwork);
            iwork = std::vector<int32_t>(liwork);

            FORTRAN(pdsyevx)("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.template at<CPU>()), &ione, &ione, desca, &d1, &d1, 
                             &ione, &nev__, &abs_tol, &m, &nz, &w[0], &orfac, reinterpret_cast<double*>(Z__.template at<CPU>()), &ione, &ione, descz, &work[0], &lwork, 
                             &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (ftn_int)1, (ftn_int)1, (ftn_int)1); 

        }

        if (info) {
            if ((info / 2) % 2) {
                std::stringstream s;
                s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                  << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                int k = A__.blacs_grid().comm().size();
                for (int i = 0; i < A__.blacs_grid().comm().size() - 1; i++) {
                    if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0)) {
                        k = i + 1;
                        break;
                    }
                }
               
                s << "number of eigenvalue clusters : " << k << std::endl;
                for (int i = 0; i < k; i++) {
                    s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                }
                TERMINATE(s);
            }

            std::stringstream s;
            s << "pdsyevx returned " << info; 
            TERMINATE(s);
        }

        if ((m != nev__) || (nz != nev__)) {
            TERMINATE("Not all eigen-vectors or eigen-values are found.");
        }

        std::memcpy(eval__, &w[0], nev__ * sizeof(double));

        return 0;
    }
};

template <typename T>
std::unique_ptr<Eigensolver<T>> Eigensolver_factory(ev_solver_t ev_solver_type__)
{
    Eigensolver<T>* ptr;
    switch (ev_solver_type__) {
        case ev_solver_t::lapack: {
            ptr = new Eigensolver_lapack<T>();
            break;
        }
        case ev_solver_t::scalapack: {
            ptr = new Eigensolver_scalapack<T>();
            break;
        }
        #ifdef __ELPA
        case ev_solver_t::elpa1: {
            ptr = new Eigensolver_elpa<T>(1);
            break;
        }
        case ev_solver_t::elpa2: {
            ptr = new Eigensolver_elpa<T>(2);
            break;
        }
        #endif
        default: {
            TERMINATE("not implemented");
        }
    }
    return std::move(std::unique_ptr<Eigensolver<T>>(ptr));
}

} // namespace

// TODO: simplify the interface, use only dmatrix<T> as input
//
// need to solve Ax = E*x  (standard)
//               Ax = E*Bx (generalized)
// for N lowes eigen-pairs
// for all eigen-pairs

/// Type of the solver to use for the standard or generalized eigen-value problem
enum ev_solver_t 
{
    /// use LAPACK
    ev_lapack, 

    /// use ScaLAPACK
    ev_scalapack,

    /// use ELPA1 solver
    ev_elpa1,

    /// use ELPA2 (2-stage) solver
    ev_elpa2,

    /// use MAGMA
    ev_magma,

    /// use PLASMA
    ev_plasma,

    ev_rs_gpu,

    ev_rs_cpu
};

/// Base class for eigen-value problems.
class Eigenproblem
{
    public:

        virtual ~Eigenproblem()
        {
        }
        
        /// Complex generalized Hermitian-definite eigenproblem.
        virtual int solve(int32_t         matrix_size,
                          int32_t         nevec,
                          double_complex* A,
                          int32_t         lda,
                          double_complex* B,
                          int32_t         ldb,
                          double*         eval,
                          double_complex* Z,
                          int32_t         ldz,
                          int32_t         num_rows_loc = 0,
                          int32_t         num_cols_loc = 0) const
        {
            TERMINATE("solver is not implemented");
            return -1;
        }

        /// Real generalized symmetric-definite eigenproblem.
        virtual int solve(int32_t matrix_size,
                          int32_t nevec,
                          double* A,
                          int32_t lda,
                          double* B,
                          int32_t ldb,
                          double* eval,
                          double* Z,
                          int32_t ldz,
                          int32_t num_rows_loc = 0,
                          int32_t num_cols_loc = 0) const
        {
            TERMINATE("solver is not implemented");
            return -1;
        }

        /// Complex standard Hermitian-definite eigenproblem with with N lowest eigen-vectors.
        virtual int solve(int32_t         matrix_size,
                          int32_t         nevec,
                          double_complex* A,
                          int32_t         lda,
                          double*         eval,
                          double_complex* Z,
                          int32_t         ldz,
                          int32_t         num_rows_loc = 0,
                          int32_t         num_cols_loc = 0) const
        {
            TERMINATE("solver is not implemented");
            return -1;
        }

        /// Real standard symmetric-definite eigenproblem with N lowest eigen-vectors.
        virtual int solve(int32_t matrix_size,
                          int32_t nevec,
                          double* A,
                          int32_t lda,
                          double* eval,
                          double* Z,
                          int32_t ldz,
                          int32_t num_rows_loc = 0,
                          int32_t num_cols_loc = 0) const
        {
            TERMINATE("solver is not implemented");
            return -1;
        }
        
        /// Complex standard Hermitian-definite eigenproblem with all eigen-vectors.
        virtual int solve(int32_t         matrix_size,
                          double_complex* A,
                          int32_t         lda,
                          double*         eval,
                          double_complex* Z,
                          int32_t         ldz) const
        {
            TERMINATE("standard eigen-value solver is not configured");
            return -1;
        }

        /// Real standard symmetric-definite eigenproblem with all eigen-vectors.
        virtual int solve(int32_t matrix_size,
                          double* A,
                          int32_t lda,
                          double* eval,
                          double* Z,
                          int32_t ldz) const
        {
            TERMINATE("standard eigen-value solver is not configured");
            return -1;
        }

        virtual bool parallel() const = 0;

        virtual ev_solver_t type() const = 0;
};

/// Interface for LAPACK eigen-value solvers.
class Eigenproblem_lapack: public Eigenproblem
{
    private:

        double abstol_;
    
        std::vector<int32_t> get_work_sizes(int32_t matrix_size) const
        {
            std::vector<int32_t> work_sizes(3);
            
            work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
            work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
            work_sizes[2] = 3 + 5 * matrix_size;
            return work_sizes;
        }

    public:

        Eigenproblem_lapack(double abstol__ = 1e-12) : abstol_(abstol__)
        {
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            int nb = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
            int lwork = (nb + 1) * matrix_size;
            int lrwork = 7 * matrix_size;
            int liwork = 5 * matrix_size;
            
            std::vector<double_complex> work(lwork);
            std::vector<double> rwork(lrwork);
            std::vector<int32_t> iwork(liwork);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> w(matrix_size);
            double vl = 0.0;
            double vu = 0.0;
            int32_t m;
            int32_t info;
       
            int32_t ione = 1;
            FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size, A, &lda, B, &ldb, &vl, &vu, &ione, &nevec, const_cast<double*>(&abstol_), &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &rwork[0], &iwork[0], &ifail[0], &info, (int32_t)1, 
                            (int32_t)1, (int32_t)1);

            if (m != nevec)
            {
                std::stringstream s;
                s << "not all eigen-values are found" << std::endl
                  << "target number of eign-values: " << nevec << std::endl
                  << "number of eign-values found: " << m << std::endl
                  << "matrix size: " << matrix_size;
                WARNING(s);
                return 1;
            }

            if (info)
            {
                std::stringstream s;
                s << "zhegvx returned " << info; 
                TERMINATE(s);
            }

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* B, int32_t ldb,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            int nb = linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size, 0, 0, 0);
            int lwork = (nb + 3) * matrix_size + 1024;
            int liwork = 5 * matrix_size;
            
            std::vector<double> work(lwork);
            std::vector<int32_t> iwork(liwork);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> w(matrix_size);
            double vl = 0.0;
            double vu = 0.0;
            int32_t m;
            int32_t info;
       
            int32_t ione = 1;
            FORTRAN(dsygvx)(&ione, "V", "I", "U", &matrix_size, A, &lda, B, &ldb, &vl, &vu, &ione, &nevec, const_cast<double*>(&abstol_), &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info, (int32_t)1, 
                            (int32_t)1, (int32_t)1);

            if (m != nevec)
            {
                //== std::stringstream s;
                //== s << "not all eigen-values are found" << std::endl
                //==   << "target number of eign-values: " << nevec << std::endl
                //==   << "number of eign-values found: " << m;
                //== WARNING(s);
                return 1;
            }

            if (info)
            {
                std::stringstream s;
                s << "dsygvx returned " << info; 
                TERMINATE(s);
            }

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t ldz) const
        {
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            FORTRAN(zheevd)("V", "U", &matrix_size, A, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                            &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);
            
            for (int i = 0; i < matrix_size; i++)
                std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double_complex));
            
            if (info)
            {
                std::stringstream s;
                s << "zheevd returned " << info; 
                TERMINATE(s);
            }
            return 0;
        }

        int solve(int32_t matrix_size, double* A, int32_t lda, double* eval, double* Z, int32_t ldz) const
        {
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size);

            int32_t lwork = 1 + 6 * matrix_size + 2 * matrix_size * matrix_size;
            int32_t liwork = 3 + 5 * matrix_size; 
            
            std::vector<double> work(lwork);
            std::vector<int32_t> iwork(liwork);
            int32_t info;

            FORTRAN(dsyevd)("V", "U", &matrix_size, A, &lda, eval, &work[0], &lwork, 
                            &iwork[0], &liwork, &info, (int32_t)1, (int32_t)1);
            
            for (int i = 0; i < matrix_size; i++)
                std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double));
            
            if (info)
            {
                std::stringstream s;
                s << "dsyevd returned " << info; 
                TERMINATE(s);
            }
            return 0;
        }

        int solve(int32_t matrix_size,
                  int nevec,
                  double* A,
                  int32_t lda,
                  double* eval,
                  double* Z,
                  int32_t ldz,
                  int32_t num_rows_loc = 0,
                  int32_t num_cols_loc = 0) const
        {
            int32_t lwork = -1;
            double lwork1, vl, vu;
            int32_t il, iu, m, info;
            std::vector<double> w(matrix_size);
            std::vector<int32_t> iwork(5 * matrix_size);
            std::vector<int32_t> ifail(matrix_size);

            il = 1;
            iu = nevec;

            FORTRAN(dsyevx)("V", "I", "U", &matrix_size, A, &lda, &vl, &vu, &il, &iu, const_cast<double*>(&abstol_), &m, 
                            &w[0], Z, &ldz, &lwork1, &lwork, &iwork[0], &ifail[0], &info, (int32_t)1, (int32_t)1, (int32_t)1);

            lwork = static_cast<int32_t>(lwork1 + 1);
            std::vector<double> work(lwork);

            FORTRAN(dsyevx)("V", "I", "U", &matrix_size, A, &lda, &vl, &vu, &il, &iu, const_cast<double*>(&abstol_), &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &iwork[0], &ifail[0], &info, (int32_t)1, (int32_t)1, (int32_t)1);
            
            if (m != nevec)
            {
                std::stringstream s;
                s << "not all eigen-values are found" << std::endl
                  << "target number of eign-values: " << nevec << std::endl
                  << "number of eign-values found: " << m;
                WARNING(s);
                return 1;
            }

            if (info)
            {
                std::stringstream s;
                s << "dsyevx returned " << info; 
                TERMINATE(s);
            }

            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            return 0;
        }

        int solve(int32_t         matrix_size,
                  int             nevec,
                  double_complex* A,
                  int32_t         lda,
                  double*         eval,
                  double_complex* Z,
                  int32_t         ldz,
                  int32_t         num_rows_loc = 0,
                  int32_t         num_cols_loc = 0) const
        {
            int32_t lwork = -1;
            double vl, vu;
            int32_t il, iu, m, info;
            std::vector<double> w(matrix_size);
            std::vector<int32_t> iwork(5 * matrix_size);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> rwork(7 * matrix_size);
            std::vector<double_complex> work(3);

            il = 1;
            iu = nevec;

            FORTRAN(zheevx)("V", "I", "U", &matrix_size, A, &lda, &vl, &vu, &il, &iu, const_cast<double*>(&abstol_), &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &rwork[0], &iwork[0], &ifail[0], &info,
                            (int32_t)1, (int32_t)1, (int32_t)1);

            lwork = static_cast<int32_t>(work[0].real()) + 1;
            work.resize(lwork);

            FORTRAN(zheevx)("V", "I", "U", &matrix_size, A, &lda, &vl, &vu, &il, &iu, const_cast<double*>(&abstol_), &m, 
                            &w[0], Z, &ldz, &work[0], &lwork, &rwork[0], &iwork[0], &ifail[0], &info,
                            (int32_t)1, (int32_t)1, (int32_t)1);
            
            if (m != nevec) {
                std::stringstream s;
                s << "not all eigen-values are found" << std::endl
                  << "target number of eign-values: " << nevec << std::endl
                  << "number of eign-values found: " << m;
                WARNING(s);
                return 1;
            }

            if (info) {
                std::stringstream s;
                s << "dsyevx returned " << info; 
                TERMINATE(s);
            }

            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            return 0;
        }

        bool parallel() const
        {
            return false;
        }

        ev_solver_t type() const
        {
            return ev_lapack;
        }
};

#ifdef __PLASMA
extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
                                      int32_t ldz, double* eval);
#endif

/// Interface for PLASMA eigen-value solvers.
class Eigenproblem_plasma: public Eigenproblem
{
    public:

        Eigenproblem_plasma()
        {
        }

        #ifdef __PLASMA
        void solve(int32_t matrix_size, double_complex* A, int32_t lda, double* eval, double_complex* Z, int32_t ldz) const
        {
            //plasma_set_num_threads(1);
            //omp_set_num_threads(1);
            //printf("before call to plasma_zheevd_wrapper\n");
            plasma_zheevd_wrapper(matrix_size, a, lda, z, lda, eval);
            //printf("after call to plasma_zheevd_wrapper\n");
            //plasma_set_num_threads(8);
            //omp_set_num_threads(8);
        }
        #endif
        
        bool parallel() const
        {
            return false;
        }

        ev_solver_t type() const
        {
            return ev_plasma;
        }
};

//#ifdef __MAGMA
//extern "C" int magma_zhegvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, 
//                                             void* b, int32_t ldb, double* eval);
//
//extern "C" int magma_dsygvdx_2stage_wrapper(int32_t matrix_size, int32_t nv, void* a, int32_t lda, void* b, 
//                                             int32_t ldb, double* eval);
//
//extern "C" int magma_dsyevdx_wrapper(int32_t matrix_size, int32_t nv, double* a, int32_t lda, double* eval);
//
//extern "C" int magma_zheevdx_wrapper(int32_t matrix_size, int32_t nv, double_complex* a, int32_t lda, double* eval);
//
//extern "C" int magma_zheevdx_2stage_wrapper(int32_t matrix_size, int32_t nv, cuDoubleComplex* a, int32_t lda, double* eval);
//#endif

/// Interface for MAGMA eigen-value solvers.
class Eigenproblem_magma: public Eigenproblem
{
    public:

        Eigenproblem_magma()
        {
        }

        #ifdef __MAGMA
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            int nt = omp_get_max_threads();
            
            int result = magma::zhegvdx_2stage(matrix_size, nevec, A, lda, B, ldb, eval);

            if (nt != omp_get_max_threads()) {
                TERMINATE("magma has changed the number of threads");
            }

            if (result) {
                return Eigenproblem_lapack().solve(matrix_size, nevec, A, lda, B, ldb, eval, Z, ldz);
            } else {
                #pragma omp parallel for
                for (int i = 0; i < nevec; i++) {
                    std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double_complex));
                }

                return 0;
            }
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* B, int32_t ldb,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            int nt = omp_get_max_threads();
            
            int result = magma::dsygvdx_2stage(matrix_size, nevec, A, lda, B, ldb, eval);

            if (nt != omp_get_max_threads()) {
                TERMINATE("magma has changed the number of threads");
            }

            if (result) {
                return Eigenproblem_lapack().solve(matrix_size, nevec, A, lda, B, ldb, eval, Z, ldz);
            } else {
                #pragma omp parallel for
                for (int i = 0; i < nevec; i++) {
                    std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double));
                }

                return 0;
            }
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            int nt = omp_get_max_threads();
            
            int result = magma::dsyevdx(matrix_size, nevec, A, lda, eval);

            if (nt != omp_get_max_threads()) {
                TERMINATE("magma has changed the number of threads");
            }

            if (result) {
                return Eigenproblem_lapack().solve(matrix_size, nevec, A, lda, eval, Z, ldz);
            } else {
                #pragma omp parallel for
                for (int i = 0; i < nevec; i++) {
                    std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double));
                }

                return 0;
            }
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            int nt = omp_get_max_threads();
            
            int result = magma::zheevdx(matrix_size, nevec, (magmaDoubleComplex*)A, lda, eval);

            if (nt != omp_get_max_threads()) {
                TERMINATE("magma has changed the number of threads");
            }

            if (result) {
                return Eigenproblem_lapack().solve(matrix_size, nevec, A, lda, eval, Z, ldz);
            } else {
                #pragma omp parallel for
                for (int i = 0; i < nevec; i++) {
                    std::memcpy(&Z[ldz * i], &A[lda * i], matrix_size * sizeof(double_complex));
                }
                return 0;
            } 
        }
        #endif

        bool parallel() const
        {
            return false;
        }

        ev_solver_t type() const
        {
            return ev_magma;
        }
};

/// Interface for ScaLAPACK eigen-value solvers.
class Eigenproblem_scalapack: public Eigenproblem
{
    private:

        int32_t bs_row_;
        int32_t bs_col_;
        int num_ranks_row_;
        int num_ranks_col_;
        int blacs_context_;
        double abstol_;
        
        //== #ifdef __SCALAPACK
        //== std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
        //==                                     int blacs_context) const
        //== {
        //==     std::vector<int32_t> work_sizes(3);
        //==     
        //==     int32_t nn = std::max(matrix_size, std::max(nb, 2));
        //==     
        //==     int32_t np0 = linalg_base::numroc(nn, nb, 0, 0, nprow);
        //==     int32_t mq0 = linalg_base::numroc(nn, nb, 0, 0, npcol);
        //== 
        //==     work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;
        //== 
        //==     work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;
        //== 
        //==     work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;
        //==     
        //==     return work_sizes;
        //== }

        //== std::vector<int32_t> get_work_sizes_gevp(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
        //==                                          int blacs_context) const
        //== {
        //==     std::vector<int32_t> work_sizes(3);
        //==     
        //==     int32_t nn = std::max(matrix_size, std::max(nb, 2));
        //==     
        //==     int32_t neig = std::max(1024, nb);

        //==     int32_t nmax3 = std::max(neig, std::max(nb, 2));
        //==     
        //==     int32_t np = nprow * npcol;

        //==     // due to the mess in the documentation, take the maximum of np0, nq0, mq0
        //==     int32_t nmpq0 = std::max(linalg_base::numroc(nn, nb, 0, 0, nprow), 
        //==                           std::max(linalg_base::numroc(nn, nb, 0, 0, npcol),
        //==                                    linalg_base::numroc(nmax3, nb, 0, 0, npcol))); 

        //==     int32_t anb = linalg_base::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
        //==     int32_t sqnpc = (int32_t)pow(double(np), 0.5);
        //==     int32_t nps = std::max(linalg_base::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);

        //==     work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
        //==     work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
        //==     work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);

        //==     work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) + 
        //==                     linalg_base::iceil(neig, np) * nn + neig * matrix_size;

        //==     int32_t nnp = std::max(matrix_size, std::max(np + 1, 4));
        //==     work_sizes[2] = 6 * nnp;

        //==     return work_sizes;
        //== }
        //== #endif

    public:

        Eigenproblem_scalapack(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__, double abstol__ = 1e-12)
            : bs_row_(bs_row__),
              bs_col_(bs_col__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              blacs_context_(blacs_grid__.context()),
              abstol_(abstol__)
        {
        }

        #ifdef __SCALAPACK
        int solve(int32_t         matrix_size,
                  double_complex* A,
                  int32_t         lda,
                  double*         eval,
                  double_complex* Z,
                  int32_t         ldz) const
        {
            int desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);
            
            int descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);

            int32_t info;
            int32_t ione = 1;

            int32_t lwork = -1;
            int32_t lrwork = -1;
            int32_t liwork = -1;
            std::vector<double_complex> work(1);
            std::vector<double> rwork(1);
            std::vector<int32_t> iwork(1);

            /* work size query */
            FORTRAN(pzheevd)("V", "U", &matrix_size, A, &ione, &ione, desca, eval, Z, &ione, &ione, descz, &work[0], 
                             &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (int32_t)1, 
                             (int32_t)1);
            
            lwork = static_cast<int32_t>(work[0].real()) + 1;
            lrwork = static_cast<int32_t>(rwork[0]) + 1;
            liwork = iwork[0];

            work = std::vector<double_complex>(lwork);
            rwork = std::vector<double>(lrwork);
            iwork = std::vector<int32_t>(liwork);

            FORTRAN(pzheevd)("V", "U", &matrix_size, A, &ione, &ione, desca, eval, Z, &ione, &ione, descz, &work[0], 
                             &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (int32_t)1, 
                             (int32_t)1);

            if (info)
            {
                std::stringstream s;
                s << "pzheevd returned " << info; 
                TERMINATE(s);
            }

            return 0;
        }

        int solve(int32_t matrix_size,  int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldb); 

            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz); 

            //std::vector<int32_t> work_sizes = get_work_sizes_gevp(matrix_size, std::max(bs_row_, bs_col_), 
            //                                                      num_ranks_row_, num_ranks_col_, blacs_context_);
            //

            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);
            
            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;
            
            int32_t lwork = -1;
            int32_t lrwork = -1;
            int32_t liwork = -1;
            std::vector<double_complex> work(1);
            std::vector<double> rwork(3);
            std::vector<int32_t> iwork(1);
            /* work size query */
            FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork, 
                             &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 
            lwork = static_cast<int32_t>(work[0].real()) + 4096;
            lrwork = static_cast<int32_t>(rwork[0]) + 4096;
            liwork = iwork[0] + 4096;

            work = std::vector<double_complex>(lwork);
            rwork = std::vector<double>(lrwork);
            iwork = std::vector<int32_t>(liwork);
            
            FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork, 
                             &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    TERMINATE(s);
                }

                std::stringstream s;
                s << "pzhegvx returned " << info; 
                TERMINATE(s);
            }

            if ((m != nevec) || (nz != nevec))
                TERMINATE("Not all eigen-vectors or eigen-values are found.");

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t matrix_size,  int32_t nevec, 
                  double* A, int32_t lda,
                  double* B, int32_t ldb,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldb);

            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
            
            int32_t lwork = -1;
            int32_t liwork;

            double work1;

            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;

            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);
            
            /* work size query */
            FORTRAN(pdsygvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work1, &lwork, 
                             &liwork, &lwork, &ifail[0], &iclustr[0], &gap[0], &info, (int32_t)1, (int32_t)1, (int32_t)1); 
            
            lwork = static_cast<int32_t>(work1) + 4 * (1 << 20);
            
            std::vector<double> work(lwork);
            std::vector<int32_t> iwork(liwork);
            
            FORTRAN(pdsygvx)(&ione, "V", "I", "U", &matrix_size, A, &ione, &ione, desca, B, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork, 
                             &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    TERMINATE(s);
                }

                std::stringstream s;
                s << "pzhegvx returned " << info; 
                TERMINATE(s);
            }

            if ((m != nevec) || (nz != nevec))
                TERMINATE("Not all eigen-vectors or eigen-values are found.");

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t matrix_size,
                  int32_t nevec, 
                  double* A,
                  int32_t lda,
                  double* eval, 
                  double* Z,
                  int32_t ldz,
                  int32_t num_rows_loc = 0,
                  int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);

            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
            
            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;

            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);

            /* work size query */
            std::vector<double> work(3);
            std::vector<int32_t> iwork(1);
            int32_t lwork = -1;
            int32_t liwork = -1;
            FORTRAN(pdsyevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork, 
                             &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, (int32_t)1, (int32_t)1, (int32_t)1); 
            
            lwork = static_cast<int32_t>(work[0]) + 4 * (1 << 20);
            liwork = iwork[0];
            
            work = std::vector<double>(lwork);
            iwork = std::vector<int32_t>(liwork);

            FORTRAN(pdsyevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork, 
                             &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    TERMINATE(s);
                }

                std::stringstream s;
                s << "pdsyevx returned " << info; 
                TERMINATE(s);
            }

            if ((m != nevec) || (nz != nevec))
                TERMINATE("Not all eigen-vectors or eigen-values are found.");

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }

        int solve(int32_t         matrix_size,
                  int32_t         nevec, 
                  double_complex* A,
                  int32_t         lda,
                  double*         eval, 
                  double_complex* Z,
                  int32_t         ldz,
                  int32_t         num_rows_loc = 0,
                  int32_t         num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, lda);

            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, blacs_context_, ldz);
            
            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;

            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);

            /* work size query */
            std::vector<double_complex> work(3);
            std::vector<double> rwork(3);
            std::vector<int32_t> iwork(1);
            int32_t lwork = -1;
            int32_t lrwork = -1;
            int32_t liwork = -1;
            FORTRAN(pzheevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork,
                             &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 
            
            lwork = static_cast<int32_t>(work[0].real()) + (1 << 16);
            lrwork = static_cast<int32_t>(rwork[0]) + (1 << 16);
            liwork = iwork[0];

            work = std::vector<double_complex>(lwork);
            rwork = std::vector<double>(lrwork);
            iwork = std::vector<int32_t>(liwork);

            FORTRAN(pzheevx)("V", "I", "U", &matrix_size, A, &ione, &ione, desca, &d1, &d1, 
                             &ione, &nevec, const_cast<double*>(&abstol_), &m, &nz, &w[0], &orfac, Z, &ione, &ione, descz, &work[0], &lwork,
                             &rwork[0], &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    TERMINATE(s);
                }

                std::stringstream s;
                s << "pzheevx returned " << info; 
                TERMINATE(s);
            }

            if ((m != nevec) || (nz != nevec))
                TERMINATE("Not all eigen-vectors or eigen-values are found.");

            std::memcpy(eval, &w[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel() const
        {
            return true;
        }

        ev_solver_t type() const
        {
            return ev_scalapack;
        }
};

#ifdef __ELPA
class Eigenproblem_elpa: public Eigenproblem
{
    protected:
        
        int32_t block_size_;
        int32_t num_ranks_row_;
        int32_t rank_row_;
        int32_t num_ranks_col_;
        int32_t rank_col_;
        int blacs_context_;
        Communicator const& comm_row_;
        Communicator const& comm_col_;
        Communicator const& comm_all_;
        int32_t mpi_comm_rows_;
        int32_t mpi_comm_cols_;
        int32_t mpi_comm_all_;

    public:

        Eigenproblem_elpa(BLACS_grid const& blacs_grid__, int32_t block_size__)
            : block_size_(block_size__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context()), 
              comm_row_(blacs_grid__.comm_row()), 
              comm_col_(blacs_grid__.comm_col()),
              comm_all_(blacs_grid__.comm())
        {
            mpi_comm_rows_ = MPI_Comm_c2f(comm_row_.mpi_comm());
            mpi_comm_cols_ = MPI_Comm_c2f(comm_col_.mpi_comm());
            mpi_comm_all_  = MPI_Comm_c2f(comm_all_.mpi_comm());
        }
        
        #ifdef __ELPA
        void transform_to_standard(int32_t matrix_size__,
                                   double_complex* A__, int32_t lda__,
                                   double_complex* B__, int32_t ldb__,
                                   int32_t num_rows_loc__, int32_t num_cols_loc__,
                                   matrix<double_complex>& tmp1__,
                                   matrix<double_complex>& tmp2__) const
        {
            PROFILE("Eigenproblem_elpa:transform_to_standard");
            
            /* compute Cholesky decomposition of B: B=L*L^H; overwrite B with L */
            FORTRAN(elpa_cholesky_complex_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__, &mpi_comm_rows_, &mpi_comm_cols_);
            /* invert L */
            FORTRAN(elpa_invert_trm_complex_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__, &mpi_comm_rows_, &mpi_comm_cols_);
       
            FORTRAN(elpa_mult_ah_b_complex_wrapper)("U", "L", &matrix_size__, &matrix_size__, B__, &ldb__, &num_cols_loc__, A__, &lda__, &num_cols_loc__, &block_size_, 
                                                    &mpi_comm_rows_, &mpi_comm_cols_, tmp1__.at<CPU>(), &num_rows_loc__, &num_cols_loc__,
                                                    (int32_t)1, (int32_t)1);

            int32_t descc[9];
            linalg_base::descinit(descc, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0, blacs_context_, lda__);
            
            linalg_base::pztranc(matrix_size__, matrix_size__, linalg_const<double_complex>::one(), tmp1__.at<CPU>(), 1, 1, descc, 
                                 linalg_const<double_complex>::zero(), tmp2__.at<CPU>(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex_wrapper)("U", "U", &matrix_size__, &matrix_size__, B__, &ldb__, &num_cols_loc__, 
                                                    tmp2__.at<CPU>(), &num_rows_loc__, &num_cols_loc__,
                                                    &block_size_, &mpi_comm_rows_, &mpi_comm_cols_, A__, &lda__, &num_cols_loc__,
                                                    (int32_t)1, (int32_t)1);

            linalg_base::pztranc(matrix_size__, matrix_size__, linalg_const<double_complex>::one(), A__, 1, 1, descc, linalg_const<double_complex>::zero(), 
                                 tmp1__.at<CPU>(), 1, 1, descc);

            for (int i = 0; i < num_cols_loc__; i++)
            {
                int32_t n_col = linalg_base::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg_base::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < num_rows_loc__; j++) 
                {
                    A__[j + i * lda__] = tmp1__(j, i);
                }
            }
        }

        void transform_to_standard(int32_t matrix_size__,
                                   double* A__, int32_t lda__,
                                   double* B__, int32_t ldb__,
                                   int32_t num_rows_loc__, int32_t num_cols_loc__,
                                   matrix<double>& tmp1__,
                                   matrix<double>& tmp2__) const
        {
            PROFILE("Eigenproblem_elpa:transform_to_standard");
            
            /* compute Cholesky decomposition of B: B=L*L^H; overwrite B with L */
            FORTRAN(elpa_cholesky_real_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__, &mpi_comm_rows_, &mpi_comm_cols_);
            /* invert L */
            FORTRAN(elpa_invert_trm_real_wrapper)(&matrix_size__, B__, &ldb__, &block_size_, &num_cols_loc__, &mpi_comm_rows_, &mpi_comm_cols_);
       
            FORTRAN(elpa_mult_at_b_real_wrapper)("U", "L", &matrix_size__, &matrix_size__, B__, &ldb__, &num_cols_loc__, A__, &lda__, &num_cols_loc__, &block_size_, 
                                                 &mpi_comm_rows_, &mpi_comm_cols_, tmp1__.at<CPU>(), &num_rows_loc__, &num_cols_loc__, 
                                                 (int32_t)1, (int32_t)1);

            int32_t descc[9];
            linalg_base::descinit(descc, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0, blacs_context_, lda__);
            
            linalg_base::pdtran(matrix_size__, matrix_size__, 1.0, tmp1__.at<CPU>(), 1, 1, descc, 0.0,
                                tmp2__.at<CPU>(), 1, 1, descc);

            FORTRAN(elpa_mult_at_b_real_wrapper)("U", "U", &matrix_size__, &matrix_size__, B__, &ldb__, &num_cols_loc__, tmp2__.at<CPU>(), &num_rows_loc__, 
                                                &num_cols_loc__, &block_size_, &mpi_comm_rows_, &mpi_comm_cols_, A__, &lda__, &num_cols_loc__, 
                                                (int32_t)1, (int32_t)1);

            linalg_base::pdtran(matrix_size__, matrix_size__, 1.0, A__, 1, 1, descc, 0.0, tmp1__.at<CPU>(), 1, 1, descc);

            for (int i = 0; i < num_cols_loc__; i++)
            {
                int32_t n_col = linalg_base::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg_base::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < num_rows_loc__; j++) 
                {
                    A__[j + i * lda__] = tmp1__(j, i);
                }
            }
        }

        void transform_back(int32_t matrix_size__, int32_t nevec__,
                            double_complex* B__, int32_t ldb__,
                            double_complex* Z__, int32_t ldz__,
                            int32_t num_rows_loc__, int32_t num_cols_loc__,
                            matrix<double_complex>& tmp1__,
                            matrix<double_complex>& tmp2__) const
        {
            PROFILE("Eigenproblem_elpa:transform_back");

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0, blacs_context_, ldb__);

            linalg_base::pztranc(matrix_size__, matrix_size__, linalg_const<double_complex>::one(), B__, 1, 1, descb, linalg_const<double_complex>::zero(), 
                                 tmp2__.at<CPU>(), 1, 1, descb);

            FORTRAN(elpa_mult_ah_b_complex_wrapper)("L", "N", &matrix_size__, &nevec__, tmp2__.at<CPU>(), &num_rows_loc__, &num_cols_loc__, tmp1__.at<CPU>(), 
                                                    &num_rows_loc__, &num_cols_loc__, &block_size_, &mpi_comm_rows_, &mpi_comm_cols_, Z__, &ldz__, &num_cols_loc__,
                                                    (int32_t)1, (int32_t)1);
        }

        void transform_back(int32_t matrix_size__, int32_t nevec__,
                            double* B__, int32_t ldb__,
                            double* Z__, int32_t ldz__,
                            int32_t num_rows_loc__, int32_t num_cols_loc__,
                            matrix<double>& tmp1__,
                            matrix<double>& tmp2__) const
        {
            PROFILE("Eigenproblem_elpa:transform_back");

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size__, matrix_size__, block_size_, block_size_, 0, 0, blacs_context_, ldb__);

            linalg_base::pdtran(matrix_size__, matrix_size__, 1.0, B__, 1, 1, descb, 0.0, tmp2__.at<CPU>(), 1, 1, descb);

            FORTRAN(elpa_mult_at_b_real_wrapper)("L", "N", &matrix_size__, &nevec__, tmp2__.at<CPU>(), &num_rows_loc__, &num_cols_loc__, tmp1__.at<CPU>(), 
                                                 &num_rows_loc__, &num_cols_loc__, &block_size_, &mpi_comm_rows_, &mpi_comm_cols_, Z__, &ldz__, &num_cols_loc__,
                                                 (int32_t)1, (int32_t)1);
        }
        #endif

};


class Eigenproblem_elpa1: public Eigenproblem_elpa
{
    public:
        
        Eigenproblem_elpa1(BLACS_grid const& blacs_grid__, int32_t block_size__)
            : Eigenproblem_elpa(blacs_grid__, block_size__)
        {
        }
        
        #ifdef __ELPA
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc,
                  int32_t num_cols_loc) const
        {
            assert(nevec <= matrix_size);

            matrix<double_complex> tmp1(num_rows_loc, num_cols_loc);
            matrix<double_complex> tmp2(num_rows_loc, num_cols_loc);
            
            transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa1|diag");
            FORTRAN(elpa_solve_evp_complex)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at<CPU>(), &num_rows_loc, 
                                            &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);

            t.stop();
            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* B, int32_t ldb,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc,
                  int32_t num_cols_loc) const
        {
            assert(nevec <= matrix_size);

            matrix<double> tmp1(num_rows_loc, num_cols_loc);
            matrix<double> tmp2(num_rows_loc, num_cols_loc);

            transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa1|diag");
            FORTRAN(elpa_solve_evp_real)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at<CPU>(), &num_rows_loc, 
                                         &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);

            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc,
                  int32_t num_cols_loc) const
        {
            assert(nevec <= matrix_size);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa1|diag");
            FORTRAN(elpa_solve_evp_real)(&matrix_size, &nevec, A, &lda, &w[0], Z, &ldz, 
                                         &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc) const
        {
            assert(nevec <= matrix_size);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa1|diag");
            FORTRAN(elpa_solve_evp_complex)(&matrix_size, &nevec, A, &lda, &w[0], Z, &ldz, 
                                            &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            return 0;
        }
        #endif

        bool parallel() const
        {
            return true;
        }

        ev_solver_t type() const
        {
            return ev_elpa1;
        }
};

class Eigenproblem_elpa2: public Eigenproblem_elpa
{
    public:
        
        Eigenproblem_elpa2(BLACS_grid const& blacs_grid__, int32_t block_size__)
            : Eigenproblem_elpa(blacs_grid__, block_size__)
        {
        }

        #ifdef __ELPA
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            matrix<double_complex> tmp1(num_rows_loc, num_cols_loc);
            matrix<double_complex> tmp2(num_rows_loc, num_cols_loc);

            transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa2|diag");
            FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at<CPU>(), &num_rows_loc, 
                                                   &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);

            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* B, int32_t ldb,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc = 0, int32_t num_cols_loc = 0) const
        {
            assert(nevec <= matrix_size);

            matrix<double> tmp1(num_rows_loc, num_cols_loc);
            matrix<double> tmp2(num_rows_loc, num_cols_loc);

            transform_to_standard(matrix_size, A, lda, B, ldb, num_rows_loc, num_cols_loc, tmp1, tmp2);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa2|diag");
            FORTRAN(elpa_solve_evp_real_2stage)(&matrix_size, &nevec, A, &lda, &w[0], tmp1.at<CPU>(), &num_rows_loc, 
                                                &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            transform_back(matrix_size, nevec, B, ldb, Z, ldz, num_rows_loc, num_cols_loc, tmp1, tmp2);

            return 0;
        }

        int solve(int32_t matrix_size, int32_t nevec, 
                  double* A, int32_t lda,
                  double* eval, 
                  double* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc) const
        {
            assert(nevec <= matrix_size);

            std::vector<double> w(matrix_size);
            sddk::timer t("Eigenproblem_elpa2|diag");
            FORTRAN(elpa_solve_evp_real_2stage)(&matrix_size, &nevec, A, &lda, &w[0], Z, &ldz, 
                                                &block_size_, &num_cols_loc, &mpi_comm_rows_, &mpi_comm_cols_, &mpi_comm_all_);
            t.stop();
            std::memcpy(eval, &w[0], nevec * sizeof(double));
            
            return 0;
        }
        #endif
        
        bool parallel() const
        {
            return true;
        }

        ev_solver_t type() const
        {
            return ev_elpa2;
        }
};
#endif

#ifdef __RS_GEN_EIG
void libevs_gen_eig(char uplo, int n, int nev,
                    double_complex* a, int ia, int ja, int* desca,
                    double_complex* b, int ib, int jb, int* descb,
                    double* d,
                    double_complex* q, int iq, int jq, int* descq,
                    int* info);

void libevs_gen_eig_cpu(char uplo, int n, int nev,
                        double_complex* a, int ia, int ja, int* desca,
                        double_complex* b, int ib, int jb, int* descb,
                        double* d,
                        double_complex* q, int iq, int jq, int* descq,
                        int* info);
#endif

class Eigenproblem_RS_CPU: public Eigenproblem
{
    private:

        int32_t bs_row_;
        int32_t bs_col_;
        int num_ranks_row_;
        int rank_row_;
        int num_ranks_col_;
        int rank_col_;
        int blacs_context_;
        
    public:

        Eigenproblem_RS_CPU(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__)
            : bs_row_(bs_row__),
              bs_col_(bs_col__),
              num_ranks_row_(blacs_grid__.num_ranks_row()),
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context())
        {
        }

        #ifdef __RS_GEN_EIG
        int solve(int32_t matrix_size, int32_t nevec, 
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc) const
        {
        
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, 
                                  blacs_context_, lda);

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, 
                                  blacs_context_, ldb); 

            mdarray<double_complex, 2> ztmp(num_rows_loc, num_cols_loc);
            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, 
                                        blacs_context_, num_rows_loc); 
            
            std::vector<double> eval_tmp(matrix_size);

            int info;
            libevs_gen_eig_cpu('L', matrix_size, nevec, A, 1, 1, desca, B, 1, 1, descb, &eval_tmp[0], ztmp.at<CPU>(), 1, 1, descz, &info);
            if (info) {
                std::stringstream s;
                s << "libevs_gen_eig_cpu: info=" << info; 
                TERMINATE(s);
            }

            for (int i = 0; i < linalg_base::numroc(nevec, bs_col_, rank_col_, 0, num_ranks_col_); i++) {
                std::memcpy(&Z[ldz * i], &ztmp(0, i), num_rows_loc * sizeof(double_complex));
            }

            std::memcpy(eval, &eval_tmp[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel() const
        {
            return true;
        }

        ev_solver_t type() const
        {
            return ev_rs_cpu;
        }

};

class Eigenproblem_RS_GPU: public Eigenproblem
{
    private:

        int32_t bs_row_;
        int32_t bs_col_;
        int num_ranks_row_;
        int rank_row_;
        int num_ranks_col_;
        int rank_col_;
        int blacs_context_;
        
    public:

        Eigenproblem_RS_GPU(BLACS_grid const& blacs_grid__, int32_t bs_row__, int32_t bs_col__)
            : bs_row_(bs_row__),
              bs_col_(bs_col__),
              num_ranks_row_(blacs_grid__.num_ranks_row()),
              rank_row_(blacs_grid__.rank_row()),
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              blacs_context_(blacs_grid__.context())
        {
        }

        #ifdef __RS_GEN_EIG
        int solve(int32_t matrix_size, int32_t nevec,
                  double_complex* A, int32_t lda,
                  double_complex* B, int32_t ldb,
                  double* eval, 
                  double_complex* Z, int32_t ldz,
                  int32_t num_rows_loc, int32_t num_cols_loc) const
        {
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg_base::descinit(desca, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, 
                                  blacs_context_, lda);

            int32_t descb[9];
            linalg_base::descinit(descb, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, 
                                  blacs_context_, ldb); 

            mdarray<double_complex, 2> ztmp(num_rows_loc, num_cols_loc, memory_t::host_pinned);
            int32_t descz[9];
            linalg_base::descinit(descz, matrix_size, matrix_size, bs_row_, bs_col_, 0, 0, 
                                  blacs_context_, num_rows_loc); 
            
            std::vector<double> eval_tmp(matrix_size);

            int info;
            libevs_gen_eig('L', matrix_size, nevec, A, 1, 1, desca, B, 1, 1, descb, &eval_tmp[0], ztmp.at<CPU>(), 1, 1, descz, &info);
            if (info) {
                std::stringstream s;
                s << "libevs_gen_eig: info=" << info; 
                TERMINATE(s);
            }

            for (int i = 0; i < linalg_base::numroc(nevec, bs_col_, rank_col_, 0, num_ranks_col_); i++) {
                std::memcpy(&Z[ldz * i], &ztmp(0, i), num_rows_loc * sizeof(double_complex));
            }

            std::memcpy(eval, &eval_tmp[0], nevec * sizeof(double));

            return 0;
        }
        #endif

        bool parallel() const
        {
            return true;
        }

        ev_solver_t type() const
        {
            return ev_rs_gpu;
        }
};

#endif // __EIGENPROBLEM_H__


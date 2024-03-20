/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file eigenproblem.hpp
 *
 *  \brief Contains definition and implementation of various eigenvalue solver interfaces.
 */

#ifndef __EIGENPROBLEM_HPP__
#define __EIGENPROBLEM_HPP__

#include "core/profiler.hpp"
#include "core/rte/rte.hpp"
#include "core/omp.hpp"
#include "linalg.hpp"
#include "eigensolver.hpp"
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
#include "core/acc/magma.hpp"
#endif

#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
#include "core/acc/cusolver.hpp"
#endif

#if defined(SIRIUS_DLAF)
#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/eigensolver/gen_eigensolver.h>
#endif

namespace sirius {

namespace la {

class Eigensolver_lapack : public Eigensolver
{
  public:
    Eigensolver_lapack()
        : Eigensolver(ev_solver_t::lapack, false, memory_t::host, memory_t::host)
    {
    }

    /// wrapper for solving a standard eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_lapack|dsyevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_lapack|zheevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_lapack|ssyevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_lapack|cheevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    template <typename T>
    int
    solve_(ftn_int matrix_size__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
        ftn_int info;
        ftn_int lda = A__.ld();

        ftn_int lwork;
        ftn_int liwork = 3 + 5 * matrix_size__;
        ftn_int lrwork = 1 + 5 * matrix_size__ + 2 * matrix_size__ * matrix_size__; // only required in complex

        if (std::is_scalar<T>::value) {
            lwork = 1 + 6 * matrix_size__ + 2 * matrix_size__ * matrix_size__;
        } else {
            lwork = 2 * matrix_size__ + matrix_size__ * matrix_size__;
        }

        auto& mph = get_memory_pool(memory_t::host);

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);
        auto rwork = mph.get_unique_ptr<real_type<T>>(lrwork); // only required in complex

        if (std::is_same<T, double>::value) {
            FORTRAN(dsyevd)
            ("V", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<double*>(eval__), reinterpret_cast<double*>(work.get()), &lwork, iwork.get(), &liwork,
             &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(zheevd)
            ("V", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<double*>(eval__), reinterpret_cast<std::complex<double>*>(work.get()), &lwork,
             reinterpret_cast<double*>(rwork.get()), &lrwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(ssyevd)
            ("V", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<float*>(eval__), reinterpret_cast<float*>(work.get()), &lwork, iwork.get(), &liwork,
             &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(cheevd)
            ("V", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<float*>(eval__), reinterpret_cast<std::complex<float>*>(work.get()), &lwork,
             reinterpret_cast<float*>(rwork.get()), &lrwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        }

        if (!info) {
            for (int i = 0; i < matrix_size__; i++) {
                std::copy(A__.at(memory_t::host, 0, i), A__.at(memory_t::host, 0, i) + matrix_size__,
                          Z__.at(memory_t::host, 0, i));
            }
        }
        return info;
    }

    /// wrapper for solving a standard eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_lapack|dsyevr");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_lapack|zheevx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_lapack|ssyevr");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_lapack|cheevx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    template <typename T>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
        real_type<T> vl, vu;

        ftn_int il{1};
        ftn_int m{-1};
        ftn_int info;

        auto& mph = get_memory_pool(memory_t::host);

        auto w      = mph.get_unique_ptr<real_type<T>>(matrix_size__);
        auto isuppz = mph.get_unique_ptr<ftn_int>(2 * matrix_size__); // for real matrix
        auto ifail  = mph.get_unique_ptr<ftn_int>(matrix_size__);     // for complex matrix

        ftn_int lda = A__.ld();
        ftn_int ldz = Z__.ld();

        real_type<T> abs_tol = 2 * linalg_base::dlamch('S');

        ftn_int liwork;
        ftn_int lwork;
        int nb;
        ftn_int lrwork = 7 * matrix_size__; // only require in complex

        liwork = 10 * matrix_size__;
        if (std::is_same<T, double>::value) {
            nb     = std::max(linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size__, -1, -1, -1),
                              linalg_base::ilaenv(1, "DORMTR", "U", matrix_size__, -1, -1, -1));
            liwork = 10 * matrix_size__;
            lwork  = std::max((nb + 6) * matrix_size__, 26 * matrix_size__);
        } else if (std::is_same<T, std::complex<double>>::value) {
            nb     = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size__, -1, -1, -1);
            liwork = 5 * matrix_size__;
            lwork  = (nb + 1) * matrix_size__;
        } else if (std::is_same<T, float>::value) {
            nb     = std::max(linalg_base::ilaenv(1, "SSYTRD", "U", matrix_size__, -1, -1, -1),
                              linalg_base::ilaenv(1, "SORMTR", "U", matrix_size__, -1, -1, -1));
            liwork = 10 * matrix_size__;
            lwork  = std::max((nb + 6) * matrix_size__, 26 * matrix_size__);
        } else if (std::is_same<T, std::complex<float>>::value) {
            nb     = linalg_base::ilaenv(1, "CHETRD", "U", matrix_size__, -1, -1, -1);
            liwork = 5 * matrix_size__;
            lwork  = (nb + 1) * matrix_size__;
        }

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);
        auto rwork = mph.get_unique_ptr<real_type<T>>(lrwork); // only required in complex

        if (std::is_same<T, double>::value) {
            FORTRAN(dsyevr)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<double*>(&vl), reinterpret_cast<double*>(&vu), &il, &nev__,
             reinterpret_cast<double*>(&abs_tol), &m, reinterpret_cast<double*>(w.get()),
             reinterpret_cast<double*>(Z__.at(memory_t::host)), &ldz, isuppz.get(),
             reinterpret_cast<double*>(work.get()), &lwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1,
             (ftn_int)1);
            lwork = std::max((nb + 6) * matrix_size__, 26 * matrix_size__);
        } else if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(zheevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<double*>(&vl), reinterpret_cast<double*>(&vu), &il, &nev__,
             reinterpret_cast<double*>(&abs_tol), &m, reinterpret_cast<double*>(w.get()),
             reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), &ldz,
             reinterpret_cast<std::complex<double>*>(work.get()), &lwork, reinterpret_cast<double*>(rwork.get()),
             iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(ssyevr)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &nev__,
             reinterpret_cast<float*>(&abs_tol), &m, reinterpret_cast<float*>(w.get()),
             reinterpret_cast<float*>(Z__.at(memory_t::host)), &ldz, isuppz.get(), reinterpret_cast<float*>(work.get()),
             &lwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(cheevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<float*>(&vl), reinterpret_cast<float*>(&vu), &il, &nev__,
             reinterpret_cast<float*>(&abs_tol), &m, reinterpret_cast<float*>(w.get()),
             reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), &ldz,
             reinterpret_cast<std::complex<float>*>(work.get()), &lwork, reinterpret_cast<float*>(rwork.get()),
             iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        }

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eigen-values: " << nev__ << std::endl
              << "number of eigen-values found: " << m << std::endl
              << "matrix_size : " << matrix_size__ << std::endl
              << "lda : " << lda << std::endl
              << "lda : " << lda << std::endl
              << "nb : " << nb << std::endl
              << "liwork : " << liwork << std::endl
              << "lwork : " << lwork << std::endl;
            RTE_WARNING(s);
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    /// wrapper for solving a generalized eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_lapack|dsygvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_lapack|zhegvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__,
          dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_lapack|ssygvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__,
          float* eval__, dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_lapack|chegvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    template <typename T>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, real_type<T>* eval__,
           dmatrix<T>& Z__)
    {
        ftn_int info;

        ftn_int lda = A__.ld();
        ftn_int ldb = B__.ld();
        ftn_int ldz = Z__.ld();

        real_type<T> abs_tol = 2 * linalg_base::dlamch('S');
        real_type<T> vl{0};
        real_type<T> vu{0};

        ftn_int ione{1};
        ftn_int m{0};

        auto& mph = get_memory_pool(memory_t::host);

        auto w     = mph.get_unique_ptr<real_type<T>>(matrix_size__);
        auto ifail = mph.get_unique_ptr<ftn_int>(matrix_size__);

        int nb, lwork, liwork;
        int lrwork = 0; // only required in complex
        if (std::is_same<T, double>::value) {
            nb     = linalg_base::ilaenv(1, "DSYTRD", "U", matrix_size__, 0, 0, 0);
            lwork  = (nb + 3) * matrix_size__ + 1024;
            liwork = 5 * matrix_size__;
        } else if (std::is_same<T, std::complex<double>>::value) {
            nb     = linalg_base::ilaenv(1, "ZHETRD", "U", matrix_size__, 0, 0, 0);
            lwork  = (nb + 1) * matrix_size__;
            lrwork = 7 * matrix_size__;
            liwork = 5 * matrix_size__;
        } else if (std::is_same<T, float>::value) {
            nb     = linalg_base::ilaenv(1, "SSYTRD", "U", matrix_size__, 0, 0, 0);
            lwork  = (nb + 3) * matrix_size__ + 1024;
            liwork = 5 * matrix_size__;
        } else if (std::is_same<T, std::complex<float>>::value) {
            nb     = linalg_base::ilaenv(1, "CHETRD", "U", matrix_size__, 0, 0, 0);
            lwork  = (nb + 1) * matrix_size__;
            lrwork = 7 * matrix_size__;
            liwork = 5 * matrix_size__;
        }

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);
        auto rwork = mph.get_unique_ptr<real_type<T>>(lrwork); // only required in complex

        if (std::is_same<T, double>::value) {
            FORTRAN(dsygvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<double*>(B__.at(memory_t::host)), &ldb, reinterpret_cast<double*>(&vl),
             reinterpret_cast<double*>(&vu), &ione, &nev__, reinterpret_cast<double*>(&abs_tol), &m,
             reinterpret_cast<double*>(w.get()), reinterpret_cast<double*>(Z__.at(memory_t::host)), &ldz,
             reinterpret_cast<double*>(work.get()), &lwork, iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1,
             (ftn_int)1);
        } else if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(zhegvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)),
             &lda, reinterpret_cast<std::complex<double>*>(B__.at(memory_t::host)), &ldb,
             reinterpret_cast<double*>(&vl), reinterpret_cast<double*>(&vu), &ione, &nev__,
             reinterpret_cast<double*>(&abs_tol), &m, reinterpret_cast<double*>(w.get()),
             reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), &ldz,
             reinterpret_cast<std::complex<double>*>(work.get()), &lwork, reinterpret_cast<double*>(rwork.get()),
             iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(ssygvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<float*>(B__.at(memory_t::host)), &ldb, reinterpret_cast<float*>(&vl),
             reinterpret_cast<float*>(&vu), &ione, &nev__, reinterpret_cast<float*>(&abs_tol), &m,
             reinterpret_cast<float*>(w.get()), reinterpret_cast<float*>(Z__.at(memory_t::host)), &ldz,
             reinterpret_cast<float*>(work.get()), &lwork, iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1,
             (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(chegvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &lda,
             reinterpret_cast<std::complex<float>*>(B__.at(memory_t::host)), &ldb, reinterpret_cast<float*>(&vl),
             reinterpret_cast<float*>(&vu), &ione, &nev__, reinterpret_cast<float*>(&abs_tol), &m,
             reinterpret_cast<float*>(w.get()), reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), &ldz,
             reinterpret_cast<std::complex<float>*>(work.get()), &lwork, reinterpret_cast<float*>(rwork.get()),
             iwork.get(), ifail.get(), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        }

        if (m != nev__) {
            std::stringstream s;
            s << "not all eigen-values are found" << std::endl
              << "target number of eigen-values: " << nev__ << std::endl
              << "number of eigen-values found: " << m;
            RTE_WARNING(s);
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }
};

#ifdef SIRIUS_ELPA
class Eigensolver_elpa : public Eigensolver
{
  private:
    int stage_;

    // template <typename T>
    // void to_std(ftn_int matrix_size__, dmatrix<T>& A__, dmatrix<T>& B__, dmatrix<T>& Z__) const
    //{
    //     PROFILE("Eigensolver_elpa|to_std");

    //    if (A__.num_cols_local() != Z__.num_cols_local()) {
    //        std::stringstream s;
    //        s << "number of columns in A and Z doesn't match" << std::endl
    //          << "  number of cols in A (local and global): " << A__.num_cols_local() << " " << A__.num_cols()
    //          << std::endl
    //          << "  number of cols in B (local and global): " << B__.num_cols_local() << " " << B__.num_cols()
    //          << std::endl
    //          << "  number of cols in Z (local and global): " << Z__.num_cols_local() << " " << Z__.num_cols()
    //          << std::endl
    //          << "  number of rows in A (local and global): " << A__.num_rows_local() << " " << A__.num_rows()
    //          << std::endl
    //          << "  number of rows in B (local and global): " << B__.num_rows_local() << " " << B__.num_rows()
    //          << std::endl
    //          << "  number of rows in Z (local and global): " << Z__.num_rows_local() << " " << Z__.num_rows()
    //          << std::endl;
    //        RTE_THROW(s);
    //    }
    //    if (A__.bs_row() != A__.bs_col()) {
    //        RTE_THROW("wrong block size");
    //    }

    //    /* Cholesky factorization B = U^{H}*U */
    //    linalg(linalg_t::scalapack).potrf(matrix_size__, B__.at(memory_t::host), B__.ld(), B__.descriptor());
    //    /* inversion of the triangular matrix */
    //    linalg(linalg_t::scalapack).trtri(matrix_size__, B__.at(memory_t::host), B__.ld(), B__.descriptor());
    //    /* U^{-1} is upper triangular matrix */
    //    for (int i = 0; i < matrix_size__; i++) {
    //        for (int j = i + 1; j < matrix_size__; j++) {
    //            B__.set(j, i, 0);
    //        }
    //    }
    //    /* transform to standard eigen-problem */
    //    /* A * U{-1} -> Z */
    //    linalg(linalg_t::scalapack).gemm('N', 'N', matrix_size__, matrix_size__, matrix_size__,
    //        &linalg_const<T>::one(), A__, 0, 0, B__, 0, 0, &linalg_const<T>::zero(), Z__, 0, 0);
    //    /* U^{-H} * Z = U{-H} * A * U^{-1} -> A */
    //    linalg(linalg_t::scalapack).gemm('C', 'N', matrix_size__, matrix_size__, matrix_size__,
    //        &linalg_const<T>::one(), B__, 0, 0, Z__, 0, 0,  &linalg_const<T>::zero(), A__, 0, 0);
    //}

    // template <typename T>
    // void bt(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, dmatrix<T>& Z__) const
    //{
    //     PROFILE("Eigensolver_elpa|bt");
    //     /* back-transform of eigen-vectors */
    //     linalg(linalg_t::scalapack).gemm('N', 'N', matrix_size__, nev__, matrix_size__, &linalg_const<T>::one(),
    //               B__, 0, 0, Z__, 0, 0, &linalg_const<T>::zero(), A__, 0, 0);
    //     A__ >> Z__;

    //}
  public:
    Eigensolver_elpa(int stage__);
    static void
    initialize();

    static void
    finalize();
    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override;

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__) override;

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override;

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override;

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override;

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override;

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override;
    /// Solve a standard eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override;
};
#else
class Eigensolver_elpa : public Eigensolver
{
  public:
    Eigensolver_elpa(int stage__)
        : Eigensolver(ev_solver_t::elpa, true, memory_t::host, memory_t::host)
    {
    }
};
#endif

#ifdef SIRIUS_SCALAPACK
class Eigensolver_scalapack : public Eigensolver
{
  private:
    double const ortfac_{1e-6};
    double const abstol_{1e-12};

  public:
    Eigensolver_scalapack()
        : Eigensolver(ev_solver_t::scalapack, true, memory_t::host, memory_t::host)
    {
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    template <typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    int
    solve_(ftn_int matrix_size__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
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
        T work1;
        real_type<T> rwork1;
        ftn_int iwork1;

        /* work size query */
        if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(pzheevd)
            ("V", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<double*>(eval__), reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)),
             &ione, &ione, descz, reinterpret_cast<std::complex<double>*>(&work1), &lwork,
             reinterpret_cast<double*>(&rwork1), &lrwork, &iwork1, &liwork, &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(pcheevd)
            ("V", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<float*>(eval__), reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)),
             &ione, &ione, descz, reinterpret_cast<std::complex<float>*>(&work1), &lwork,
             reinterpret_cast<float*>(&rwork1), &lrwork, &iwork1, &liwork, &info, (ftn_int)1, (ftn_int)1);
        }

        lwork  = static_cast<ftn_int>(work1.real()) + 1;
        lrwork = static_cast<ftn_int>(rwork1) + 1;
        liwork = iwork1;

        auto& mph  = get_memory_pool(memory_t::host);
        auto work  = mph.get_unique_ptr<T>(lwork);
        auto rwork = mph.get_unique_ptr<real_type<T>>(lrwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);

        if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(pzheevd)
            ("V", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<double*>(eval__), reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)),
             &ione, &ione, descz, reinterpret_cast<std::complex<double>*>(work.get()), &lwork,
             reinterpret_cast<double*>(rwork.get()), &lrwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(pcheevd)
            ("V", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<float*>(eval__), reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)),
             &ione, &ione, descz, reinterpret_cast<std::complex<float>*>(work.get()), &lwork,
             reinterpret_cast<float*>(rwork.get()), &lrwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        }
        return info;
    }

    /// wrapper for solving a standard eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pzheevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pcheevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    int
    solve_(ftn_int matrix_size__, dmatrix<T>& A__, T* eval__, dmatrix<T>& Z__)
    {
        ftn_int info;
        ftn_int ione{1};

        ftn_int lwork{-1};
        ftn_int liwork{-1};
        T work1[10];
        ftn_int iwork1[10];

        /* work size query */
        if (std::is_same<T, double>::value) {
            FORTRAN(pdsyevd)
            ("V", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &ione, &ione,
             const_cast<ftn_int*>(A__.descriptor()), reinterpret_cast<double*>(eval__),
             reinterpret_cast<double*>(Z__.at(memory_t::host)), &ione, &ione, const_cast<ftn_int*>(Z__.descriptor()),
             reinterpret_cast<double*>(work1), &lwork, iwork1, &liwork, &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(pssyevd)
            ("V", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &ione, &ione,
             const_cast<ftn_int*>(A__.descriptor()), reinterpret_cast<float*>(eval__),
             reinterpret_cast<float*>(Z__.at(memory_t::host)), &ione, &ione, const_cast<ftn_int*>(Z__.descriptor()),
             reinterpret_cast<float*>(work1), &lwork, iwork1, &liwork, &info, (ftn_int)1, (ftn_int)1);
        }

        lwork  = static_cast<ftn_int>(work1[0]) + 1;
        liwork = iwork1[0];

        auto& mph  = get_memory_pool(memory_t::host);
        auto work  = mph.get_unique_ptr<T>(lwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);

        if (std::is_same<T, double>::value) {
            FORTRAN(pdsyevd)
            ("V", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &ione, &ione,
             const_cast<ftn_int*>(A__.descriptor()), reinterpret_cast<double*>(eval__),
             reinterpret_cast<double*>(Z__.at(memory_t::host)), &ione, &ione, const_cast<ftn_int*>(Z__.descriptor()),
             reinterpret_cast<double*>(work.get()), &lwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(pssyevd)
            ("V", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &ione, &ione,
             const_cast<ftn_int*>(A__.descriptor()), reinterpret_cast<float*>(eval__),
             reinterpret_cast<float*>(Z__.at(memory_t::host)), &ione, &ione, const_cast<ftn_int*>(Z__.descriptor()),
             reinterpret_cast<float*>(work.get()), &lwork, iwork.get(), &liwork, &info, (ftn_int)1, (ftn_int)1);
        }
        return info;
    }

    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pdsyevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pssyevd");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, T* eval__, dmatrix<T>& Z__)
    {
        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        T d1;
        ftn_int info{-1};

        auto& mph = get_memory_pool(memory_t::host);

        auto ifail   = mph.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mph.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mph.get_unique_ptr<T>(A__.blacs_grid().comm().size());
        auto w       = mph.get_unique_ptr<T>(matrix_size__);

        /* work size query */
        T work3[3];
        ftn_int iwork1;
        ftn_int lwork{-1};
        ftn_int liwork{-1};

        if (std::is_same<T, double>::value) {
            FORTRAN(pdsyevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &ione, &ione, desca,
             reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<double*>(w.get()), &ortfac_, reinterpret_cast<double*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<double*>(work3), &lwork, &iwork1, &liwork, ifail.get(), iclustr.get(),
             reinterpret_cast<double*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(pssyevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &ione, &ione, desca,
             reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<float*>(w.get()), &ortfac_, reinterpret_cast<float*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<float*>(work3), &lwork, &iwork1, &liwork, ifail.get(), iclustr.get(),
             reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        }

        lwork  = static_cast<ftn_int>(work3[0]) + 4 * (1 << 20);
        liwork = iwork1;

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);

        if (std::is_same<T, double>::value) {
            FORTRAN(pdsyevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &ione, &ione, desca,
             reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<double*>(w.get()), &ortfac_, reinterpret_cast<double*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<double*>(work.get()), &lwork, iwork.get(), &liwork, ifail.get(),
             iclustr.get(), reinterpret_cast<double*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(pssyevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &ione, &ione, desca,
             reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<float*>(w.get()), &ortfac_, reinterpret_cast<float*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<float*>(work.get()), &lwork, iwork.get(), &liwork, ifail.get(),
             iclustr.get(), reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        }

        if ((m != nev__) || (nz != nev__)) {
            RTE_WARNING("Not all eigen-vectors or eigen-values are found.");
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
                RTE_WARNING(s);
            }

            std::stringstream s;
            if (std::is_same<T, double>::value) {
                s << "pdsyevx returned " << info;
            } else if (std::is_same<T, float>::value) {
                s << "pssyevx returned " << info;
            }
            RTE_WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pdsyevx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pssyevx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    template <typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        real_type<T> d1;
        ftn_int info{-1};

        auto& mph = get_memory_pool(memory_t::host);

        auto ifail   = mph.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mph.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mph.get_unique_ptr<real_type<T>>(A__.blacs_grid().comm().size());
        auto w       = mph.get_unique_ptr<real_type<T>>(matrix_size__);

        /* work size query */
        T work3[3];
        real_type<T> rwork3[3];
        ftn_int iwork1;
        ftn_int lwork  = -1;
        ftn_int lrwork = -1;
        ftn_int liwork = -1;

        if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(pzheevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), &ione,
             &ione, desca, reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m,
             &nz, reinterpret_cast<double*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<double>*>(work3), &lwork, reinterpret_cast<double*>(rwork3), &lrwork,
             &iwork1, &liwork, ifail.get(), iclustr.get(), reinterpret_cast<double*>(gap.get()), &info, (ftn_int)1,
             (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(pcheevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &ione,
             &ione, desca, reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m,
             &nz, reinterpret_cast<float*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<float>*>(work3), &lwork, reinterpret_cast<float*>(rwork3), &lrwork, &iwork1,
             &liwork, ifail.get(), iclustr.get(), reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1, (ftn_int)1,
             (ftn_int)1);
        }

        lwork  = static_cast<int32_t>(work3[0].real()) + (1 << 16);
        lrwork = static_cast<int32_t>(rwork3[0]) + (1 << 16);
        liwork = iwork1;

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto rwork = mph.get_unique_ptr<real_type<T>>(lrwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);

        if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(pzheevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), &ione,
             &ione, desca, reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m,
             &nz, reinterpret_cast<double*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<double>*>(work.get()), &lwork, reinterpret_cast<double*>(rwork.get()),
             &lrwork, iwork.get(), &liwork, ifail.get(), iclustr.get(), reinterpret_cast<double*>(gap.get()), &info,
             (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(pcheevx)
            ("V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), &ione,
             &ione, desca, reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m,
             &nz, reinterpret_cast<float*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<float>*>(work.get()), &lwork, reinterpret_cast<float*>(rwork.get()), &lrwork,
             iwork.get(), &liwork, ifail.get(), iclustr.get(), reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1,
             (ftn_int)1, (ftn_int)1);
        }

        if ((m != nev__) || (nz != nev__)) {
            RTE_WARNING("Not all eigen-vectors or eigen-values are found.");
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
                RTE_WARNING(s);
            }

            std::stringstream s;
            if (std::is_same<T, std::complex<double>>::value) {
                s << "pzheevx returned " << info;
            } else if (std::is_same<T, std::complex<float>>::value) {
                s << "pcheevx returned " << info;
            }
            RTE_WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pzheevx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pcheevx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, T* eval__, dmatrix<T>& Z__)
    {
        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descb[9];
        linalg_base::descinit(descb, matrix_size__, matrix_size__, B__.bs_row(), B__.bs_col(), 0, 0,
                              B__.blacs_grid().context(), B__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        auto& mph    = get_memory_pool(memory_t::host);
        auto ifail   = mph.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mph.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mph.get_unique_ptr<T>(A__.blacs_grid().comm().size());
        auto w       = mph.get_unique_ptr<T>(matrix_size__);

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        T d1;
        ftn_int info{-1};

        T work1[3];
        ftn_int lwork  = -1;
        ftn_int liwork = -1;
        /* work size query */
        if (std::is_same<T, double>::value) {
            FORTRAN(pdsygvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<double*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<double*>(w.get()), &ortfac_, reinterpret_cast<double*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<double*>(work1), &lwork, &liwork, &lwork, ifail.get(), iclustr.get(),
             reinterpret_cast<double*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(pssygvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<float*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<float*>(w.get()), &ortfac_, reinterpret_cast<float*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<float*>(work1), &lwork, &liwork, &lwork, ifail.get(), iclustr.get(),
             reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        }

        lwork = static_cast<int32_t>(work1[0]) + 4 * (1 << 20);

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);

        if (std::is_same<T, double>::value) {
            FORTRAN(pdsygvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<double*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<double*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<double*>(w.get()), &ortfac_, reinterpret_cast<double*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<double*>(work.get()), &lwork, iwork.get(), &liwork, ifail.get(),
             iclustr.get(), reinterpret_cast<double*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, float>::value) {
            FORTRAN(pssygvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<float*>(A__.at(memory_t::host)), &ione, &ione,
             desca, reinterpret_cast<float*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<float*>(w.get()), &ortfac_, reinterpret_cast<float*>(Z__.at(memory_t::host)), &ione,
             &ione, descz, reinterpret_cast<float*>(work.get()), &lwork, iwork.get(), &liwork, ifail.get(),
             iclustr.get(), reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1, (ftn_int)1, (ftn_int)1);
        }

        if ((m != nev__) || (nz != nev__)) {
            RTE_WARNING("Not all eigen-vectors or eigen-values are found.");
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
                RTE_WARNING(s);
            }

            std::stringstream s;
            if (std::is_same<T, double>::value) {
                s << "pdsygvx returned " << info;
            } else if (std::is_same<T, float>::value) {
                s << "pssygvx returned " << info;
            }
            RTE_WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pdsygvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__,
          dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pssygvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    template <typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, real_type<T>* eval__,
           dmatrix<T>& Z__)
    {
        ftn_int desca[9];
        linalg_base::descinit(desca, matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0,
                              A__.blacs_grid().context(), A__.ld());

        ftn_int descb[9];
        linalg_base::descinit(descb, matrix_size__, matrix_size__, B__.bs_row(), B__.bs_col(), 0, 0,
                              B__.blacs_grid().context(), B__.ld());

        ftn_int descz[9];
        linalg_base::descinit(descz, matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0,
                              Z__.blacs_grid().context(), Z__.ld());

        auto& mph = get_memory_pool(memory_t::host);

        auto ifail   = mph.get_unique_ptr<ftn_int>(matrix_size__);
        auto iclustr = mph.get_unique_ptr<ftn_int>(2 * A__.blacs_grid().comm().size());
        auto gap     = mph.get_unique_ptr<real_type<T>>(A__.blacs_grid().comm().size());
        auto w       = mph.get_unique_ptr<real_type<T>>(matrix_size__);

        ftn_int ione{1};

        ftn_int m{-1};
        ftn_int nz{-1};
        real_type<T> d1;
        ftn_int info{-1};

        ftn_int lwork{-1};
        ftn_int lrwork{-1};
        ftn_int liwork{-1};

        T work1;
        real_type<T> rwork3[3];
        ftn_int iwork1;

        /* work size query */
        if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(pzhegvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)),
             &ione, &ione, desca, reinterpret_cast<std::complex<double>*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<double*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<double>*>(&work1), &lwork, reinterpret_cast<double*>(rwork3), &lrwork,
             &iwork1, &liwork, ifail.get(), iclustr.get(), reinterpret_cast<double*>(gap.get()), &info, (ftn_int)1,
             (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(pchegvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)),
             &ione, &ione, desca, reinterpret_cast<std::complex<float>*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<float*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<float>*>(&work1), &lwork, reinterpret_cast<float*>(rwork3), &lrwork, &iwork1,
             &liwork, ifail.get(), iclustr.get(), reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1, (ftn_int)1,
             (ftn_int)1);
        }

        lwork  = 2 * static_cast<int32_t>(work1.real()) + 4096;
        lrwork = 2 * static_cast<int32_t>(rwork3[0]) + 4096;
        liwork = 2 * iwork1 + 4096;

        auto work  = mph.get_unique_ptr<T>(lwork);
        auto rwork = mph.get_unique_ptr<real_type<T>>(lrwork);
        auto iwork = mph.get_unique_ptr<ftn_int>(liwork);

        if (std::is_same<T, std::complex<double>>::value) {
            FORTRAN(pzhegvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)),
             &ione, &ione, desca, reinterpret_cast<std::complex<double>*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<double*>(&d1), reinterpret_cast<double*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<double*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<double>*>(work.get()), &lwork, reinterpret_cast<double*>(rwork.get()),
             &lrwork, iwork.get(), &liwork, ifail.get(), iclustr.get(), reinterpret_cast<double*>(gap.get()), &info,
             (ftn_int)1, (ftn_int)1, (ftn_int)1);
        } else if (std::is_same<T, std::complex<float>>::value) {
            FORTRAN(pchegvx)
            (&ione, "V", "I", "U", &matrix_size__, reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)),
             &ione, &ione, desca, reinterpret_cast<std::complex<float>*>(B__.at(memory_t::host)), &ione, &ione, descb,
             reinterpret_cast<float*>(&d1), reinterpret_cast<float*>(&d1), &ione, &nev__, &abstol_, &m, &nz,
             reinterpret_cast<float*>(w.get()), &ortfac_,
             reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), &ione, &ione, descz,
             reinterpret_cast<std::complex<float>*>(work.get()), &lwork, reinterpret_cast<float*>(rwork.get()), &lrwork,
             iwork.get(), &liwork, ifail.get(), iclustr.get(), reinterpret_cast<float*>(gap.get()), &info, (ftn_int)1,
             (ftn_int)1, (ftn_int)1);
        }

        if ((m != nev__) || (nz != nev__)) {
            RTE_WARNING("Not all eigen-vectors or eigen-values are found.");
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
                RTE_WARNING(s);
            }

            std::stringstream s;
            if (std::is_same<T, std::complex<double>>::value) {
                s << "pzhegvx returned " << info;
            } else if (std::is_same<T, std::complex<float>>::value) {
                s << "pchegvx returned " << info;
            }
            RTE_WARNING(s);
        } else {
            std::copy(w.get(), w.get() + nev__, eval__);
        }

        return info;
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pzhegvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__,
          float* eval__, dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_scalapack|pchegvx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }
};
#else
class Eigensolver_scalapack : public Eigensolver
{
  public:
    Eigensolver_scalapack()
        : Eigensolver(ev_solver_t::scalapack, true, memory_t::host, memory_t::host)
    {
    }
};
#endif

#ifdef SIRIUS_DLAF
class Eigensolver_dlaf : public Eigensolver
{
  public:
    Eigensolver_dlaf()
        : Eigensolver(ev_solver_t::dlaf, true, memory_t::host, memory_t::host)
    {
    }

    static void
    initialize();
    static void
    finalize();

    /// Solve a standard eigen-value problem for all eigen-pairs.
    // template <typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    template <typename T>
    int
    solve_(ftn_int matrix_size__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
        DLAF_descriptor desca{
                matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0, 0, 0, static_cast<int>(A__.ld())};
        DLAF_descriptor descz{
                matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0, 0, 0, static_cast<int>(Z__.ld())};

        if (std::is_same_v<T, std::complex<double>>) {
            return dlaf_hermitian_eigensolver_z(A__.blacs_grid().context(), 'L',
                                                reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)), desca,
                                                reinterpret_cast<double*>(eval__),
                                                reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)), descz);
        } else if (std::is_same_v<T, std::complex<float>>) {
            return dlaf_hermitian_eigensolver_c(A__.blacs_grid().context(), 'L',
                                                reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)), desca,
                                                reinterpret_cast<float*>(eval__),
                                                reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)), descz);
        } else if (std::is_same_v<T, double>) {
            return dlaf_symmetric_eigensolver_d(
                    A__.blacs_grid().context(), 'L', reinterpret_cast<double*>(A__.at(memory_t::host)), desca,
                    reinterpret_cast<double*>(eval__), reinterpret_cast<double*>(Z__.at(memory_t::host)), descz);
        } else if (std::is_same_v<T, float>) {
            return dlaf_symmetric_eigensolver_s(
                    A__.blacs_grid().context(), 'L', reinterpret_cast<float*>(A__.at(memory_t::host)), desca,
                    reinterpret_cast<float*>(eval__), reinterpret_cast<float*>(Z__.at(memory_t::host)), descz);
        }
    }

    /// wrapper for solving a standard eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_eigensolver_z");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_eigensolver_c");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_eigensolver_d");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_eigensolver_s");
        return solve_(matrix_size__, A__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for N lowest eigen-pairs.
    template <typename T>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
        auto& mph = get_memory_pool(memory_t::host);
        auto w    = mph.get_unique_ptr<real_type<T>>(matrix_size__);

        auto info = solve_(matrix_size__, A__, w.get(), Z__);

        std::copy(w.get(), w.get() + nev__, eval__);

        return info;
    }

    /// wrapper for solving a standard eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_eigensolver_z");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_eigensolver_c");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_eigensolver_d");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_eigensolver_s");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    /// Solve a generalized eigen-value problem for all eigen-pairs.
    template <typename T>
    int
    solve_(ftn_int matrix_size__, dmatrix<T>& A__, dmatrix<T>& B__, real_type<T>* eval__, dmatrix<T>& Z__)
    {
        DLAF_descriptor desca{
                matrix_size__, matrix_size__, A__.bs_row(), A__.bs_col(), 0, 0, 0, 0, static_cast<int>(A__.ld())};
        DLAF_descriptor descb{
                matrix_size__, matrix_size__, B__.bs_row(), B__.bs_col(), 0, 0, 0, 0, static_cast<int>(B__.ld())};
        DLAF_descriptor descz{
                matrix_size__, matrix_size__, Z__.bs_row(), Z__.bs_col(), 0, 0, 0, 0, static_cast<int>(Z__.ld())};

        if (std::is_same_v<T, std::complex<double>>) {
            return dlaf_hermitian_generalized_eigensolver_z(
                    A__.blacs_grid().context(), 'L', reinterpret_cast<std::complex<double>*>(A__.at(memory_t::host)),
                    desca, reinterpret_cast<std::complex<double>*>(B__.at(memory_t::host)), descb,
                    reinterpret_cast<double*>(eval__), reinterpret_cast<std::complex<double>*>(Z__.at(memory_t::host)),
                    descz);
        } else if (std::is_same_v<T, std::complex<float>>) {
            return dlaf_hermitian_generalized_eigensolver_c(
                    A__.blacs_grid().context(), 'L', reinterpret_cast<std::complex<float>*>(A__.at(memory_t::host)),
                    desca, reinterpret_cast<std::complex<float>*>(B__.at(memory_t::host)), descb,
                    reinterpret_cast<float*>(eval__), reinterpret_cast<std::complex<float>*>(Z__.at(memory_t::host)),
                    descz);
        } else if (std::is_same_v<T, double>) {
            return dlaf_symmetric_generalized_eigensolver_d(
                    A__.blacs_grid().context(), 'L', reinterpret_cast<double*>(A__.at(memory_t::host)), desca,
                    reinterpret_cast<double*>(B__.at(memory_t::host)), descb, reinterpret_cast<double*>(eval__),
                    reinterpret_cast<double*>(Z__.at(memory_t::host)), descz);
        } else if (std::is_same_v<T, float>) {
            return dlaf_symmetric_generalized_eigensolver_s(
                    A__.blacs_grid().context(), 'L', reinterpret_cast<float*>(A__.at(memory_t::host)), desca,
                    reinterpret_cast<float*>(A__.at(memory_t::host)), descb, reinterpret_cast<float*>(eval__),
                    reinterpret_cast<float*>(Z__.at(memory_t::host)), descz);
        }
    }

    /// wrapper for solving a generalized eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_generalized_eigensolver_z");
        return solve_(matrix_size__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_generalized_eigensolver_c");
        return solve_(matrix_size__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_generalized_eigensolver_d");
        return solve_(matrix_size__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_generalized_eigensolver_s");
        return solve_(matrix_size__, A__, B__, eval__, Z__);
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    template <typename T>
    int
    solve_(ftn_int matrix_size__, ftn_int nev__, dmatrix<T>& A__, dmatrix<T>& B__, real_type<T>* eval__,
           dmatrix<T>& Z__)
    {
        auto& mph = get_memory_pool(memory_t::host);
        auto w    = mph.get_unique_ptr<real_type<T>>(matrix_size__);

        auto info = solve_(matrix_size__, A__, B__, w.get(), Z__);

        std::copy(w.get(), w.get() + nev__, eval__);

        return info;
    }

    /// wrapper for solving a generalized eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_generalized_eigensolver_z");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__,
          float* eval__, dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_hermitian_generalized_eigensolver_c");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_generalized_eigensolver_d");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__,
          dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_dlaf|dlaf_symmetric_generalized_eigensolver_s");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }
};
#else
class Eigensolver_dlaf : public Eigensolver
{
  public:
    Eigensolver_dlaf()
        : Eigensolver(ev_solver_t::dlaf, true, memory_t::host, memory_t::host)
    {
    }
};
#endif

#ifdef SIRIUS_MAGMA
class Eigensolver_magma : public Eigensolver
{
  public:
    Eigensolver_magma()
        : Eigensolver(ev_solver_t::magma, false, memory_t::host_pinned, memory_t::host)
    {
    }

    /// Solve a generalized eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_magma|dsygvdx");

        int nt  = omp_get_max_threads();
        int lda = A__.ld();
        int ldb = B__.ld();

        auto& mph  = get_memory_pool(memory_t::host);
        auto& mphp = get_memory_pool(memory_t::host_pinned);
        auto w     = mph.get_unique_ptr<double>(matrix_size__);

        int m;
        int info;

        int lwork;
        int liwork;
        magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

        auto h_work = mphp.get_unique_ptr<double>(lwork);
        auto iwork  = mph.get_unique_ptr<magma_int_t>(liwork);

        magma_dsygvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size__, A__.at(memory_t::host), lda,
                             B__.at(memory_t::host), ldb, 0.0, 0.0, 1, nev__, &m, w.get(), h_work.get(), lwork,
                             iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            RTE_THROW("magma has changed the number of threads");
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
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_magma|zhegvdx");

        int nt  = omp_get_max_threads();
        int lda = A__.ld();
        int ldb = B__.ld();

        auto& mph  = get_memory_pool(memory_t::host);
        auto& mphp = get_memory_pool(memory_t::host_pinned);
        auto w     = mph.get_unique_ptr<double>(matrix_size__);

        int m;
        int info;

        int lwork;
        int lrwork;
        int liwork;
        magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

        auto h_work = mphp.get_unique_ptr<std::complex<double>>(lwork);
        auto rwork  = mphp.get_unique_ptr<double>(lrwork);
        auto iwork  = mph.get_unique_ptr<magma_int_t>(liwork);

        magma_zhegvdx_2stage(1, MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
                             reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::host)), lda,
                             reinterpret_cast<magmaDoubleComplex*>(B__.at(memory_t::host)), ldb, 0.0, 0.0, 1, nev__, &m,
                             w.get(), reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork, rwork.get(), lrwork,
                             iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            RTE_THROW("magma has changed the number of threads");
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
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_magma|dsygvdx");

        // Bug in magma for small matrix sizes -> call lapack instead as workaround
        if (matrix_size__ <= 128) {
            return Eigensolver_lapack().solve(matrix_size__, nev__, A__, eval__, Z__);
        }

        auto& mph  = get_memory_pool(memory_t::host);
        auto& mphp = get_memory_pool(memory_t::host_pinned);

        int nt  = omp_get_max_threads();
        int lda = A__.ld();
        auto w  = mph.get_unique_ptr<double>(matrix_size__);

        int lwork;
        int liwork;
        magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

        auto h_work = mphp.get_unique_ptr<double>(lwork);
        auto iwork  = mph.get_unique_ptr<magma_int_t>(liwork);

        int info;
        int m;

        magma_dsyevdx(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__, A__.at(memory_t::host), lda, 0.0, 0.0, 1, nev__,
                      &m, w.get(), h_work.get(), lwork, iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            RTE_THROW("magma has changed the number of threads");
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
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_magma|zheevdx");

        int nt     = omp_get_max_threads();
        int lda    = A__.ld();
        auto& mph  = get_memory_pool(memory_t::host);
        auto& mphp = get_memory_pool(memory_t::host_pinned);
        auto w     = mph.get_unique_ptr<double>(matrix_size__);

        int info, m;

        int lwork;
        int lrwork;
        int liwork;
        magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

        auto h_work = mphp.get_unique_ptr<std::complex<double>>(lwork);
        auto rwork  = mphp.get_unique_ptr<double>(lrwork);
        auto iwork  = mph.get_unique_ptr<magma_int_t>(liwork);

        magma_zheevdx_2stage(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
                             reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::host)), lda, 0.0, 0.0, 1, nev__, &m,
                             w.get(), reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork, rwork.get(), lrwork,
                             iwork.get(), liwork, &info);

        if (nt != omp_get_max_threads()) {
            RTE_THROW("magma has changed the number of threads");
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

class Eigensolver_magma_gpu : public Eigensolver
{
  public:
    Eigensolver_magma_gpu()
        : Eigensolver(ev_solver_t::magma, false, memory_t::host_pinned, memory_t::device)
    {
    }

    /// Solve a hermitian eigen-value problem for N lowest eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_magma_gpu|zheevdx");

        int nt     = omp_get_max_threads();
        int lda    = A__.ld();
        auto& mph  = get_memory_pool(memory_t::host);
        auto& mphp = get_memory_pool(memory_t::host_pinned);
        auto w     = mph.get_unique_ptr<double>(matrix_size__);

        acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);

        int info, m;

        int lwork;
        int lrwork;
        int liwork;
        magma_zheevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &lrwork, &liwork);

        int llda    = matrix_size__ + 32;
        auto z_work = mphp.get_unique_ptr<std::complex<double>>(llda * matrix_size__);

        auto h_work = mphp.get_unique_ptr<std::complex<double>>(lwork);
        auto rwork  = mphp.get_unique_ptr<double>(lrwork);
        auto iwork  = mph.get_unique_ptr<magma_int_t>(liwork);

        magma_zheevdx_gpu(MagmaVec, MagmaRangeI, MagmaLower, matrix_size__,
                          reinterpret_cast<magmaDoubleComplex*>(A__.at(memory_t::device)), lda, 0.0, 0.0, 1, nev__, &m,
                          w.get(), reinterpret_cast<magmaDoubleComplex*>(z_work.get()), llda,
                          reinterpret_cast<magmaDoubleComplex*>(h_work.get()), lwork, rwork.get(), lrwork, iwork.get(),
                          liwork, &info);

        if (nt != omp_get_max_threads()) {
            RTE_THROW("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }

        return info;
    }

    /// Solve a symmetric eigen-value problem for N lower eigen-pairs.
    int
    solve(ftn_int matrix_size__, ftn_int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_magma_gpu|dsyevdx");

        int nt     = omp_get_max_threads();
        int lda    = A__.ld();
        auto& mph  = get_memory_pool(memory_t::host);
        auto& mphp = get_memory_pool(memory_t::host_pinned);
        auto w     = mph.get_unique_ptr<double>(matrix_size__);

        acc::copyin(A__.at(memory_t::device), A__.ld(), A__.at(memory_t::host), A__.ld(), matrix_size__, matrix_size__);

        int info, m;

        int lwork;
        int liwork;
        magma_dsyevdx_getworksize(matrix_size__, magma_get_parallel_numthreads(), 1, &lwork, &liwork);

        int llda    = matrix_size__ + 32;
        auto z_work = mphp.get_unique_ptr<double>(llda * matrix_size__);
        auto h_work = mphp.get_unique_ptr<double>(lwork);
        auto iwork  = mph.get_unique_ptr<magma_int_t>(liwork);

        magma_dsyevdx_gpu(MagmaVec /*jobz*/, MagmaRangeI /*range*/, MagmaLower /*uplo*/, matrix_size__ /*n*/,
                          A__.at(memory_t::device) /*dA*/, lda /*ldda*/, 0.0 /*vl*/, 0.0 /*vu*/, 1 /*il*/, nev__ /*iu*/,
                          &m /*mout*/, w.get() /*w*/, z_work.get() /*wA*/, llda /*ldwa*/, h_work.get() /*work*/,
                          lwork /*lwork*/, iwork.get() /*iwork*/, liwork /*liwork*/, &info /*info*/);

        if (nt != omp_get_max_threads()) {
            RTE_THROW("magma has changed the number of threads");
        }

        if (m < nev__) {
            return 1;
        }

        if (!info) {
            std::copy(w.get(), w.get() + nev__, eval__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }

        return info;
    }
};
#else
class Eigensolver_magma : public Eigensolver
{
  public:
    Eigensolver_magma()
        : Eigensolver(ev_solver_t::magma, false, memory_t::host_pinned, memory_t::host)
    {
    }
};

class Eigensolver_magma_gpu : public Eigensolver
{
  public:
    Eigensolver_magma_gpu()
        : Eigensolver(ev_solver_t::magma, false, memory_t::host_pinned, memory_t::device)
    {
    }
};
#endif

#if defined(SIRIUS_CUDA)
class Eigensolver_cuda : public Eigensolver
{
  public:
    Eigensolver_cuda()
        : Eigensolver(ev_solver_t::cusolver, false, memory_t::host_pinned, memory_t::device)
    {
    }

    template <typename T>
    int
    solve_(ftn_int matrix_size__, int nev__, dmatrix<T>& A__, real_type<T>* eval__, dmatrix<T>& Z__)
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
            CALL_CUSOLVER(cusolverDnSsyevdx_bufferSize,
                          (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                           reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__, &h_meig,
                           reinterpret_cast<float*>(w.get()), &lwork));
        } else if (std::is_same<T, std::complex<double>>::value) {
            CALL_CUSOLVER(cusolverDnZheevdx_bufferSize,
                          (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                           reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__,
                           &h_meig, reinterpret_cast<double*>(w.get()), &lwork));
        } else if (std::is_same<T, std::complex<float>>::value) {
            CALL_CUSOLVER(cusolverDnCheevdx_bufferSize,
                          (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                           reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1, nev__,
                           &h_meig, reinterpret_cast<float*>(w.get()), &lwork));
        }

        auto work = mpd.get_unique_ptr<T>(lwork);

        int info;
        auto dinfo = mpd.get_unique_ptr<int>(1);
        if (std::is_same<T, double>::value) {
            CALL_CUSOLVER(cusolverDnDsyevdx, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<double*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1,
                                              nev__, &h_meig, reinterpret_cast<double*>(w.get()),
                                              reinterpret_cast<double*>(work.get()), lwork, dinfo.get()));
        } else if (std::is_same<T, float>::value) {
            CALL_CUSOLVER(cusolverDnSsyevdx, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(), vl, vu, 1,
                                              nev__, &h_meig, reinterpret_cast<float*>(w.get()),
                                              reinterpret_cast<float*>(work.get()), lwork, dinfo.get()));
        } else if (std::is_same<T, std::complex<double>>::value) {
            CALL_CUSOLVER(cusolverDnZheevdx, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                              vl, vu, 1, nev__, &h_meig, reinterpret_cast<double*>(w.get()),
                                              reinterpret_cast<cuDoubleComplex*>(work.get()), lwork, dinfo.get()));
        } else if (std::is_same<T, std::complex<float>>::value) {
            CALL_CUSOLVER(cusolverDnCheevdx, (acc::cusolver::cusolver_handle(), jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(), vl,
                                              vu, 1, nev__, &h_meig, reinterpret_cast<float*>(w.get()),
                                              reinterpret_cast<cuFloatComplex*>(work.get()), lwork, dinfo.get()));
        }

        acc::copyout(&info, dinfo.get(), 1);
        if (!info) {
            acc::copyout(eval__, w.get(), nev__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }
        return info;
    }

    /// wrapper for dynamic binding
    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<float>& A__, float* eval__, dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_cuda|dsyevdx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<double>& A__, double* eval__, dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_cuda|ssyevdx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<float>>& A__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_cuda|cheevdx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<double>>& A__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_cuda|zheevdx");
        return solve_(matrix_size__, nev__, A__, eval__, Z__);
    }

    template <typename T>
    int
    solve_(ftn_int matrix_size__, int nev__, dmatrix<T>& A__, dmatrix<T>& B__, real_type<T>* eval__, dmatrix<T>& Z__)
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
                           reinterpret_cast<cuFloatComplex*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1, nev__,
                           &h_meig, reinterpret_cast<float*>(w.get()), &lwork));
        }

        auto work = mpd.get_unique_ptr<T>(lwork);

        int info;
        auto dinfo = mpd.get_unique_ptr<int>(1);
        if (std::is_same<T, double>::value) {
            CALL_CUSOLVER(cusolverDnDsygvdx, (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<double*>(A__.at(memory_t::device)), A__.ld(),
                                              reinterpret_cast<double*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1,
                                              nev__, &h_meig, reinterpret_cast<double*>(w.get()),
                                              reinterpret_cast<double*>(work.get()), lwork, dinfo.get()));
        } else if (std::is_same<T, float>::value) {
            CALL_CUSOLVER(cusolverDnSsygvdx, (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<float*>(A__.at(memory_t::device)), A__.ld(),
                                              reinterpret_cast<float*>(B__.at(memory_t::device)), B__.ld(), vl, vu, 1,
                                              nev__, &h_meig, reinterpret_cast<float*>(w.get()),
                                              reinterpret_cast<float*>(work.get()), lwork, dinfo.get()));
        } else if (std::is_same<T, std::complex<double>>::value) {
            CALL_CUSOLVER(cusolverDnZhegvdx, (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<cuDoubleComplex*>(A__.at(memory_t::device)), A__.ld(),
                                              reinterpret_cast<cuDoubleComplex*>(B__.at(memory_t::device)), B__.ld(),
                                              vl, vu, 1, nev__, &h_meig, reinterpret_cast<double*>(w.get()),
                                              reinterpret_cast<cuDoubleComplex*>(work.get()), lwork, dinfo.get()));
        } else if (std::is_same<T, std::complex<float>>::value) {
            CALL_CUSOLVER(cusolverDnChegvdx, (acc::cusolver::cusolver_handle(), itype, jobz, range, uplo, matrix_size__,
                                              reinterpret_cast<cuFloatComplex*>(A__.at(memory_t::device)), A__.ld(),
                                              reinterpret_cast<cuFloatComplex*>(B__.at(memory_t::device)), B__.ld(), vl,
                                              vu, 1, nev__, &h_meig, reinterpret_cast<float*>(w.get()),
                                              reinterpret_cast<cuFloatComplex*>(work.get()), lwork, dinfo.get()));
        }

        acc::copyout(&info, dinfo.get(), 1);
        if (!info) {
            acc::copyout(eval__, w.get(), nev__);
            acc::copyout(Z__.at(memory_t::host), Z__.ld(), A__.at(memory_t::device), A__.ld(), matrix_size__, nev__);
        }
        return info;
    }

    /// wrapper for dynamic binding
    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        PROFILE("Eigensolver_cuda|dsygvdx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__,
          dmatrix<float>& Z__) override
    {
        PROFILE("Eigensolver_cuda|ssygvdx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__,
          double* eval__, dmatrix<std::complex<double>>& Z__) override
    {
        PROFILE("Eigensolver_cuda|zhegvdx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, int nev__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__,
          float* eval__, dmatrix<std::complex<float>>& Z__) override
    {
        PROFILE("Eigensolver_cuda|chegvdx");
        return solve_(matrix_size__, nev__, A__, B__, eval__, Z__);
    }

    /// Solve a standard eigen-value problem for all eigen-pairs.
    int
    solve(ftn_int matrix_size__, dmatrix<double>& A__, dmatrix<double>& B__, double* eval__,
          dmatrix<double>& Z__) override
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<float>& A__, dmatrix<float>& B__, float* eval__, dmatrix<float>& Z__) override
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<double>>& A__, dmatrix<std::complex<double>>& B__, double* eval__,
          dmatrix<std::complex<double>>& Z__) override
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }

    int
    solve(ftn_int matrix_size__, dmatrix<std::complex<float>>& A__, dmatrix<std::complex<float>>& B__, float* eval__,
          dmatrix<std::complex<float>>& Z__) override
    {
        return solve(matrix_size__, matrix_size__, A__, B__, eval__, Z__);
    }
};
#else
class Eigensolver_cuda : public Eigensolver
{
  public:
    Eigensolver_cuda()
        : Eigensolver(ev_solver_t::cusolver, false, memory_t::host_pinned, memory_t::device)
    {
    }
};
#endif

} // namespace la

} // namespace sirius

#endif // __EIGENPROBLEM_HPP__

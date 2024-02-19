// Copyright (c) 2013-2020 Anton Kozhevnikov, Thomas Schulthess
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

/** \file linalg.hpp
 *
 *  \brief Linear algebra interface.
 */

#ifndef __LINALG_HPP__
#define __LINALG_HPP__

#include <stdint.h>
#include "core/memory.hpp"
#include "core/acc/acc.hpp"
#if defined(SIRIUS_GPU)
#include "core/acc/acc_blas.hpp"
#include "core/acc/acc_lapack.hpp"
#endif
#if defined(SIRIUS_MAGMA)
#include "core/acc/magma.hpp"
#endif
#if defined(SIRIUS_GPU) and defined(SIRIUS_CUDA)
#include "core/acc/cusolver.hpp"
#endif
#include "blas_lapack.h"
#include "dmatrix.hpp"
#include "linalg_spla.hpp"

namespace sirius {

namespace la {

namespace _local {
/// check if device id has been set properly
inline bool
is_set_device_id()
{
    return acc::get_device_id() == mpi::get_device_id(acc::num_devices());
}
} // namespace _local

#define linalg_msg_wrong_type "[" + std::string(__func__) + "] wrong type of linear algebra library: " + to_string(la_)

const std::string linalg_msg_no_scalapack = "not compiled with ScaLAPACK";

class wrap
{
  private:
    lib_t la_;

  public:
    wrap(lib_t la__)
        : la_(la__)
    {
    }

    /*
        BLAS Level 1
    */

    /// vector addition
    template <typename T>
    inline void
    axpy(int n, T const* alpha, T const* x, int incx, T* y, int incy);

    /*
        matrix - matrix multiplication
    */

    /// General matrix-matrix multiplication.
    /** Compute C = alpha * op(A) * op(B) + beta * op(C) with raw pointers. */
    template <typename T>
    inline void
    gemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, T const* alpha, T const* A, ftn_int lda, T const* B,
         ftn_int ldb, T const* beta, T* C, ftn_int ldc, acc::stream_id sid = acc::stream_id(-1)) const;

    /// Distributed general matrix-matrix multiplication.
    /** Compute C = alpha * op(A) * op(B) + beta * op(C) for distributed matrices. */
    template <typename T>
    inline void
    gemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, T const* alpha, dmatrix<T> const& A, ftn_int ia,
         ftn_int ja, dmatrix<T> const& B, ftn_int ib, ftn_int jb, T const* beta, dmatrix<T>& C, ftn_int ic, ftn_int jc);

    /// Hermitian matrix times a general matrix or vice versa.
    /** Perform one of the matrix-matrix operations \n
     *  C = alpha * A * B + beta * C (side = 'L') \n
     *  C = alpha * B * A + beta * C (side = 'R'), \n
     *  where A is a hermitian matrix with upper (uplo = 'U') of lower (uplo = 'L') triangular part defined.
     */
    template <typename T>
    inline void
    hemm(char side, char uplo, ftn_int m, ftn_int n, T const* alpha, T const* A, ftn_len lda, T const* B, ftn_len ldb,
         T const* beta, T* C, ftn_len ldc);

    template <typename T>
    inline void
    trmm(char side, char uplo, char transa, ftn_int m, ftn_int n, T const* aplha, T const* A, ftn_int lda, T* B,
         ftn_int ldb, acc::stream_id sid = acc::stream_id(-1)) const;

    /*
        rank2 update
    */

    template <typename T>
    inline void
    ger(ftn_int m, ftn_int n, T const* alpha, T const* x, ftn_int incx, T const* y, ftn_int incy, T* A, ftn_int lda,
        acc::stream_id sid = acc::stream_id(-1)) const;

    /*
        matrix factorization
    */

    /// Cholesky factorization
    template <typename T>
    inline int
    potrf(ftn_int n, T* A, ftn_int lda, ftn_int const* desca = nullptr) const;

    /// LU factorization of general matrix.
    template <typename T>
    inline int
    getrf(ftn_int m, ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// LU factorization of general matrix.
    template <typename T>
    inline int
    getrf(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, ftn_int* ipiv) const;

    template <typename T>
    inline int
    getrs(char trans, ftn_int n, ftn_int nrhs, T const* A, ftn_int lda, ftn_int* ipiv, T* B, ftn_int ldb) const;

    /// U*D*U^H factorization of hermitian or symmetric matrix.
    template <typename T>
    inline int
    sytrf(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// solve Ax=b in place of b where A is factorized with sytrf.
    template <typename T>
    inline int
    sytrs(ftn_int n, ftn_int nrhs, T* A, ftn_int lda, ftn_int* ipiv, T* b, ftn_int ldb) const;

    /*
        matrix inversion
    */

    /// Inversion of a triangular matrix.
    template <typename T>
    inline int
    trtri(ftn_int n, T* A, ftn_int lda, ftn_int const* desca = nullptr) const;

    template <typename T>
    inline int
    getri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// Inversion of factorized symmetric triangular matrix.
    template <typename T>
    inline int
    sytri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// Invert a general matrix.
    template <typename T>
    inline void
    geinv(ftn_int n, matrix<T>& A) const
    {
        std::vector<int> ipiv(n);
        int info = this->getrf(n, n, A.at(memory_t::host), A.ld(), &ipiv[0]);
        if (info) {
            std::printf("getrf returned %i\n", info);
            exit(-1);
        }

        info = this->getri(n, A.at(memory_t::host), A.ld(), &ipiv[0]);
        if (info) {
            std::printf("getri returned %i\n", info);
            exit(-1);
        }
    }

    template <typename T>
    inline void
    syinv(ftn_int n, matrix<T>& A) const
    {
        std::vector<int> ipiv(n);
        int info = this->sytrf(n, A.at(memory_t::host), A.ld(), &ipiv[0]);
        if (info) {
            std::printf("sytrf returned %i\n", info);
            exit(-1);
        }

        info = this->sytri(n, A.at(memory_t::host), A.ld(), &ipiv[0]);
        if (info) {
            std::printf("sytri returned %i\n", info);
            exit(-1);
        }
    }

    template <typename T>
    inline bool
    sysolve(ftn_int n, matrix<T>& A, mdarray<T, 1>& b) const
    {
        std::vector<int> ipiv(n);
        int info = this->sytrf(n, A.at(memory_t::host), A.ld(), ipiv.data());
        if (info)
            return false;

        info = this->sytrs(n, 1, A.at(memory_t::host), A.ld(), ipiv.data(), b.at(memory_t::host), b.ld());

        return !info;
    }

    /*
        solution of a linear system
    */

    /// Compute the solution to system of linear equations A * X = B for general tri-diagonal matrix.
    template <typename T>
    inline int
    gtsv(ftn_int n, ftn_int nrhs, T* dl, T* d, T* du, T* b, ftn_int ldb) const;

    /// Compute the solution to system of linear equations A * X = B for general matrix.
    template <typename T>
    inline int
    gesv(ftn_int n, ftn_int nrhs, T* A, ftn_int lda, T* B, ftn_int ldb) const;

    /*
        matrix transposition
    */

    /// Conjugate transpose matrix
    /** \param [in]  m   Number of rows of the target sub-matrix.
        \param [in]  n   Number of columns of the target sub-matrix.
        \param [in]  A   Input matrix
        \param [in]  ia  Starting row index of sub-matrix inside A
        \param [in]  ja  Starting column index of sub-matrix inside A
        \param [out] C   Output matrix
        \param [in]  ic  Starting row index of sub-matrix inside C
        \param [in]  jc  Starting column index of sub-matrix inside C
     */
    template <typename T>
    inline void
    tranc(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, dmatrix<T>& C, ftn_int ic, ftn_int jc) const;

    /// Transpose matrix without conjugation.
    template <typename T>
    inline void
    tranu(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, dmatrix<T>& C, ftn_int ic, ftn_int jc) const;

    // Constructing a Given's rotation
    template <typename T>
    inline std::tuple<ftn_double, ftn_double, ftn_double>
    lartg(T f, T g) const;

    template <typename T>
    inline void
    geqrf(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja);
};

template <>
inline void
wrap::geqrf<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ftn_int lwork = -1;
            ftn_double_complex z;
            ftn_int info;
            FORTRAN(pzgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork, &info);
            lwork = static_cast<int>(z.real() + 1);
            std::vector<ftn_double_complex> work(lwork);
            std::vector<ftn_double_complex> tau(std::max(m, n));
            FORTRAN(pzgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(), work.data(), &lwork,
             &info);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::lapack: {
            if (A.comm().size() != 1) {
                RTE_THROW("can't use lapack for distributed matrix; use scalapck instead");
            }
            ftn_int lwork = -1;
            ftn_double_complex z;
            ftn_int info;
            ftn_int lda = A.ld();
            FORTRAN(zgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, &z, &z, &lwork, &info);
            lwork = static_cast<int>(z.real() + 1);
            std::vector<ftn_double_complex> work(lwork);
            std::vector<ftn_double_complex> tau(std::max(m, n));
            FORTRAN(zgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, tau.data(), work.data(), &lwork, &info);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::geqrf<ftn_double>(ftn_int m, ftn_int n, dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ftn_int lwork = -1;
            ftn_double z;
            ftn_int info;
            FORTRAN(pdgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork, &info);
            lwork = static_cast<int>(z + 1);
            std::vector<ftn_double> work(lwork);
            std::vector<ftn_double> tau(std::max(m, n));
            FORTRAN(pdgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(), work.data(), &lwork,
             &info);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::lapack: {
            if (A.comm().size() != 1) {
                RTE_THROW("can't use lapack for distributed matrix; use scalapck instead");
            }
            ftn_int lwork = -1;
            ftn_double z;
            ftn_int info;
            ftn_int lda = A.ld();
            FORTRAN(dgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, &z, &z, &lwork, &info);
            lwork = static_cast<int>(z + 1);
            std::vector<ftn_double> work(lwork);
            std::vector<ftn_double> tau(std::max(m, n));
            FORTRAN(dgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, tau.data(), work.data(), &lwork, &info);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::geqrf<ftn_complex>(ftn_int m, ftn_int n, dmatrix<ftn_complex>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ftn_int lwork = -1;
            ftn_complex z;
            ftn_int info;
            FORTRAN(pcgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork, &info);
            lwork = static_cast<int>(z.real() + 1);
            std::vector<ftn_complex> work(lwork);
            std::vector<ftn_complex> tau(std::max(m, n));
            FORTRAN(pcgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(), work.data(), &lwork,
             &info);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::lapack: {
            if (A.comm().size() != 1) {
                RTE_THROW("can't use lapack for distributed matrix; use scalapck instead");
            }
            ftn_int lwork = -1;
            ftn_complex z;
            ftn_int info;
            ftn_int lda = A.ld();
            FORTRAN(cgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, &z, &z, &lwork, &info);
            lwork = static_cast<int>(z.real() + 1);
            std::vector<ftn_complex> work(lwork);
            std::vector<ftn_complex> tau(std::max(m, n));
            FORTRAN(cgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, tau.data(), work.data(), &lwork, &info);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::geqrf<ftn_single>(ftn_int m, ftn_int n, dmatrix<ftn_single>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ftn_int lwork = -1;
            ftn_single z;
            ftn_int info;
            FORTRAN(psgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork, &info);
            lwork = static_cast<int>(z + 1);
            std::vector<ftn_single> work(lwork);
            std::vector<ftn_single> tau(std::max(m, n));
            FORTRAN(psgeqrf)
            (&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(), work.data(), &lwork,
             &info);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::lapack: {
            if (A.comm().size() != 1) {
                RTE_THROW("can't use lapack for distributed matrix; use scalapck instead");
            }
            ftn_int lwork = -1;
            ftn_single z;
            ftn_int info;
            ftn_int lda = A.ld();
            FORTRAN(sgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, &z, &z, &lwork, &info);
            lwork = static_cast<int>(z + 1);
            std::vector<ftn_single> work(lwork);
            std::vector<ftn_single> tau(std::max(m, n));
            FORTRAN(sgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, tau.data(), work.data(), &lwork, &info);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::axpy(int n, ftn_double_complex const* alpha, ftn_double_complex const* x, int incx, ftn_double_complex* y,
           int incy)
{
    assert(n > 0);
    assert(incx > 0);
    assert(incy > 0);

    switch (la_) {
        case lib_t::blas: {
            FORTRAN(zaxpy)(&n, alpha, x, &incx, y, &incy);
            break;
        }
#if defined(SIRIUS_GPU)
        case lib_t::gpublas: {
            acc::blas::zaxpy(n, reinterpret_cast<const acc_complex_double_t*>(alpha),
                             reinterpret_cast<const acc_complex_double_t*>(x), incx,
                             reinterpret_cast<acc_complex_double_t*>(y), incy);
            break;
        }
#endif
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_single>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_single const* alpha,
                       ftn_single const* A, ftn_int lda, ftn_single const* B, ftn_int ldb, ftn_single const* beta,
                       ftn_single* C, ftn_int ldc, acc::stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(sgemm)
            (&transa, &transb, &m, &n, &k, const_cast<float*>(alpha), const_cast<float*>(A), &lda,
             const_cast<float*>(B), &ldb, const_cast<float*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        case lib_t::spla: {
            splablas::sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_double>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* alpha,
                       ftn_double const* A, ftn_int lda, ftn_double const* B, ftn_int ldb, ftn_double const* beta,
                       ftn_double* C, ftn_int ldc, acc::stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(dgemm)
            (&transa, &transb, &m, &n, &k, const_cast<double*>(alpha), const_cast<double*>(A), &lda,
             const_cast<double*>(B), &ldb, const_cast<double*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        case lib_t::spla: {
            splablas::dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_complex const* alpha,
                        ftn_complex const* A, ftn_int lda, ftn_complex const* B, ftn_int ldb, ftn_complex const* beta,
                        ftn_complex* C, ftn_int ldc, acc::stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(cgemm)
            (&transa, &transb, &m, &n, &k, const_cast<ftn_complex*>(alpha), const_cast<ftn_complex*>(A), &lda,
             const_cast<ftn_complex*>(B), &ldb, const_cast<ftn_complex*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::cgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_float_t const*>(alpha),
                             reinterpret_cast<acc_complex_float_t const*>(A), lda,
                             reinterpret_cast<acc_complex_float_t const*>(B), ldb,
                             reinterpret_cast<acc_complex_float_t const*>(beta),
                             reinterpret_cast<acc_complex_float_t*>(C), ldc, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::cgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_float_t const*>(alpha),
                                 reinterpret_cast<acc_complex_float_t const*>(A), lda,
                                 reinterpret_cast<acc_complex_float_t const*>(B), ldb,
                                 reinterpret_cast<acc_complex_float_t const*>(beta),
                                 reinterpret_cast<acc_complex_float_t*>(C), ldc);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        case lib_t::spla: {
            splablas::cgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_double_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k,
                               ftn_double_complex const* alpha, ftn_double_complex const* A, ftn_int lda,
                               ftn_double_complex const* B, ftn_int ldb, ftn_double_complex const* beta,
                               ftn_double_complex* C, ftn_int ldc, acc::stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(zgemm)
            (&transa, &transb, &m, &n, &k, const_cast<ftn_double_complex*>(alpha), const_cast<ftn_double_complex*>(A),
             &lda, const_cast<ftn_double_complex*>(B), &ldb, const_cast<ftn_double_complex*>(beta), C, &ldc, (ftn_len)1,
             (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::zgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_double_t const*>(alpha),
                             reinterpret_cast<acc_complex_double_t const*>(A), lda,
                             reinterpret_cast<acc_complex_double_t const*>(B), ldb,
                             reinterpret_cast<acc_complex_double_t const*>(beta),
                             reinterpret_cast<acc_complex_double_t*>(C), ldc, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::zgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_double_t const*>(alpha),
                                 reinterpret_cast<acc_complex_double_t const*>(A), lda,
                                 reinterpret_cast<acc_complex_double_t const*>(B), ldb,
                                 reinterpret_cast<acc_complex_double_t const*>(beta),
                                 reinterpret_cast<acc_complex_double_t*>(C), ldc);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        case lib_t::spla: {
            splablas::zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_single>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_single const* alpha,
                       dmatrix<ftn_single> const& A, ftn_int ia, ftn_int ja, dmatrix<ftn_single> const& B, ftn_int ib,
                       ftn_int jb, ftn_single const* beta, dmatrix<ftn_single>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++;
            ja++;
            ib++;
            jb++;
            ic++;
            jc++;
            FORTRAN(psgemm)
            (&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(), B.at(memory_t::host),
             &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(), (ftn_len)1, (ftn_len)1);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_double>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* alpha,
                       dmatrix<ftn_double> const& A, ftn_int ia, ftn_int ja, dmatrix<ftn_double> const& B, ftn_int ib,
                       ftn_int jb, ftn_double const* beta, dmatrix<ftn_double>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++;
            ja++;
            ib++;
            jb++;
            ic++;
            jc++;
            FORTRAN(pdgemm)
            (&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(), B.at(memory_t::host),
             &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(), (ftn_len)1, (ftn_len)1);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_complex const* alpha,
                        dmatrix<ftn_complex> const& A, ftn_int ia, ftn_int ja, dmatrix<ftn_complex> const& B,
                        ftn_int ib, ftn_int jb, ftn_complex const* beta, dmatrix<ftn_complex>& C, ftn_int ic,
                        ftn_int jc)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++;
            ja++;
            ib++;
            jb++;
            ic++;
            jc++;
            FORTRAN(pcgemm)
            (&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(), B.at(memory_t::host),
             &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(), (ftn_len)1, (ftn_len)1);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::gemm<ftn_double_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k,
                               ftn_double_complex const* alpha, dmatrix<ftn_double_complex> const& A, ftn_int ia,
                               ftn_int ja, dmatrix<ftn_double_complex> const& B, ftn_int ib, ftn_int jb,
                               ftn_double_complex const* beta, dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++;
            ja++;
            ib++;
            jb++;
            ic++;
            jc++;
            FORTRAN(pzgemm)
            (&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(), B.at(memory_t::host),
             &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(), (ftn_len)1, (ftn_len)1);
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::hemm<ftn_complex>(char side, char uplo, ftn_int m, ftn_int n, ftn_complex const* alpha, ftn_complex const* A,
                        ftn_len lda, ftn_complex const* B, ftn_len ldb, ftn_complex const* beta, ftn_complex* C,
                        ftn_len ldc)
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(chemm)
            (&side, &uplo, &m, &n, const_cast<ftn_complex*>(alpha), const_cast<ftn_complex*>(A), &lda,
             const_cast<ftn_complex*>(B), &ldb, const_cast<ftn_complex*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::hemm<ftn_double_complex>(char side, char uplo, ftn_int m, ftn_int n, ftn_double_complex const* alpha,
                               ftn_double_complex const* A, ftn_len lda, ftn_double_complex const* B, ftn_len ldb,
                               ftn_double_complex const* beta, ftn_double_complex* C, ftn_len ldc)
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(zhemm)
            (&side, &uplo, &m, &n, const_cast<ftn_double_complex*>(alpha), const_cast<ftn_double_complex*>(A), &lda,
             const_cast<ftn_double_complex*>(B), &ldb, const_cast<ftn_double_complex*>(beta), C, &ldc, (ftn_len)1,
             (ftn_len)1);
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::ger<ftn_single>(ftn_int m, ftn_int n, ftn_single const* alpha, ftn_single const* x, ftn_int incx,
                      ftn_single const* y, ftn_int incy, ftn_single* A, ftn_int lda, acc::stream_id sid) const
{
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(sger)
            (&m, &n, const_cast<ftn_single*>(alpha), const_cast<ftn_single*>(x), &incx, const_cast<ftn_single*>(y),
             &incy, A, &lda);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::sger(m, n, alpha, x, incx, y, incy, A, lda, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
            RTE_THROW("(s,c)ger is not implemented in cublasxt");
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::ger<ftn_double>(ftn_int m, ftn_int n, ftn_double const* alpha, ftn_double const* x, ftn_int incx,
                      ftn_double const* y, ftn_int incy, ftn_double* A, ftn_int lda, acc::stream_id sid) const
{
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(dger)
            (&m, &n, const_cast<ftn_double*>(alpha), const_cast<ftn_double*>(x), &incx, const_cast<ftn_double*>(y),
             &incy, A, &lda);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::dger(m, n, alpha, x, incx, y, incy, A, lda, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
            RTE_THROW("(d,z)ger is not implemented in cublasxt");
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::trmm<ftn_double>(char side, char uplo, char transa, ftn_int m, ftn_int n, ftn_double const* alpha,
                       ftn_double const* A, ftn_int lda, ftn_double* B, ftn_int ldb, acc::stream_id sid) const
{
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(dtrmm)
            (&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_double*>(alpha), const_cast<ftn_double*>(A), &lda, B,
             &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::dtrmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::dtrmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::trmm<ftn_single>(char side, char uplo, char transa, ftn_int m, ftn_int n, ftn_single const* alpha,
                       ftn_single const* A, ftn_int lda, ftn_single* B, ftn_int ldb, acc::stream_id sid) const
{
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(strmm)
            (&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_single*>(alpha), const_cast<ftn_single*>(A), &lda, B,
             &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::strmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::strmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::trmm<ftn_double_complex>(char side, char uplo, char transa, ftn_int m, ftn_int n, ftn_double_complex const* alpha,
                               ftn_double_complex const* A, ftn_int lda, ftn_double_complex* B, ftn_int ldb,
                               acc::stream_id sid) const
{
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(ztrmm)
            (&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_double_complex*>(alpha),
             const_cast<ftn_double_complex*>(A), &lda, B, &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::ztrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_double_t const*>(alpha),
                             reinterpret_cast<acc_complex_double_t const*>(A), lda,
                             reinterpret_cast<acc_complex_double_t*>(B), ldb, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::ztrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_double_t const*>(alpha),
                                 reinterpret_cast<acc_complex_double_t const*>(A), lda,
                                 reinterpret_cast<acc_complex_double_t*>(B), ldb);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::trmm<ftn_complex>(char side, char uplo, char transa, ftn_int m, ftn_int n, ftn_complex const* alpha,
                        ftn_complex const* A, ftn_int lda, ftn_complex* B, ftn_int ldb, acc::stream_id sid) const
{
    switch (la_) {
        case lib_t::blas: {
            FORTRAN(ctrmm)
            (&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_complex*>(alpha), const_cast<ftn_complex*>(A), &lda, B,
             &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU)
            acc::blas::ctrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_float_t const*>(alpha),
                             reinterpret_cast<acc_complex_float_t const*>(A), lda,
                             reinterpret_cast<acc_complex_float_t*>(B), ldb, sid());
#else
            RTE_THROW("not compiled with GPU blas support!");
#endif
            break;
        }
        case lib_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::blas::xt::ctrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_float_t const*>(alpha),
                                 reinterpret_cast<acc_complex_float_t const*>(A), lda,
                                 reinterpret_cast<acc_complex_float_t*>(B), ldb);
#else
            RTE_THROW("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline int
wrap::potrf<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(dpotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::dpotrf('U', n, A, lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pdpotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::potrf<ftn_double>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::potrf<ftn_single>(ftn_int n, ftn_single* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(spotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::spotrf('U', n, A, lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pspotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::potrf<ftn_single>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::potrf<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(zpotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pzpotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::zpotrf('U', n, reinterpret_cast<magmaDoubleComplex*>(A), lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::potrf<ftn_double_complex>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::potrf<ftn_complex>(ftn_int n, ftn_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(cpotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pcpotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::cpotrf('U', n, reinterpret_cast<magmaFloatComplex*>(A), lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::potrf<ftn_complex>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::trtri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(dtrtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pdtrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::dtrtri('U', n, A, lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::trtri<ftn_double>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::trtri<ftn_single>(ftn_int n, ftn_single* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(strtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pstrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::strtri('U', n, A, lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::trtri<ftn_single>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::trtri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(ztrtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pztrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::ztrtri('U', n, reinterpret_cast<magmaDoubleComplex*>(A), lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::trtri<ftn_double_complex>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::trtri<ftn_complex>(ftn_int n, ftn_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(ctrtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pctrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        case lib_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::ctrtri('U', n, reinterpret_cast<magmaFloatComplex*>(A), lda);
#else
            RTE_THROW("not compiled with magma");
#endif
            break;
        }
        case lib_t::gpublas: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            acc::cusolver::trtri<ftn_complex>(n, A, lda);
#else
            RTE_THROW("not compiled with CUDA");
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::gtsv<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* dl, ftn_double* d, ftn_double* du, ftn_double* b,
                       ftn_int ldb) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::gtsv<ftn_double_complex>(ftn_int n, ftn_int nrhs, ftn_double_complex* dl, ftn_double_complex* d,
                               ftn_double_complex* du, ftn_double_complex* b, ftn_int ldb) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::gesv<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* A, ftn_int lda, ftn_double* B, ftn_int ldb) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            std::vector<ftn_int> ipiv(n);
            FORTRAN(dgesv)(&n, &nrhs, A, &lda, &ipiv[0], B, &ldb, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::gesv<ftn_double_complex>(ftn_int n, ftn_int nrhs, ftn_double_complex* A, ftn_int lda, ftn_double_complex* B,
                               ftn_int ldb) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            std::vector<ftn_int> ipiv(n);
            FORTRAN(zgesv)(&n, &nrhs, A, &lda, &ipiv[0], B, &ldb, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

// LU factorization, double
template <>
inline int
wrap::getrf<ftn_double>(ftn_int m, ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(dgetrf)(&m, &n, A, &lda, ipiv, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

// LU factorization, double_complex
template <>
inline int
wrap::getrf<ftn_double_complex>(ftn_int m, ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(zgetrf)(&m, &n, A, &lda, ipiv, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::getrf<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ftn_int info;
            ia++;
            ja++;
            FORTRAN(pzgetrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), ipiv, &info);
            return info;
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::getrs<ftn_double_complex>(char trans, ftn_int n, ftn_int nrhs, const ftn_double_complex* A, ftn_int lda,
                                ftn_int* ipiv, ftn_double_complex* B, ftn_int ldb) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(zgetrs)(&trans, &n, &nrhs, const_cast<ftn_double_complex*>(A), &lda, ipiv, B, &ldb, &info);
            return info;
            break;
        }
#if defined(SIRIUS_GPU)
        case lib_t::gpublas: {
            return acc::lapack::getrs(trans, n, nrhs, reinterpret_cast<const acc_complex_double_t*>(A), lda, ipiv,
                                      reinterpret_cast<acc_complex_double_t*>(B), ldb);
            break;
        }
#endif
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline void
wrap::tranc<ftn_complex>(ftn_int m, ftn_int n, dmatrix<ftn_complex>& A, ftn_int ia, ftn_int ja, dmatrix<ftn_complex>& C,
                         ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ic++;
            jc++;

            auto A_ptr = (A.num_rows_local() * A.num_cols_local() > 0) ? A.at(memory_t::host) : nullptr;
            auto C_ptr = (C.num_rows_local() * C.num_cols_local() > 0) ? C.at(memory_t::host) : nullptr;

            FORTRAN(pctranc)
            (&m, &n, const_cast<ftn_complex*>(&constant<ftn_complex>::one()), A_ptr, &ia, &ja, A.descriptor(),
             const_cast<ftn_complex*>(&constant<ftn_complex>::zero()), C_ptr, &ic, &jc, C.descriptor());
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::tranu<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ic++;
            jc++;

            auto A_ptr = (A.num_rows_local() * A.num_cols_local() > 0) ? A.at(memory_t::host) : nullptr;
            auto C_ptr = (C.num_rows_local() * C.num_cols_local() > 0) ? C.at(memory_t::host) : nullptr;

            FORTRAN(pztranu)
            (&m, &n, const_cast<ftn_double_complex*>(&constant<ftn_double_complex>::one()), A_ptr, &ia, &ja,
             A.descriptor(), const_cast<ftn_double_complex*>(&constant<ftn_double_complex>::zero()), C_ptr, &ic, &jc,
             C.descriptor());
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::tranc<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ic++;
            jc++;

            auto A_ptr = (A.num_rows_local() * A.num_cols_local() > 0) ? A.at(memory_t::host) : nullptr;
            auto C_ptr = (C.num_rows_local() * C.num_cols_local() > 0) ? C.at(memory_t::host) : nullptr;

            FORTRAN(pztranc)
            (&m, &n, const_cast<ftn_double_complex*>(&constant<ftn_double_complex>::one()), A_ptr, &ia, &ja,
             A.descriptor(), const_cast<ftn_double_complex*>(&constant<ftn_double_complex>::zero()), C_ptr, &ic, &jc,
             C.descriptor());
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::tranc<ftn_single>(ftn_int m, ftn_int n, dmatrix<ftn_single>& A, ftn_int ia, ftn_int ja, dmatrix<ftn_single>& C,
                        ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ic++;
            jc++;

            auto A_ptr = (A.num_rows_local() * A.num_cols_local() > 0) ? A.at(memory_t::host) : nullptr;
            auto C_ptr = (C.num_rows_local() * C.num_cols_local() > 0) ? C.at(memory_t::host) : nullptr;

            FORTRAN(pstran)
            (&m, &n, const_cast<ftn_single*>(&constant<ftn_single>::one()), A_ptr, &ia, &ja, A.descriptor(),
             const_cast<ftn_single*>(&constant<ftn_single>::zero()), C_ptr, &ic, &jc, C.descriptor());
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::tranu<ftn_double>(ftn_int m, ftn_int n, dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja, dmatrix<ftn_double>& C,
                        ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ic++;
            jc++;

            auto A_ptr = (A.num_rows_local() * A.num_cols_local() > 0) ? A.at(memory_t::host) : nullptr;
            auto C_ptr = (C.num_rows_local() * C.num_cols_local() > 0) ? C.at(memory_t::host) : nullptr;

            FORTRAN(pdtran)
            (&m, &n, const_cast<ftn_double*>(&constant<ftn_double>::one()), A_ptr, &ia, &ja, A.descriptor(),
             const_cast<ftn_double*>(&constant<ftn_double>::zero()), C_ptr, &ic, &jc, C.descriptor());
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void
wrap::tranc<ftn_double>(ftn_int m, ftn_int n, dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja, dmatrix<ftn_double>& C,
                        ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case lib_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++;
            ja++;
            ic++;
            jc++;

            auto A_ptr = (A.num_rows_local() * A.num_cols_local() > 0) ? A.at(memory_t::host) : nullptr;
            auto C_ptr = (C.num_rows_local() * C.num_cols_local() > 0) ? C.at(memory_t::host) : nullptr;

            FORTRAN(pdtran)
            (&m, &n, const_cast<ftn_double*>(&constant<ftn_double>::one()), A_ptr, &ia, &ja, A.descriptor(),
             const_cast<ftn_double*>(&constant<ftn_double>::zero()), C_ptr, &ic, &jc, C.descriptor());
#else
            RTE_THROW(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
}

// Inversion of LU factorized matrix, double
template <>
inline int
wrap::getri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int nb    = linalg_base::ilaenv(1, "dgetri", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double> work(lwork);

            int32_t info;
            FORTRAN(dgetri)(&n, A, &lda, ipiv, &work[0], &lwork, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

// Inversion of LU factorized matrix, double_complex
template <>
inline int
wrap::getri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int nb    = linalg_base::ilaenv(1, "zgetri", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double_complex> work(lwork);

            int32_t info;
            FORTRAN(zgetri)(&n, A, &lda, ipiv, &work[0], &lwork, &info);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::sytrf<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int nb    = linalg_base::ilaenv(1, "zhetrf", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double_complex> work(lwork);

            ftn_int info;
            FORTRAN(zhetrf)("U", &n, A, &lda, ipiv, &work[0], &lwork, &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::sytrf<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int nb    = linalg_base::ilaenv(1, "dsytrf", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double> work(lwork);

            ftn_int info;
            FORTRAN(dsytrf)("U", &n, A, &lda, ipiv, &work[0], &lwork, &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::sytri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            std::vector<ftn_double> work(n);
            ftn_int info;
            FORTRAN(dsytri)("U", &n, A, &lda, ipiv, &work[0], &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::sytrs<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* A, ftn_int lda, ftn_int* ipiv, ftn_double* b,
                        ftn_int ldb) const
{
    switch (la_) {
        case lib_t::lapack: {
            ftn_int info;
            FORTRAN(dsytrs)("U", &n, &nrhs, A, &lda, ipiv, b, &ldb, &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline int
wrap::sytri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case lib_t::lapack: {
            std::vector<ftn_double_complex> work(n);
            ftn_int info;
            FORTRAN(zhetri)("U", &n, A, &lda, ipiv, &work[0], &info, (ftn_len)1);
            return info;
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template <>
inline std::tuple<ftn_double, ftn_double, ftn_double>
wrap::lartg(ftn_double f, ftn_double g) const
{
    ftn_double cs{0}, sn{0}, r{0};
    switch (la_) {
        case lib_t::lapack: {
            FORTRAN(dlartg)(&f, &g, &cs, &sn, &r);
        }
        default: {
            RTE_THROW(linalg_msg_wrong_type);
            break;
        }
    }
    return std::make_tuple(cs, sn, r);
}

template <typename T>
inline void
check_hermitian(std::string const& name, matrix<T> const& mtrx, int n = -1)
{
    assert(mtrx.size(0) == mtrx.size(1));

    double maxdiff = 0.0;
    int i0         = -1;
    int j0         = -1;

    if (n == -1) {
        n = static_cast<int>(mtrx.size(0));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = std::abs(mtrx(i, j) - std::conj(mtrx(j, i)));
            if (diff > maxdiff) {
                maxdiff = diff;
                i0      = i;
                j0      = j;
            }
        }
    }

    if (maxdiff > 1e-10) {
        std::stringstream s;
        s << name << " is not a symmetric or hermitian matrix" << std::endl
          << "  maximum error: i, j : " << i0 << " " << j0 << " diff : " << maxdiff;

        RTE_WARNING(s);
    }
}

template <typename T>
inline real_type<T>
check_hermitian(dmatrix<T>& mtrx__, int n__)
{
    real_type<T> max_diff{0};
    if (mtrx__.comm().size() != 1) {
        dmatrix<T> tmp(n__, n__, mtrx__.blacs_grid(), mtrx__.bs_row(), mtrx__.bs_col());
        wrap(lib_t::scalapack).tranc(n__, n__, mtrx__, 0, 0, tmp, 0, 0);
        for (int i = 0; i < tmp.num_cols_local(); i++) {
            for (int j = 0; j < tmp.num_rows_local(); j++) {
                max_diff = std::max(max_diff, std::abs(mtrx__(j, i) - tmp(j, i)));
            }
        }
        mtrx__.blacs_grid().comm().template allreduce<real_type<T>, mpi::op_t::max>(&max_diff, 1);
    } else {
        for (int i = 0; i < n__; i++) {
            for (int j = 0; j < n__; j++) {
                max_diff = std::max(max_diff, std::abs(mtrx__(j, i) - std::conj(mtrx__(i, j))));
            }
        }
    }
    return max_diff;
}

template <typename T>
inline double
check_identity(dmatrix<T>& mtrx__, int n__)
{
    real_type<T> max_diff{0};
    for (int i = 0; i < mtrx__.num_cols_local(); i++) {
        int icol = mtrx__.icol(i);
        if (icol < n__) {
            for (int j = 0; j < mtrx__.num_rows_local(); j++) {
                int jrow = mtrx__.irow(j);
                if (jrow < n__) {
                    if (icol == jrow) {
                        max_diff = std::max(max_diff, std::abs(mtrx__(j, i) - static_cast<real_type<T>>(1.0)));
                    } else {
                        max_diff = std::max(max_diff, std::abs(mtrx__(j, i)));
                    }
                }
            }
        }
    }
    mtrx__.comm().template allreduce<real_type<T>, mpi::op_t::max>(&max_diff, 1);
    return max_diff;
}

template <typename T>
inline double
check_diagonal(dmatrix<T>& mtrx__, int n__, mdarray<double, 1> const& diag__)
{
    double max_diff{0};
    for (int i = 0; i < mtrx__.num_cols_local(); i++) {
        int icol = mtrx__.icol(i);
        if (icol < n__) {
            for (int j = 0; j < mtrx__.num_rows_local(); j++) {
                int jrow = mtrx__.irow(j);
                if (jrow < n__) {
                    if (icol == jrow) {
                        max_diff = std::max(max_diff, std::abs(mtrx__(j, i) - diag__[icol]));
                    } else {
                        max_diff = std::max(max_diff, std::abs(mtrx__(j, i)));
                    }
                }
            }
        }
    }
    mtrx__.comm().template allreduce<double, mpi::op_t::max>(&max_diff, 1);
    return max_diff;
}

/** Perform one of the following operations:
 *    A <= U A U^{H} (kind = 0)
 *    A <= U^{H} A U (kind = 1)
 */
template <typename T>
inline void
unitary_similarity_transform(int kind__, dmatrix<T>& A__, dmatrix<T> const& U__, int n__)
{
    // TODO: use memory pool to allocate tmp matrix
    if (!(kind__ == 0 || kind__ == 1)) {
        RTE_THROW("wrong 'kind' parameter");
    }
    char c1 = kind__ == 0 ? 'N' : 'C';
    char c2 = kind__ == 0 ? 'C' : 'N';
    if (A__.comm().size() != 1) {
        dmatrix<T> tmp(n__, n__, A__.blacs_grid(), A__.bs_row(), A__.bs_col());

        /* compute tmp <= U A or U^{H} A */
        wrap(lib_t::scalapack)
                .gemm(c1, 'N', n__, n__, n__, &constant<T>::one(), U__, 0, 0, A__, 0, 0, &constant<T>::zero(), tmp, 0,
                      0);

        /* compute A <= tmp U^{H} or tmp U */
        wrap(lib_t::scalapack)
                .gemm('N', c2, n__, n__, n__, &constant<T>::one(), tmp, 0, 0, U__, 0, 0, &constant<T>::zero(), A__, 0,
                      0);
    } else {
        dmatrix<T> tmp(n__, n__);

        /* compute tmp <= U A or U^{H} A */
        wrap(lib_t::blas)
                .gemm(c1, 'N', n__, n__, n__, &constant<T>::one(), U__.at(memory_t::host), U__.ld(),
                      A__.at(memory_t::host), A__.ld(), &constant<T>::zero(), tmp.at(memory_t::host), tmp.ld());

        /* compute A <= tmp U^{H} or tmp U */
        wrap(lib_t::blas)
                .gemm('N', c2, n__, n__, n__, &constant<T>::one(), tmp.at(memory_t::host), tmp.ld(),
                      U__.at(memory_t::host), U__.ld(), &constant<T>::zero(), A__.at(memory_t::host), A__.ld());
    }
}

} // namespace la

} // namespace sirius

#endif // __LINALG_HPP__

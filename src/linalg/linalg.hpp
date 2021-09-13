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
#ifdef SIRIUS_GPU
#include "gpu/acc_blas.hpp"
#endif
#ifdef SIRIUS_MAGMA
#include "gpu/magma.hpp"
#endif
#include "blas_lapack.h"
#include "memory.hpp"
#include "dmatrix.hpp"
#include "gpu/acc.hpp"
#include "linalg_spla.hpp"

namespace sddk {

namespace _local {
/// check if device id has been set properly
inline bool is_set_device_id()
{
    return acc::get_device_id() == sddk::get_device_id(acc::num_devices());
}
}

#define linalg_msg_wrong_type "[" + std::string(__func__) + "] wrong type of linear algebra library: " + to_string(la_)

const std::string linalg_msg_no_scalapack = "not compiled with ScaLAPACK";

class linalg
{
  private:
    linalg_t la_;
  public:
    linalg(linalg_t la__)
        : la_(la__)
    {
    }

    /*
        matrix - matrix multiplication
    */

    /// General matrix-matrix multiplication.
    /** Compute C = alpha * op(A) * op(B) + beta * op(C) with raw pointers. */
    template <typename T>
    inline void gemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, T const* alpha, T const* A, ftn_int lda,
                     T const* B, ftn_int ldb, T const* beta, T* C, ftn_int ldc, stream_id sid = stream_id(-1)) const;

     /// Distributed general matrix-matrix multiplication.
     /** Compute C = alpha * op(A) * op(B) + beta * op(C) for distributed matrices. */
     template <typename T>
     inline void gemm(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, T const* alpha,
                      sddk::dmatrix<T> const& A, ftn_int ia, ftn_int ja, sddk::dmatrix<T> const& B,
                      ftn_int ib, ftn_int jb, T const* beta, sddk::dmatrix<T>& C, ftn_int ic, ftn_int jc);

    /// Hermitian matrix times a general matrix or vice versa.
    /** Perform one of the matrix-matrix operations \n
     *  C = alpha * A * B + beta * C (side = 'L') \n
     *  C = alpha * B * A + beta * C (side = 'R'), \n
     *  where A is a hermitian matrix with upper (uplo = 'U') of lower (uplo = 'L') triangular part defined.
     */
    template<typename T>
    inline void hemm(char side, char uplo, ftn_int m, ftn_int n, T const* alpha, T const* A, ftn_len lda,
                     T const* B, ftn_len ldb, T const* beta, T* C, ftn_len ldc);

    template <typename T>
    inline void trmm(char side, char uplo, char transa, ftn_int m, ftn_int n, T const* aplha, T const* A, ftn_int lda,
                     T* B, ftn_int ldb, stream_id sid = stream_id(-1)) const;

    /*
        rank2 update
    */

    template<typename T>
    inline void ger(ftn_int m, ftn_int n, T const* alpha, T const* x, ftn_int incx, T const* y, ftn_int incy, T* A, ftn_int lda,
                    stream_id sid = stream_id(-1)) const;

    /*
        matrix factorization
    */

    /// Cholesky factorization
    template <typename T>
    inline int potrf(ftn_int n, T* A, ftn_int lda, ftn_int const* desca = nullptr) const;

    /// LU factorization of general matrix.
    template <typename T>
    inline int getrf(ftn_int m, ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// LU factorization of general matrix.
    template <typename T>
    inline int getrf(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, ftn_int* ipiv) const;

    /// U*D*U^H factorization of hermitian or symmetric matrix.
    template <typename T>
    inline int sytrf(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// solve Ax=b in place of b where A is factorized with sytrf.
    template <typename T>
    inline int sytrs(ftn_int n, ftn_int nrhs, T* A, ftn_int lda, ftn_int* ipiv, T* b, ftn_int ldb) const;

    /*
        matrix inversion
    */

    /// Inversion of a triangular matrix.
    template <typename T>
    inline int trtri(ftn_int n, T* A, ftn_int lda, ftn_int const* desca = nullptr) const;

    template <typename T>
    inline int getri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// Inversion of factorized symmetric triangular matrix.
    template <typename T>
    inline int sytri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv) const;

    /// Invert a general matrix.
    template <typename T>
    inline void geinv(ftn_int n, matrix<T>& A) const
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
    inline void syinv(ftn_int n, matrix<T>& A) const
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
    inline bool sysolve(ftn_int n, matrix<T> &A, mdarray<T, 1> &b) const
    {
        std::vector<int> ipiv(n);
        int info = this->sytrf(n, A.at(memory_t::host), A.ld(), ipiv.data());
        if (info) return false;

        info = this->sytrs(n, 1, A.at(memory_t::host), A.ld(), ipiv.data(), b.at(memory_t::host), b.ld());

        return !info;
    }

    /*
        solution of a linear system
    */

    /// Compute the solution to system of linear equations A * X = B for general tri-diagonal matrix.
    template <typename T>
    inline int gtsv(ftn_int n, ftn_int nrhs, T* dl, T* d, T* du, T* b, ftn_int ldb) const;

    /// Compute the solution to system of linear equations A * X = B for general matrix.
    template <typename T>
    inline int gesv(ftn_int n, ftn_int nrhs, T* A, ftn_int lda, T* B, ftn_int ldb) const;

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
    inline void tranc(ftn_int m, ftn_int n, sddk::dmatrix<T>& A, ftn_int ia, ftn_int ja, sddk::dmatrix<T>& C,
        ftn_int ic, ftn_int jc) const;

    /// Transpose matrix without conjugation.
    template <typename T>
    inline void tranu(ftn_int m, ftn_int n, sddk::dmatrix<T>& A, ftn_int ia, ftn_int ja, sddk::dmatrix<T>& C,
        ftn_int ic, ftn_int jc) const;

    // Constructing a Given's rotation
    template <typename T>
    inline std::tuple<ftn_double, ftn_double, ftn_double> lartg(T f, T g) const;

    template <typename T>
    inline void geqrf(ftn_int m, ftn_int n, sddk::dmatrix<T>& A, ftn_int ia, ftn_int ja);
};

template<>
inline void
linalg::geqrf<ftn_double_complex>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ftn_int lwork = -1;
            ftn_double_complex z;
            ftn_int info;
            FORTRAN(pzgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork,
                             &info);
            lwork = static_cast<int>(z.real() + 1);
            std::vector<ftn_double_complex> work(lwork);
            std::vector<ftn_double_complex> tau(std::max(m, n));
            FORTRAN(pzgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(),
                             work.data(), &lwork, &info);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::geqrf<ftn_double>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ftn_int lwork = -1;
            ftn_double z;
            ftn_int info;
            FORTRAN(pdgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork,
                             &info);
            lwork = static_cast<int>(z + 1);
            std::vector<ftn_double> work(lwork);
            std::vector<ftn_double> tau(std::max(m, n));
            FORTRAN(pdgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(),
                             work.data(), &lwork, &info);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}


template<>
inline void
linalg::geqrf<ftn_complex>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_complex>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ftn_int lwork = -1;
            ftn_complex z;
            ftn_int info;
            FORTRAN(pcgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork,
                             &info);
            lwork = static_cast<int>(z.real() + 1);
            std::vector<ftn_complex> work(lwork);
            std::vector<ftn_complex> tau(std::max(m, n));
            FORTRAN(pcgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(),
                             work.data(), &lwork, &info);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::geqrf<ftn_single>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_single>& A, ftn_int ia, ftn_int ja)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ftn_int lwork = -1;
            ftn_single z;
            ftn_int info;
            FORTRAN(psgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), &z, &z, &lwork,
                             &info);
            lwork = static_cast<int>(z + 1);
            std::vector<ftn_single> work(lwork);
            std::vector<ftn_single> tau(std::max(m, n));
            FORTRAN(psgeqrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), tau.data(),
                             work.data(), &lwork, &info);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

//inline void linalg<CPU>::geqrf<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja)
//{
//    ftn_int lwork = -1;
//    ftn_double_complex z;
//    ftn_int info;
//    ftn_int lda = A.ld();
//    FORTRAN(zgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, &z, &z, &lwork, &info);
//    lwork = static_cast<int>(z.real() + 1);
//    std::vector<ftn_double_complex> work(lwork);
//    std::vector<ftn_double_complex> tau(std::max(m, n));
//    FORTRAN(zgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, tau.data(), work.data(), &lwork, &info);
//}
//
//template <>
//inline void linalg<CPU>::geqrf<ftn_double>(ftn_int m, ftn_int n, dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja)
//{
//    ftn_int lwork = -1;
//    ftn_double z;
//    ftn_int info;
//    ftn_int lda = A.ld();
//    FORTRAN(dgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, &z, &z, &lwork, &info);
//    lwork = static_cast<int>(z + 1);
//    std::vector<ftn_double> work(lwork);
//    std::vector<ftn_double> tau(std::max(m, n));
//    FORTRAN(dgeqrf)(&m, &n, A.at(memory_t::host, ia, ja), &lda, tau.data(), work.data(), &lwork, &info);
//}



template <>
inline void linalg::gemm<ftn_single>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_single const* alpha,
                                     ftn_single const* A, ftn_int lda, ftn_single const* B, ftn_int ldb,
                                     ftn_single const* beta, ftn_single* C, ftn_int ldc, stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(sgemm)
            (&transa, &transb, &m, &n, &k, const_cast<float*>(alpha), const_cast<float*>(A), &lda,
             const_cast<float*>(B), &ldb, const_cast<float*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;
        }
        case linalg_t::spla: {
            splablas::sgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::gemm<ftn_double>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* alpha,
                                      ftn_double const* A, ftn_int lda, ftn_double const* B, ftn_int ldb,
                                      ftn_double const* beta, ftn_double* C, ftn_int ldc, stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(dgemm)(&transa, &transb, &m, &n, &k, const_cast<double*>(alpha), const_cast<double*>(A), &lda,
                           const_cast<double*>(B), &ldb, const_cast<double*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;

        }
        case linalg_t::spla: {
            splablas::dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::gemm<ftn_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_complex const* alpha,
                                      ftn_complex const* A, ftn_int lda, ftn_complex const* B, ftn_int ldb, ftn_complex const *beta,
                                      ftn_complex* C, ftn_int ldc, stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(cgemm)
            (&transa, &transb, &m, &n, &k, const_cast<ftn_complex*>(alpha), const_cast<ftn_complex*>(A), &lda,
             const_cast<ftn_complex*>(B), &ldb, const_cast<ftn_complex*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::cgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_float_t const*>(alpha),
                           reinterpret_cast<acc_complex_float_t const*>(A), lda,
                           reinterpret_cast<acc_complex_float_t const*>(B), ldb,
                           reinterpret_cast<acc_complex_float_t const*>(beta),
                           reinterpret_cast<acc_complex_float_t*>(C), ldc, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::cgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_float_t const*>(alpha),
                               reinterpret_cast<acc_complex_float_t const*>(A), lda,
                               reinterpret_cast<acc_complex_float_t const*>(B), ldb,
                               reinterpret_cast<acc_complex_float_t const*>(beta),
                               reinterpret_cast<acc_complex_float_t*>(C), ldc);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;
        }
        case linalg_t::spla: {
            splablas::cgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::gemm<ftn_double_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k,
                                              ftn_double_complex const* alpha, ftn_double_complex const* A, ftn_int lda,
                                              ftn_double_complex const* B, ftn_int ldb, ftn_double_complex const *beta,
                                              ftn_double_complex* C, ftn_int ldc, stream_id sid) const
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    assert(k > 0);
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(zgemm)(&transa, &transb, &m, &n, &k, const_cast<ftn_double_complex*>(alpha),
                           const_cast<ftn_double_complex*>(A), &lda, const_cast<ftn_double_complex*>(B), &ldb,
                           const_cast<ftn_double_complex*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::zgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_double_t const*>(alpha),
                          reinterpret_cast<acc_complex_double_t const*>(A), lda, reinterpret_cast<acc_complex_double_t const*>(B),
                          ldb, reinterpret_cast<acc_complex_double_t const*>(beta),
                          reinterpret_cast<acc_complex_double_t*>(C), ldc, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;

        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::zgemm(transa, transb, m, n, k, reinterpret_cast<acc_complex_double_t const*>(alpha),
                              reinterpret_cast<acc_complex_double_t const*>(A), lda,
                              reinterpret_cast<acc_complex_double_t const*>(B), ldb,
                              reinterpret_cast<acc_complex_double_t const*>(beta),
                              reinterpret_cast<acc_complex_double_t*>(C), ldc);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;

        }
        case linalg_t::spla: {
            splablas::zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::gemm<ftn_single>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_single const* alpha,
                          sddk::dmatrix<ftn_single> const& A, ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_single> const& B,
                          ftn_int ib, ftn_int jb, ftn_single const* beta, sddk::dmatrix<ftn_single>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++; ja++;
            ib++; jb++;
            ic++; jc++;
            FORTRAN(psgemm)(&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(),
                            B.at(memory_t::host), &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(),
                            (ftn_len)1, (ftn_len)1);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::gemm<ftn_double>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* alpha,
                         sddk::dmatrix<ftn_double> const& A, ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_double> const& B,
                         ftn_int ib, ftn_int jb, ftn_double const* beta, sddk::dmatrix<ftn_double>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++; ja++;
            ib++; jb++;
            ic++; jc++;
            FORTRAN(pdgemm)(&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(),
                            B.at(memory_t::host), &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(),
                            (ftn_len)1, (ftn_len)1);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::gemm<ftn_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k, ftn_complex const* alpha,
                          sddk::dmatrix<ftn_complex> const& A, ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_complex> const& B,
                          ftn_int ib, ftn_int jb, ftn_complex const* beta, sddk::dmatrix<ftn_complex>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++; ja++;
            ib++; jb++;
            ic++; jc++;
            FORTRAN(pcgemm)(&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(),
                            B.at(memory_t::host), &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(),
                            (ftn_len)1, (ftn_len)1);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::gemm<ftn_double_complex>(char transa, char transb, ftn_int m, ftn_int n, ftn_int k,
                                  ftn_double_complex const* alpha, sddk::dmatrix<ftn_double_complex> const& A,
                                  ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_double_complex> const& B,
                                  ftn_int ib, ftn_int jb, ftn_double_complex const* beta,
                                  sddk::dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc)
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(A.ld() != 0);
            assert(B.ld() != 0);
            assert(C.ld() != 0);

            ia++; ja++;
            ib++; jb++;
            ic++; jc++;
            FORTRAN(pzgemm)(&transa, &transb, &m, &n, &k, alpha, A.at(memory_t::host), &ia, &ja, A.descriptor(),
                            B.at(memory_t::host), &ib, &jb, B.descriptor(), beta, C.at(memory_t::host), &ic, &jc, C.descriptor(),
                            (ftn_len)1, (ftn_len)1);
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::hemm<ftn_complex>(char side, char uplo, ftn_int m, ftn_int n, ftn_complex const* alpha, ftn_complex const* A,
                          ftn_len lda, ftn_complex const* B, ftn_len ldb, ftn_complex const* beta, ftn_complex* C, ftn_len ldc)
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(chemm)
            (&side, &uplo, &m, &n, const_cast<ftn_complex*>(alpha), const_cast<ftn_complex*>(A), &lda,
             const_cast<ftn_complex*>(B), &ldb, const_cast<ftn_complex*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void
linalg::hemm<ftn_double_complex>(char side, char uplo, ftn_int m, ftn_int n, ftn_double_complex const* alpha,
                                  ftn_double_complex const* A, ftn_len lda, ftn_double_complex const* B, ftn_len ldb,
                                  ftn_double_complex const* beta, ftn_double_complex* C, ftn_len ldc)
{
    assert(lda > 0);
    assert(ldb > 0);
    assert(ldc > 0);
    assert(m > 0);
    assert(n > 0);
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(zhemm)(&side, &uplo, &m, &n, const_cast<ftn_double_complex*>(alpha),
                           const_cast<ftn_double_complex*>(A), &lda, const_cast<ftn_double_complex*>(B), &ldb,
                           const_cast<ftn_double_complex*>(beta), C, &ldc, (ftn_len)1, (ftn_len)1);
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void linalg::ger<ftn_single>(ftn_int m, ftn_int n, ftn_single const* alpha, ftn_single const* x, ftn_int incx,
                                    ftn_single const* y, ftn_int incy, ftn_single* A, ftn_int lda, stream_id sid) const
{
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(sger)(&m, &n, const_cast<ftn_single*>(alpha), const_cast<ftn_single*>(x), &incx,
                          const_cast<ftn_single*>(y), &incy, A, &lda);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::sger(m, n, alpha, x, incx, y, incy, A, lda, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
            throw std::runtime_error("(s,c)ger is not implemented in cublasxt");
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void linalg::ger<ftn_double>(ftn_int m, ftn_int n, ftn_double const* alpha, ftn_double const* x, ftn_int incx,
                                     ftn_double const* y, ftn_int incy, ftn_double* A, ftn_int lda, stream_id sid) const
{
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(dger)(&m, &n, const_cast<ftn_double*>(alpha), const_cast<ftn_double*>(x), &incx,
                          const_cast<ftn_double*>(y), &incy, A, &lda);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::dger(m, n, alpha, x, incx, y, incy, A, lda, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
            throw std::runtime_error("(d,z)ger is not implemented in cublasxt");
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::trmm<ftn_double>(char side, char uplo, char transa, ftn_int m, ftn_int n, ftn_double const* alpha,
                                      ftn_double const* A, ftn_int lda, ftn_double* B, ftn_int ldb, stream_id sid) const
{
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(dtrmm)(&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_double*>(alpha),
                           const_cast<ftn_double*>(A), &lda, B, &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case  linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::dtrmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::dtrmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::trmm<ftn_single>(char side, char uplo, char transa, ftn_int m, ftn_int n, ftn_single const* alpha,
                                     ftn_single const* A, ftn_int lda, ftn_single* B, ftn_int ldb, stream_id sid) const
{
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(strmm)(&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_single*>(alpha),
                           const_cast<ftn_single*>(A), &lda, B, &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case  linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::strmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::strmm(side, uplo, transa, 'N', m, n, alpha, A, lda, B, ldb);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::trmm<ftn_double_complex>(char side, char uplo, char transa, ftn_int m, ftn_int n,
                                             ftn_double_complex const* alpha, ftn_double_complex const* A,
                                             ftn_int lda, ftn_double_complex* B, ftn_int ldb, stream_id sid) const
{
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(ztrmm)(&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_double_complex*>(alpha),
                           const_cast<ftn_double_complex*>(A), &lda, B, &ldb, (ftn_len)1, (ftn_len)1,
                           (ftn_len)1, (ftn_len)1);
            break;
        }
        case  linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::ztrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_double_t const*>(alpha),
                          reinterpret_cast<acc_complex_double_t const*>(A), lda,
                          reinterpret_cast<acc_complex_double_t*>(B), ldb, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::ztrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_double_t const*>(alpha),
                              reinterpret_cast<acc_complex_double_t const*>(A), lda, reinterpret_cast<acc_complex_double_t*>(B), ldb);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::trmm<ftn_complex>(char side, char uplo, char transa, ftn_int m, ftn_int n,
                                      ftn_complex const* alpha, ftn_complex const* A,
                                      ftn_int lda, ftn_complex* B, ftn_int ldb, stream_id sid) const
{
    switch (la_) {
        case linalg_t::blas: {
            FORTRAN(ctrmm)
                (&side, &uplo, &transa, "N", &m, &n, const_cast<ftn_complex*>(alpha), const_cast<ftn_complex*>(A), &lda, B,
                 &ldb, (ftn_len)1, (ftn_len)1, (ftn_len)1, (ftn_len)1);
            break;
        }
        case linalg_t::gpublas: {
#if defined(SIRIUS_GPU)
            accblas::ctrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_float_t const*>(alpha),
                           reinterpret_cast<acc_complex_float_t const*>(A), lda,
                           reinterpret_cast<acc_complex_float_t*>(B), ldb, sid());
#else
            throw std::runtime_error("not compiled with GPU blas support!");
#endif
            break;
        }
        case linalg_t::cublasxt: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
            accblas::xt::ctrmm(side, uplo, transa, 'N', m, n, reinterpret_cast<acc_complex_float_t const*>(alpha),
                               reinterpret_cast<acc_complex_float_t const*>(A), lda,
                               reinterpret_cast<acc_complex_float_t*>(B), ldb);
#else
            throw std::runtime_error("not compiled with cublasxt");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline int linalg::potrf<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(dpotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::dpotrf('U', n, A, lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pdpotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::potrf<ftn_single>(ftn_int n, ftn_single* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(spotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::spotrf('U', n, A, lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pspotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::potrf<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(zpotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pzpotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::zpotrf('U', n, reinterpret_cast<magmaDoubleComplex*>(A), lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::potrf<ftn_complex>(ftn_int n, ftn_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(cpotrf)("U", &n, A, &lda, &info, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pcpotrf)("U", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::cpotrf('U', n, reinterpret_cast<magmaFloatComplex*>(A), lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::trtri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(dtrtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pdtrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::dtrtri('U', n, A, lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::trtri<ftn_single>(ftn_int n, ftn_single* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(strtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pstrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::strtri('U', n, A, lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::trtri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(ztrtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pztrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::ztrtri('U', n, reinterpret_cast<magmaDoubleComplex*>(A), lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::trtri<ftn_complex>(ftn_int n, ftn_complex* A, ftn_int lda, ftn_int const* desca) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(ctrtri)("U", "N", &n, A, &lda, &info, (ftn_len)1, (ftn_len)1);
            return info;
            break;
        }
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            assert(desca != nullptr);
            ftn_int ia{1};
            ftn_int ja{1};
            ftn_int info;
            FORTRAN(pctrtri)("U", "N", &n, A, &ia, &ja, const_cast<ftn_int*>(desca), &info, (ftn_len)1, (ftn_len)1);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        case linalg_t::magma: {
#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
            return magma::ctrtri('U', n, reinterpret_cast<magmaFloatComplex*>(A), lda);
#else
            throw std::runtime_error("not compiled with magma");
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::gtsv<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* dl, ftn_double* d, ftn_double* du,
                                     ftn_double* b, ftn_int ldb) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::gtsv<ftn_double_complex>(ftn_int n, ftn_int nrhs, ftn_double_complex* dl, ftn_double_complex* d,
                                             ftn_double_complex* du, ftn_double_complex* b, ftn_int ldb) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::gesv<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* A, ftn_int lda, ftn_double* B, ftn_int ldb) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            std::vector<ftn_int> ipiv(n);
            FORTRAN(dgesv)(&n, &nrhs, A, &lda, &ipiv[0], B, &ldb, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::gesv<ftn_double_complex>(ftn_int n, ftn_int nrhs, ftn_double_complex* A, ftn_int lda,
                                             ftn_double_complex* B, ftn_int ldb) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            std::vector<ftn_int> ipiv(n);
            FORTRAN(zgesv)(&n, &nrhs, A, &lda, &ipiv[0], B, &ldb, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

// LU factorization, double
template<>
inline int linalg::getrf<ftn_double>(ftn_int m, ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(dgetrf)(&m, &n, A, &lda, ipiv, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

// LU factorization, double_complex
template<>
inline int linalg::getrf<ftn_double_complex>(ftn_int m, ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(zgetrf)(&m, &n, A, &lda, ipiv, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::getrf<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A,
                                              ftn_int ia, ftn_int ja, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined (SIRIUS_SCALAPACK)
            ftn_int info;
            ia++;
            ja++;
            FORTRAN(pzgetrf)(&m, &n, A.at(memory_t::host), &ia, &ja, const_cast<int*>(A.descriptor()), ipiv, &info);
            return info;
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline void linalg::tranc<ftn_complex>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_complex>& A,
                           ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_complex>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ic++; jc++;

            FORTRAN(pctranc)(&m, &n, const_cast<ftn_complex*>(&linalg_const<ftn_complex>::one()),
                             A.at(memory_t::host), &ia, &ja, A.descriptor(),
                             const_cast<ftn_complex*>(&linalg_const<ftn_complex>::zero()),
                             C.at(memory_t::host), &ic, &jc, C.descriptor());
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void linalg::tranu<ftn_double_complex>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_double_complex>& A,
    ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ic++; jc++;

            FORTRAN(pztranu)(&m, &n, const_cast<ftn_double_complex*>(&linalg_const<ftn_double_complex>::one()),
                             A.at(memory_t::host), &ia, &ja, A.descriptor(),
                             const_cast<ftn_double_complex*>(&linalg_const<ftn_double_complex>::zero()),
                             C.at(memory_t::host), &ic, &jc, C.descriptor());
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template<>
inline void linalg::tranc<ftn_double_complex>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_double_complex>& A,
    ftn_int ia, ftn_int ja, sddk::dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ic++; jc++;

            FORTRAN(pztranc)(&m, &n, const_cast<ftn_double_complex*>(&linalg_const<ftn_double_complex>::one()),
                             A.at(memory_t::host), &ia, &ja, A.descriptor(),
                             const_cast<ftn_double_complex*>(&linalg_const<ftn_double_complex>::zero()),
                             C.at(memory_t::host), &ic, &jc, C.descriptor());
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::tranc<ftn_single>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_single>& A, ftn_int ia, ftn_int ja,
                                      sddk::dmatrix<ftn_single>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ic++; jc++;

            FORTRAN(pstran)(&m, &n, const_cast<ftn_single*>(&linalg_const<ftn_single>::one()), A.at(memory_t::host),
                            &ia, &ja, A.descriptor(), const_cast<ftn_single*>(&linalg_const<ftn_single>::zero()),
                            C.at(memory_t::host), &ic, &jc, C.descriptor());
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::tranu<ftn_double>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja,
    sddk::dmatrix<ftn_double>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ic++; jc++;

            FORTRAN(pdtran)(&m, &n, const_cast<ftn_double*>(&linalg_const<ftn_double>::one()), A.at(memory_t::host),
                            &ia, &ja, A.descriptor(), const_cast<ftn_double*>(&linalg_const<ftn_double>::zero()),
                            C.at(memory_t::host), &ic, &jc, C.descriptor());
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <>
inline void linalg::tranc<ftn_double>(ftn_int m, ftn_int n, sddk::dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja,
    sddk::dmatrix<ftn_double>& C, ftn_int ic, ftn_int jc) const
{
    switch (la_) {
        case linalg_t::scalapack: {
#if defined(SIRIUS_SCALAPACK)
            ia++; ja++;
            ic++; jc++;

            FORTRAN(pdtran)(&m, &n, const_cast<ftn_double*>(&linalg_const<ftn_double>::one()), A.at(memory_t::host),
                            &ia, &ja, A.descriptor(), const_cast<ftn_double*>(&linalg_const<ftn_double>::zero()),
                            C.at(memory_t::host), &ic, &jc, C.descriptor());
#else
            throw std::runtime_error(linalg_msg_no_scalapack);
#endif
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

// Inversion of LU factorized matrix, double
template<>
inline int linalg::getri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int nb = linalg_base::ilaenv(1, "dgetri", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double> work(lwork);

            int32_t info;
            FORTRAN(dgetri)(&n, A, &lda, ipiv, &work[0], &lwork, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

// Inversion of LU factorized matrix, double_complex
template<>
inline int linalg::getri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int nb = linalg_base::ilaenv(1, "zgetri", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double_complex> work(lwork);

            int32_t info;
            FORTRAN(zgetri)(&n, A, &lda, ipiv, &work[0], &lwork, &info);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::sytrf<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int nb = linalg_base::ilaenv(1, "zhetrf", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double_complex> work(lwork);

            ftn_int info;
            FORTRAN(zhetrf)("U", &n, A, &lda, ipiv, &work[0], &lwork, &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::sytrf<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int nb = linalg_base::ilaenv(1, "dsytrf", "U", n, -1, -1, -1);
            ftn_int lwork = n * nb;
            std::vector<ftn_double> work(lwork);

            ftn_int info;
            FORTRAN(dsytrf)("U", &n, A, &lda, ipiv, &work[0], &lwork, &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::sytri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            std::vector<ftn_double> work(n);
            ftn_int info;
            FORTRAN(dsytri)("U", &n, A, &lda, ipiv, &work[0], &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::sytrs<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* A, ftn_int lda, ftn_int* ipiv, ftn_double* b, ftn_int ldb) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_int info;
            FORTRAN(dsytrs)("U", &n, &nrhs, A, &lda, ipiv, b, &ldb, &info, (ftn_len)1);
            return info;
            break;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline int linalg::sytri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv) const
{
    switch (la_) {
        case linalg_t::lapack: {
            std::vector<ftn_double_complex> work(n);
            ftn_int info;
            FORTRAN(zhetri)("U", &n, A, &lda, ipiv, &work[0], &info, (ftn_len)1);
            return info;
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
    return -1;
}

template<>
inline std::tuple<ftn_double, ftn_double, ftn_double> linalg::lartg(ftn_double f, ftn_double g) const
{
    switch (la_) {
        case linalg_t::lapack: {
            ftn_double cs, sn, r;
            FORTRAN(dlartg)(&f, &g, &cs, &sn, &r);
            return std::make_tuple(cs, sn, r);
        }
        default: {
            throw std::runtime_error(linalg_msg_wrong_type);
            break;
        }
    }
}

template <typename T>
inline void check_hermitian(const std::string& name, matrix<T> const& mtrx, int n = -1)
{
    assert(mtrx.size(0) == mtrx.size(1));

    double maxdiff = 0.0;
    int i0 = -1;
    int j0 = -1;

    if (n == -1) {
        n = static_cast<int>(mtrx.size(0));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = std::abs(mtrx(i, j) - std::conj(mtrx(j, i)));
            if (diff > maxdiff) {
                maxdiff = diff;
                i0 = i;
                j0 = j;
            }
        }
    }

    if (maxdiff > 1e-10) {
        std::stringstream s;
        s << name << " is not a symmetric or hermitian matrix" << std::endl
          << "  maximum error: i, j : " << i0 << " " << j0 << " diff : " << maxdiff;

        WARNING(s);
    }
}

template <typename T>
inline real_type<T> check_hermitian(dmatrix<T>& mtrx__, int n__)
{
    real_type<T> max_diff{0};
#ifdef SIRIUS_SCALAPACK
    dmatrix<T> tmp(n__, n__, mtrx__.blacs_grid(), mtrx__.bs_row(), mtrx__.bs_col());
    linalg(linalg_t::scalapack).tranc(n__, n__, mtrx__, 0, 0, tmp, 0, 0);
    for (int i = 0; i < tmp.num_cols_local(); i++) {
        for (int j = 0; j < tmp.num_rows_local(); j++) {
            max_diff = std::max(max_diff, std::abs(mtrx__(j, i) - tmp(j, i)));
        }
    }
    mtrx__.blacs_grid().comm().template allreduce<real_type<T>, mpi_op_t::max>(&max_diff, 1);
#else
    for (int i = 0; i < n__; i++) {
        for (int j = 0; j < n__; j++) {
            max_diff = std::max(max_diff, std::abs(mtrx__(j, i) - std::conj(mtrx__(i, j))));
        }
    }
#endif
    return max_diff;
}

// instantiate for the function of required types
template double check_hermitian<double>(dmatrix<double>& mtrx__, int n__);
template double check_hermitian<std::complex<double>>(dmatrix<std::complex<double>>& mtrx__, int n__);
#ifdef USE_FP32
template float  check_hermitian<float>(dmatrix<float>& mtrx__, int n__);
template float  check_hermitian<std::complex<float>>(dmatrix<std::complex<float>>& mtrx__, int n__);
#endif

template <typename T>
inline double check_identity(dmatrix<T>& mtrx__, int n__)
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
    mtrx__.comm().template allreduce<real_type<T>, mpi_op_t::max>(&max_diff, 1);
    return max_diff;
}

template <typename T>
inline double check_diagonal(dmatrix<T>& mtrx__, int n__, sddk::mdarray<double, 1> const& diag__)
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
    mtrx__.comm().template allreduce<double,  mpi_op_t::max>(&max_diff, 1);
    return max_diff;
}

} // namespace sddk

#endif // __LINALG_HPP__

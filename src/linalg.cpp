// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file linalg.cpp
 *
 *  \brief Contains full specializations of linear algebra interface classes.
 */

#include "linalg.h"
#include "constants.h"
#ifdef __GPU
#include "gpu.h"
#endif

#if defined(__SCALAPACK) && defined(__PILAENV_BLOCKSIZE)
extern "C" int pilaenv_(int* ctxt, char* prec) 
{
    return __PILAENV_BLOCKSIZE;
}
#endif

ftn_double_complex linalg_base::zone = double_complex(1, 0);
ftn_double_complex linalg_base::zzero = double_complex(0, 0);

template<>
void linalg<CPU>::gemv<ftn_double_complex>(int trans,
                                           ftn_int m,
                                           ftn_int n,
                                           ftn_double_complex alpha,
                                           ftn_double_complex const* A,
                                           ftn_int lda,
                                           ftn_double_complex const* x,
                                           ftn_int incx,
                                           ftn_double_complex beta,
                                           ftn_double_complex* y,
                                           ftn_int incy)
{
    const char *trans_c[] = {"N", "T", "C"};

    FORTRAN(zgemv)(trans_c[trans], &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy, 1);
}

template<>
void linalg<CPU>::gemv<ftn_double>(int trans,
                                   ftn_int m,
                                   ftn_int n,
                                   ftn_double alpha,
                                   ftn_double const* A,
                                   ftn_int lda,
                                   ftn_double const* x,
                                   ftn_int incx,
                                   ftn_double beta,
                                   ftn_double* y,
                                   ftn_int incy)
{
    const char *trans_c[] = {"N", "T", "C"};

    FORTRAN(dgemv)(trans_c[trans], &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy, 1);
}

template<> 
void linalg<CPU>::hemm<ftn_double_complex>(int side, int uplo, ftn_int m, ftn_int n, ftn_double_complex alpha, 
                                           ftn_double_complex* A, ftn_int lda, ftn_double_complex* B, ftn_int ldb, 
                                           ftn_double_complex beta, ftn_double_complex* C, ftn_int ldc)
{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, (ftn_len)1, 
                   (ftn_len)1);
}

template<> 
void linalg<CPU>::hemm<ftn_double_complex>(int side, int uplo, ftn_int m, ftn_int n, ftn_double_complex alpha, 
                                           matrix<ftn_double_complex>& A, matrix<ftn_double_complex>& B,
                                           ftn_double_complex beta, matrix<ftn_double_complex>& C)
{
    hemm(side, uplo, m, n, alpha, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), beta, C.at<CPU>(), C.ld());
}

// C = alpha * op(A) * op(B) + beta * op(C), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, ftn_double alpha,
                                   ftn_double const* A, ftn_int lda, ftn_double const* B, ftn_int ldb, ftn_double beta,
                                   ftn_double* C, ftn_int ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc,
                   (ftn_len)1, (ftn_len)1);
}

// C = alpha * op(A) * op(B) + beta * op(C), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, ftn_double_complex const* A, ftn_int lda,
                                           ftn_double_complex const* B, ftn_int ldb, ftn_double_complex beta,
                                           ftn_double_complex* C, ftn_int ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc,
                   (ftn_len)1, (ftn_len)1);
}

// C = op(A) * op(B), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, ftn_double const* A, ftn_int lda, 
                                   ftn_double const* B, ftn_int ldb, ftn_double* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
}

// C = op(A) * op(B), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex const* A, ftn_int lda, ftn_double_complex const* B, ftn_int ldb,
                                           ftn_double_complex* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, zone, A, lda, B, ldb, zzero, C, ldc);
}

// C = alpha * op(A) * op(B) + beta * op(C), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, ftn_double alpha,
                                   matrix<ftn_double> const& A, matrix<ftn_double> const& B, ftn_double beta, matrix<ftn_double>& C)
{
    gemm(transa, transb, m, n, k, alpha, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), beta, C.at<CPU>(), C.ld());
}

// C = alpha * op(A) * op(B) + beta * op(C), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, matrix<ftn_double_complex> const& A,
                                           matrix<ftn_double_complex> const& B, ftn_double_complex beta,
                                           matrix<ftn_double_complex>& C)
{
    gemm(transa, transb, m, n, k, alpha, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), beta, C.at<CPU>(), C.ld());
}

// C = op(A) * op(B), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, matrix<ftn_double> const& A, 
                                   matrix<ftn_double> const& B, matrix<ftn_double>& C)
{
    gemm(transa, transb, m, n, k, 1.0, A, B, 0.0, C);
}

// C = op(A) * op(B), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           matrix<ftn_double_complex> const& A, matrix<ftn_double_complex> const& B,
                                           matrix<ftn_double_complex>& C)
{
    gemm(transa, transb, m, n, k, ftn_double_complex(1, 0), A, B, ftn_double_complex(0, 0), C);
}

// LU factorization, double
template<> 
ftn_int linalg<CPU>::getrf<ftn_double>(ftn_int m, ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv)
{
    ftn_int info;
    FORTRAN(dgetrf)(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// LU factorization, double_complex
template<> 
ftn_int linalg<CPU>::getrf<ftn_double_complex>(ftn_int m, ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv)
{
    ftn_int info;
    FORTRAN(zgetrf)(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// Inversion of LU factorized matrix, double
template<> 
ftn_int linalg<CPU>::getri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv)
{
    ftn_int nb = ilaenv(1, "dgetri", "U", n, -1, -1, -1);
    ftn_int lwork = n * nb;
    std::vector<ftn_double> work(lwork);

    int32_t info;
    FORTRAN(dgetri)(&n, A, &lda, ipiv, &work[0], &lwork, &info);
    return info;
}

// Inversion of LU factorized matrix, double_complex
template<> 
ftn_int linalg<CPU>::getri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv)
{
    ftn_int nb = ilaenv(1, "zgetri", "U", n, -1, -1, -1);
    ftn_int lwork = n * nb;
    std::vector<ftn_double_complex> work(lwork);

    int32_t info;
    FORTRAN(zgetri)(&n, A, &lda, ipiv, &work[0], &lwork, &info);
    return info;
}

// Inversion of general matrix, double
template <>
void linalg<CPU>::geinv<ftn_double>(ftn_int n, matrix<ftn_double>& A)
{
    std::vector<int> ipiv(n);
    int info = getrf(n, n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("getrf returned %i\n", info);
        exit(-1);
    }

    info = getri(n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("getri returned %i\n", info);
        exit(-1);
    }
}

// Inversion of general matrix, double_complex
template <>
void linalg<CPU>::geinv<ftn_double_complex>(ftn_int n, matrix<ftn_double_complex>& A)
{
    std::vector<int> ipiv(n);
    int info = getrf(n, n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("getrf returned %i\n", info);
        exit(-1);
    }

    info = getri(n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("getri returned %i\n", info);
        exit(-1);
    }
}

template<> 
ftn_int linalg<CPU>::hetrf<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv)
{
    ftn_int nb = ilaenv(1, "zhetrf", "U", n, -1, -1, -1);
    ftn_int lwork = n * nb;
    std::vector<ftn_double_complex> work(lwork);

    ftn_int info;
    FORTRAN(zhetrf)("U", &n, A, &lda, ipiv, &work[0], &lwork, &info, (ftn_len)1);
    return info;
}

template<> 
ftn_int linalg<CPU>::hetri<ftn_double_complex>(ftn_int n, ftn_double_complex* A, ftn_int lda, ftn_int* ipiv)
{
    std::vector<ftn_double_complex> work(n);
    ftn_int info;
    FORTRAN(zhetri)("U", &n, A, &lda, ipiv, &work[0], &info, (ftn_len)1);
    return info;
}

// Inversion of hermitian matrix, double_complex
template <>
void linalg<CPU>::heinv<ftn_double_complex>(ftn_int n, matrix<ftn_double_complex>& A)
{
    std::vector<int> ipiv(n);
    int info = hetrf(n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("hetrf returned %i\n", info);
        exit(-1);
    }

    info = hetri(n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("hetri returned %i\n", info);
        exit(-1);
    }
}

template<> 
ftn_int linalg<CPU>::sytrf<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv)
{
    ftn_int nb = ilaenv(1, "dsytrf", "U", n, -1, -1, -1);
    ftn_int lwork = n * nb;
    std::vector<ftn_double> work(lwork);

    ftn_int info;
    FORTRAN(dsytrf)("U", &n, A, &lda, ipiv, &work[0], &lwork, &info, (ftn_len)1);
    return info;
}

template<> 
ftn_int linalg<CPU>::sytri<ftn_double>(ftn_int n, ftn_double* A, ftn_int lda, ftn_int* ipiv)
{
    std::vector<ftn_double> work(n);
    ftn_int info;
    FORTRAN(dsytri)("U", &n, A, &lda, ipiv, &work[0], &info, (ftn_len)1);
    return info;
}

template <>
void linalg<CPU>::syinv<ftn_double>(ftn_int n, matrix<ftn_double>& A)
{
    std::vector<int> ipiv(n);
    int info = sytrf(n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("sytrf returned %i\n", info);
        exit(-1);
    }

    info = sytri(n, A.at<CPU>(), A.ld(), &ipiv[0]);
    if (info)
    {
        printf("sytri returned %i\n", info);
        exit(-1);
    }
}

template<> 
ftn_int linalg<CPU>::gesv<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* A, ftn_int lda, ftn_double* B, ftn_int ldb)
{
    ftn_int info;
    std::vector<ftn_int> ipiv(n);
    FORTRAN(dgesv)(&n, &nrhs, A, &lda, &ipiv[0], B, &ldb, &info);
    return info;
}

template<> 
ftn_int linalg<CPU>::gesv<ftn_double_complex>(ftn_int n, ftn_int nrhs, ftn_double_complex* A, ftn_int lda,
                                              ftn_double_complex* B, ftn_int ldb)
{
    ftn_int info;
    std::vector<ftn_int> ipiv(n);
    FORTRAN(zgesv)(&n, &nrhs, A, &lda, &ipiv[0], B, &ldb, &info);
    return info;
}

template<> 
ftn_int linalg<CPU>::gtsv<ftn_double>(ftn_int n, ftn_int nrhs, ftn_double* dl, ftn_double* d, ftn_double* du,
                                      ftn_double* b, ftn_int ldb)
{
    ftn_int info;
    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;
}

template<> 
ftn_int linalg<CPU>::gtsv<ftn_double_complex>(ftn_int n, ftn_int nrhs, ftn_double_complex* dl, ftn_double_complex* d,
                                              ftn_double_complex* du, ftn_double_complex* b, ftn_int ldb)
{
    ftn_int info;
    FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;
}

#ifdef __SCALAPACK
template<>
ftn_int linalg<CPU>::getrf<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A,
                                               ftn_int ia, ftn_int ja, ftn_int* ipiv)
{
    ftn_int info;
    ia++;
    ja++;
    FORTRAN(pzgetrf)(&m, &n, A.at<CPU>(), &ia, &ja, A.descriptor(), ipiv, &info);
    return info;
}

template<>
ftn_int linalg<CPU>::getri<ftn_double_complex>(ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                               ftn_int* ipiv)
{
    ftn_int info;
    ia++;
    ja++;


    ftn_int lwork, liwork, i;
    ftn_double_complex z;
    i = -1;
    /* query work sizes */
    FORTRAN(pzgetri)(&n, A.at<CPU>(), &ia, &ja, A.descriptor(), &ipiv[0], &z, &i, &liwork, &i, &info);

    lwork = (int)real(z) + 1;
    std::vector<ftn_double_complex> work(lwork);
    std::vector<ftn_int> iwork(liwork);

    FORTRAN(pzgetri)(&n, A.at<CPU>(), &ia, &ja, A.descriptor(), &ipiv[0], &work[0], &lwork, &iwork[0], &liwork, &info);

    return info;
}

template<>
void linalg<CPU>::geinv<ftn_double_complex>(ftn_int n, dmatrix<ftn_double_complex>& A)
{
    std::vector<ftn_int> ipiv(A.num_rows_local() + A.bs_row());
    ftn_int info = getrf(n, n, A, 0, 0, &ipiv[0]);
    if (info)
    {
        printf("getrf returned %i\n", info);
        exit(-1);
    }

    info = getri(n, A, 0, 0, &ipiv[0]);
    if (info)
    {
        printf("getri returned %i\n", info);
        exit(-1);
    }
}

template <>
void linalg<CPU>::tranc<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                            dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc)
{
    ia++; ja++;
    ic++; jc++;

    pztranc(m, n, zone, A.at<CPU>(), ia, ja, A.descriptor(), zzero, C.at<CPU>(), ic, jc, C.descriptor());
}

template <>
void linalg<CPU>::tranu<ftn_double_complex>(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                            dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc)
{
    ia++; ja++;
    ic++; jc++;

    pztranu(m, n, zone, A.at<CPU>(), ia, ja, A.descriptor(), zzero, C.at<CPU>(), ic, jc, C.descriptor());
}

template <>
void linalg<CPU>::gemr2d(ftn_int m, ftn_int n, dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                         dmatrix<ftn_double_complex>& B, ftn_int ib, ftn_int jb, ftn_int gcontext)
{
    ia++; ja++;
    ib++; jb++;
    FORTRAN(pzgemr2d)(&m, &n, A.at<CPU>(), &ia, &ja, A.descriptor(), B.at<CPU>(), &ib, &jb, B.descriptor(), &gcontext);
}

template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                   ftn_double alpha, dmatrix<ftn_double>& A, ftn_int ia, ftn_int ja,
                                   dmatrix<ftn_double>& B, ftn_int ib, ftn_int jb, ftn_double beta, 
                                   dmatrix<ftn_double>& C, ftn_int ic, ftn_int jc)
{
    const char *trans[] = {"N", "T", "C"};

    ia++; ja++;
    ib++; jb++;
    ic++; jc++;
    FORTRAN(pdgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, A.at<CPU>(), &ia, &ja, A.descriptor(), 
                    B.at<CPU>(), &ib, &jb, B.descriptor(), &beta, C.at<CPU>(), &ic, &jc, C.descriptor(),
                    (ftn_len)1, (ftn_len)1);
}

template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, ftn_double alpha, 
                                   dmatrix<ftn_double>& A, dmatrix<ftn_double>& B, ftn_double beta,
                                   dmatrix<ftn_double>& C)
{
    gemm(transa, transb, m, n, k, alpha, A, 0, 0, B, 0, 0, beta, C, 0, 0);
}

template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, 
                                           dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                           dmatrix<ftn_double_complex>& B, ftn_int ib, ftn_int jb,
                                           ftn_double_complex beta, 
                                           dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc)
{
    const char *trans[] = {"N", "T", "C"};

    ia++; ja++;
    ib++; jb++;
    ic++; jc++;
    FORTRAN(pzgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, A.at<CPU>(), &ia, &ja, A.descriptor(), 
                    B.at<CPU>(), &ib, &jb, B.descriptor(), &beta, C.at<CPU>(), &ic, &jc, C.descriptor(),
                    (ftn_len)1, (ftn_len)1);
}

template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, 
                                           dmatrix<ftn_double_complex>& A, dmatrix<ftn_double_complex>& B,
                                           ftn_double_complex beta, dmatrix<ftn_double_complex>& C)
{
    gemm(transa, transb, m, n, k, alpha, A, 0, 0, B, 0, 0, beta, C, 0, 0);
}
#else
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, 
                                           dmatrix<ftn_double_complex>& A, ftn_int ia, ftn_int ja,
                                           dmatrix<ftn_double_complex>& B, ftn_int ib, ftn_int jb,
                                           ftn_double_complex beta, 
                                           dmatrix<ftn_double_complex>& C, ftn_int ic, ftn_int jc)
{
    gemm(transa, transb, m, n, k, alpha, A.at<CPU>(ia, ja), A.ld(), B.at<CPU>(ib, jb), B.ld(), beta, C.at<CPU>(ic, jc), C.ld());
}

template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, 
                                           dmatrix<ftn_double_complex>& A, dmatrix<ftn_double_complex>& B,
                                           ftn_double_complex beta, dmatrix<ftn_double_complex>& C)
{
    gemm(transa, transb, m, n, k, alpha, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), beta, C.at<CPU>(), C.ld());
}
#endif

#ifdef __GPU
template<>
void linalg<GPU>::gemv<ftn_double_complex>(int trans, ftn_int m, ftn_int n, ftn_double_complex* alpha,
                                           ftn_double_complex* A, ftn_int lda, ftn_double_complex* x, ftn_int incx,
                                           ftn_double_complex* beta, ftn_double_complex* y, ftn_int incy, 
                                           int stream_id)
{
    cublas_zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy, stream_id);
}

// Generic interface to zgemm
template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                           ftn_double_complex* alpha, ftn_double_complex const* A, ftn_int lda,
                                           ftn_double_complex const* B, ftn_int ldb, ftn_double_complex* beta, 
                                           ftn_double_complex* C, ftn_int ldc, int stream_id)
{
    cublas_zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream_id);
}

// Generic interface to dgemm
template<> 
void linalg<GPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                   ftn_double* alpha, ftn_double const* A, ftn_int lda,
                                   ftn_double const* B, ftn_int ldb, ftn_double* beta, 
                                   ftn_double* C, ftn_int ldc, int stream_id)
{
    cublas_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream_id);
}

// zgemm on default stream
template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                           ftn_double_complex* alpha, ftn_double_complex const* A, ftn_int lda,
                                           ftn_double_complex const* B, ftn_int ldb, ftn_double_complex* beta, 
                                           ftn_double_complex* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, -1);
}

// dgemm on default stream
template<> 
void linalg<GPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                   ftn_double* alpha, ftn_double const* A, ftn_int lda,
                                   ftn_double const* B, ftn_int ldb, ftn_double* beta, 
                                   ftn_double* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, -1);
}

template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                           ftn_double_complex const* A, ftn_int lda,
                                           ftn_double_complex const* B, ftn_int ldb, 
                                           ftn_double_complex* C, ftn_int ldc, int stream_id)
{
    cublas_zgemm(transa, transb, m, n, k, &zone, A, lda, B, ldb, &zzero, C, ldc, stream_id);
}

template<> 
void linalg<GPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                   ftn_double const* A, ftn_int lda,
                                   ftn_double const* B, ftn_int ldb, 
                                   ftn_double* C, ftn_int ldc, int stream_id)
{
    double alpha = 1;
    double beta = 0;
    WARNING("this may crash");
    cublas_dgemm(transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc, stream_id);
}

template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                           ftn_double_complex const* A, ftn_int lda,
                                           ftn_double_complex const* B, ftn_int ldb,
                                           ftn_double_complex* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, -1);
}

template<> 
void linalg<GPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                   ftn_double const* A, ftn_int lda,
                                   ftn_double const* B, ftn_int ldb,
                                   ftn_double* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, -1);
}

template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           matrix<ftn_double_complex> const& A, matrix<ftn_double_complex> const& B,
                                           matrix<ftn_double_complex>& C)
{
    gemm(transa, transb, m, n, k, A.at<GPU>(), A.ld(), B.at<GPU>(), B.ld(), C.at<GPU>(), C.ld());
}
#endif

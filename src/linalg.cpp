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

/** \file lin_alg.cpp
 *
 *  \brief Contains full specializations of linear algebra interface classes.
 */

#include "linalg.h"
#include "constants.h"
#ifdef _GPU_
#include "gpu_interface.h"
#endif

template<> 
void blas<CPU>::gemm<double>(int transa, int transb, int32_t m, int32_t n, int32_t k, double alpha, 
                             double* a, int32_t lda, double* b, int32_t ldb, double beta, double* c, 
                             int32_t ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<> 
void blas<CPU>::gemm<double>(int transa, int transb, int32_t m, int32_t n, int32_t k, double* a, int32_t lda, 
                             double* b, int32_t ldb, double* c, int32_t ldc)
{
    gemm(transa, transb, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

template<> 
void blas<CPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex alpha, double_complex* a, int32_t lda, double_complex* b, 
                                     int32_t ldb, double_complex beta, double_complex* c, int32_t ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<> 
void blas<CPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex* c, int32_t ldc)
{
    gemm(transa, transb, m, n, k, complex_one, a, lda, b, ldb, complex_zero, c, ldc);
}

template<> 
void blas<CPU>::hemm<double_complex>(int side, int uplo, int32_t m, int32_t n, double_complex alpha, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex beta, double_complex* c, int32_t ldc)
{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<>
void blas<CPU>::gemv<double_complex>(int trans, int32_t m, int32_t n, double_complex alpha, double_complex* a, 
                                     int32_t lda, double_complex* x, int32_t incx, double_complex beta, 
                                     double_complex* y, int32_t incy)
{
    const char *trans_c[] = {"N", "T", "C"};

    FORTRAN(zgemv)(trans_c[trans], &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
}

#ifdef _SCALAPACK_
int lin_alg<scalapack>::cyclic_block_size_ = -1;

template<> 
void blas<CPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
                                     dmatrix<double_complex>& a, int32_t ia, int32_t ja,
                                     dmatrix<double_complex>& b, int32_t ib, int32_t jb, double_complex beta, 
                                     dmatrix<double_complex>& c, int32_t ic, int32_t jc)
{
    const char *trans[] = {"N", "T", "C"};

    ia++; ja++;
    ib++; jb++;
    ic++; jc++;
    FORTRAN(pzgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a.ptr(), &ia, &ja, a.descriptor(), 
                    b.ptr(), &ib, &jb, b.descriptor(), &beta, c.ptr(), &ic, &jc, c.descriptor(), 1, 1);
}

template<> 
void blas<CPU>::gemm<double>(int transa, int transb, int32_t m, int32_t n, int32_t k, double alpha, 
                             dmatrix<double>& a, int32_t ia, int32_t ja,
                             dmatrix<double>& b, int32_t ib, int32_t jb, double beta, 
                             dmatrix<double>& c, int32_t ic, int32_t jc)
{
    const char *trans[] = {"N", "T", "C"};

    ia++; ja++;
    ib++; jb++;
    ic++; jc++;
    FORTRAN(pdgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a.ptr(), &ia, &ja, a.descriptor(), 
                    b.ptr(), &ib, &jb, b.descriptor(), &beta, c.ptr(), &ic, &jc, c.descriptor(), 1, 1);
}

template<> 
void blas<CPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
                                     dmatrix<double_complex>& a, dmatrix<double_complex>& b, double_complex beta, 
                                     dmatrix<double_complex>& c)
{
    gemm(transa, transb, m, n, k, alpha, a, 0, 0, b, 0, 0, beta, c, 0, 0);
}

template<> 
void blas<CPU>::gemm<double>(int transa, int transb, int32_t m, int32_t n, int32_t k, double alpha, 
                             dmatrix<double>& a, dmatrix<double>& b, double beta, 
                             dmatrix<double>& c)
{
    gemm(transa, transb, m, n, k, alpha, a, 0, 0, b, 0, 0, beta, c, 0, 0);
}
#else
template<> 
void blas<CPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
                                     dmatrix<double_complex>& a, int32_t ia, int32_t ja,
                                     dmatrix<double_complex>& b, int32_t ib, int32_t jb, double_complex beta, 
                                     dmatrix<double_complex>& c, int32_t ic, int32_t jc)
{
    gemm(transa, transb, m, n, k, alpha, &a(ia, ja), a.ld(), &b(ib, jb), b.ld(), beta, &c(ic, jc), c.ld());
}

template<> 
void blas<CPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
                                     dmatrix<double_complex>& a, dmatrix<double_complex>& b, double_complex beta, 
                                     dmatrix<double_complex>& c)
{
    gemm(transa, transb, m, n, k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta, c.ptr(), c.ld());
}
#endif


#ifdef _GPU_
double_complex blas<GPU>::zone = double_complex(1, 0);
double_complex blas<GPU>::zzero = double_complex(0, 0);

template<>
void blas<GPU>::gemv<double_complex>(int trans, int32_t m, int32_t n, double_complex* alpha, double_complex* a, int32_t lda,
                                     double_complex* x, int32_t incx, double_complex* beta, double_complex* y, int32_t incy, 
                                     int stream_id)
{
    cublas_zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, stream_id);
}

template<> 
void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* alpha, double_complex* a, int32_t lda, double_complex* b, 
                                     int32_t ldb, double_complex* beta, double_complex* c, int32_t ldc)
{
    cublas_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, -1);
}

template<> 
void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* alpha, double_complex* a, int32_t lda, double_complex* b, 
                                     int32_t ldb, double_complex* beta, double_complex* c, int32_t ldc, int stream_id)
{
    cublas_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream_id);
}

template<> 
void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex* c, int32_t ldc)
{
    cublas_zgemm(transa, transb, m, n, k, &zone, a, lda, b, ldb, &zzero, c, ldc, -1);
}

template<> 
void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex* c, int32_t ldc, int stream_id)
{
    cublas_zgemm(transa, transb, m, n, k, &zone, a, lda, b, ldb, &zzero, c, ldc, stream_id);
}
#endif


template<> 
int lin_alg<lapack>::gesv<double>(int32_t n, int32_t nrhs, double* a, int32_t lda, double* b, int32_t ldb)
{
    int32_t info;
    std::vector<int32_t> ipiv(n);
    FORTRAN(dgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);
    return info;
}

template<> 
int lin_alg<lapack>::gesv<double_complex>(int32_t n, int32_t nrhs, double_complex* a, int32_t lda, 
                                         double_complex* b, int32_t ldb)
{
    int32_t info;
    std::vector<int32_t> ipiv(n);
    FORTRAN(zgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);
    return info;
}

template<> 
int lin_alg<lapack>::gtsv<double>(int32_t n, int32_t nrhs, double* dl, double* d, double* du, double* b, int32_t ldb)
{
    int info;
    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;
}

template<> 
int lin_alg<lapack>::gtsv<double_complex>(int32_t n, int32_t nrhs, double_complex* dl, double_complex* d, double_complex* du, 
                                         double_complex* b, int32_t ldb)
{
    int32_t info;                   
    FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;               
}

template<> 
int lin_alg<lapack>::getrf<double>(int32_t m, int32_t n, double* a, int32_t lda, int32_t* ipiv)
{
    int32_t info;
    FORTRAN(dgetrf)(&m, &n, a, &lda, ipiv, &info);
    return info;
}
    
template<> 
int lin_alg<lapack>::getrf<double_complex>(int32_t m, int32_t n, double_complex* a, int32_t lda, int32_t* ipiv)
{
    int32_t info;
    FORTRAN(zgetrf)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

template<> 
int lin_alg<lapack>::getri<double>(int32_t n, double* a, int32_t lda, int32_t* ipiv, double* work, int32_t lwork)
{
    int32_t info;
    FORTRAN(dgetri)(&n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}

template<> 
int lin_alg<lapack>::getri<double_complex>(int32_t n, double_complex* a, int32_t lda, int32_t* ipiv, double_complex* work, 
                                          int32_t lwork)
{
    int32_t info;
    FORTRAN(zgetri)(&n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}

#if defined(_SCALAPACK_) && defined(_PILAENV_BLOCKSIZE_)
extern "C" int pilaenv_(int* ctxt, char* prec) 
{
    return _PILAENV_BLOCKSIZE_;
}
#endif



ftn_double_complex linalg_base::zone = double_complex(1, 0);
ftn_double_complex linalg_base::zzero = double_complex(0, 0);


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
                                   ftn_double* A, ftn_int lda, ftn_double* B, ftn_int ldb, ftn_double beta,
                                   ftn_double* C, ftn_int ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc,
                   (ftn_len)1, (ftn_len)1);
}

// C = alpha * op(A) * op(B) + beta * op(C), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, ftn_double_complex* A, ftn_int lda,
                                           ftn_double_complex* B, ftn_int ldb, ftn_double_complex beta,
                                           ftn_double_complex* C, ftn_int ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc,
                   (ftn_len)1, (ftn_len)1);
}

// C = op(A) * op(B), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, ftn_double* A, ftn_int lda, 
                                   ftn_double* B, ftn_int ldb, ftn_double* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, 1.0, A, lda, B, ldb, 0.0, C, ldc);
}

// C = op(A) * op(B), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex* A, ftn_int lda, ftn_double_complex* B, ftn_int ldb,
                                           ftn_double_complex* C, ftn_int ldc)
{
    gemm(transa, transb, m, n, k, ftn_double_complex(1, 0), A, lda, B, ldb, ftn_double_complex(0, 0), C, ldc);
}

// C = alpha * op(A) * op(B) + beta * op(C), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, ftn_double alpha,
                                   matrix<ftn_double>& A, matrix<ftn_double>& B, ftn_double beta, matrix<ftn_double>& C)
{
    gemm(transa, transb, m, n, k, alpha, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), beta, C.at<CPU>(), C.ld());
}

// C = alpha * op(A) * op(B) + beta * op(C), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           ftn_double_complex alpha, matrix<ftn_double_complex>& A,
                                           matrix<ftn_double_complex>& B, ftn_double_complex beta,
                                           matrix<ftn_double_complex>& C)
{
    gemm(transa, transb, m, n, k, alpha, A.at<CPU>(), A.ld(), B.at<CPU>(), B.ld(), beta, C.at<CPU>(), C.ld());
}

// C = op(A) * op(B), double
template<> 
void linalg<CPU>::gemm<ftn_double>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, matrix<ftn_double>& A, 
                                   matrix<ftn_double>& B, matrix<ftn_double>& C)
{
    gemm(transa, transb, m, n, k, 1.0, A, B, 0.0, C);
}

// C = op(A) * op(B), double_complex
template<> 
void linalg<CPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k,
                                           matrix<ftn_double_complex>& A, matrix<ftn_double_complex>& B,
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
    std::vector<ftn_int> ipiv(A.num_rows_local() + A.bs());
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

#ifdef _SCALAPACK_
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
#endif

#ifdef _GPU_

template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                           ftn_double_complex* alpha, ftn_double_complex* A, ftn_int lda,
                                           ftn_double_complex* B, ftn_int ldb, ftn_double_complex* beta, 
                                           ftn_double_complex* C, ftn_in ldc, int stream_id)
{
    cublas_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream_id);
}

template<> 
void linalg<GPU>::gemm<ftn_double_complex>(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                                           ftn_double_complex* alpha, ftn_double_complex* A, ftn_int lda,
                                           ftn_double_complex* B, ftn_int ldb, ftn_double_complex* beta, 
                                           ftn_double_complex* C, ftn_in ldc)
{
    gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, -1);
}


//template<> 
//void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
//                                     double_complex* alpha, double_complex* a, int32_t lda, double_complex* b, 
//                                     int32_t ldb, double_complex* beta, double_complex* c, int32_t ldc, int stream_id)
//{
//    cublas_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, stream_id);
//}
//
//template<> 
//void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
//                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
//                                     double_complex* c, int32_t ldc)
//{
//    cublas_zgemm(transa, transb, m, n, k, &zone, a, lda, b, ldb, &zzero, c, ldc, -1);
//}
//
//template<> 
//void blas<GPU>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
//                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
//                                     double_complex* c, int32_t ldc, int stream_id)
//{
//    cublas_zgemm(transa, transb, m, n, k, &zone, a, lda, b, ldb, &zzero, c, ldc, stream_id);
//}
#endif

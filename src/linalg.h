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

/** \file linalg.h
 *
 *  \brief Linear algebra interface.
 */

#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdint.h>
#include "typedefs.h"
#include "linalg_base.h"
#include "mdarray.h"
#include "dmatrix.h"

/// Linear algebra interface class.
template <processing_unit_t pu>
class linalg;

template<> 
class linalg<CPU>: public linalg_base
{
    public:

        /// General matrix times a vector.
        /** Perform one of the matrix-vector operations \n
         *  y = alpha * A * x + beta * y (trans = 0) \n
         *  y = alpha * A^{T} * x + beta * y (trans = 1) \n
         *  y = alpha * A^{+} * x + beta * y (trans = 2)
         */
        template<typename T>
        static void gemv(int trans, ftn_int m, ftn_int n, T alpha, T* A, ftn_int lda, T* x, ftn_int incx, 
                         T beta, T* y, ftn_int incy);
        
        /// Hermitian matrix times a general matrix or vice versa.
        /** Perform one of the matrix-matrix operations \n
         *  C = alpha * A * B + beta * C (side = 0) \n
         *  C = alpha * B * A + beta * C (side = 1), \n
         *  where A is a hermitian matrix with upper (uplo = 0) of lower (uplo = 1) triangular part defined.
         */
        template<typename T>
        static void hemm(int side, int uplo, ftn_int m, ftn_int n, T alpha, T* A, ftn_len lda, 
                         T* B, ftn_len ldb, T beta, T* C, ftn_len ldc);
        
        template<typename T>
        static void hemm(int side, int uplo, ftn_int m, ftn_int n, T alpha, matrix<T>& A, 
                         matrix<T>& B, T beta, matrix<T>& C);
        
        /// Compute C = alpha * op(A) * op(B) + beta * op(C) with raw pointers.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T alpha, T* A, ftn_int lda,
                         T* B, ftn_int ldb, T beta, T* C, ftn_int ldc);
        
        /// Compute C = op(A) * op(B) operation with raw pointers.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T* A, ftn_int lda, T* B, ftn_int ldb, 
                         T* C, ftn_int ldc);

        /// Compute C = alpha * op(A) * op(B) + beta * op(C) with matrix objects.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T alpha, matrix<T>& A, matrix<T>& B,
                         T beta, matrix<T>& C);

        /// Compute C = op(A) * op(B) operation with matrix objects.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, matrix<T>& A, matrix<T>& B,
                         matrix<T>& C);
                         
        /// Compute C = alpha * op(A) * op(B) + beta * op(C), generic interface
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T alpha, 
                         dmatrix<T>& A, ftn_int ia, ftn_int ja, dmatrix<T>& B, ftn_int ib, ftn_int jb, T beta, 
                         dmatrix<T>& C, ftn_int ic, ftn_int jc);

        /// Compute C = alpha * op(A) * op(B) + beta * op(C), simple interface - matrices start from (0, 0) corner.
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, 
                         T alpha, dmatrix<T>& A, dmatrix<T>& B, T beta, dmatrix<T>& C);

        /// Compute the solution to system of linear equations A * X = B for GT matrices.
        template <typename T> 
        static ftn_int gtsv(ftn_int n, ftn_int nrhs, T* dl, T* d, T* du, T* b, ftn_int ldb);
        
        /// Compute the solution to system of linear equations A * X = B for GE matrices.
        template <typename T> 
        static ftn_int gesv(ftn_int n, ftn_int nrhs, T* A, ftn_int lda, T* B, ftn_int ldb);

        /// LU factorization
        template <typename T>
        static ftn_int getrf(ftn_int m, ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);
        
        /// U*D*U^H factorization of hermitian matrix
        template <typename T>
        static ftn_int hetrf(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);
        
        template <typename T>
        static ftn_int getri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);
        
        template <typename T>
        static ftn_int hetri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);
        
        /// Invert a general matrix.
        template <typename T>
        static void geinv(ftn_int n, matrix<T>& A);

        /// Invert a general distributed matrix.
        template <typename T>
        static void geinv(ftn_int n, dmatrix<T>& A);

        template <typename T>
        static ftn_int sytrf(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);

        template <typename T>
        static ftn_int sytri(ftn_int n, T* A, ftn_int lda, ftn_int* ipiv);

        template <typename T>
        static void syinv(ftn_int n, matrix<T>& A);

        /// Invert a hermitian matrix.
        template <typename T>
        static void heinv(ftn_int n, matrix<T>& A);

        template <typename T>
        static ftn_int getrf(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, ftn_int* ipiv);

        template <typename T>
        static ftn_int getri(ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, ftn_int* ipiv);

        /// Conjugate transponse of the sub-matrix.
        /** \param [in] m Number of rows of the target sub-matrix.
         *  \param [in] n Number of columns of the target sub-matrix.
         */
        template <typename T>
        static void tranc(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, dmatrix<T>& C, ftn_int ic, ftn_int jc);
        
        template <typename T>
        static void tranu(ftn_int m, ftn_int n, dmatrix<T>& A, ftn_int ia, ftn_int ja, dmatrix<T>& C, ftn_int ic, ftn_int jc);
};

#ifdef _GPU_
template<> 
class linalg<GPU>: public linalg_base
{
    public:

        template<typename T>
        static void gemv(int trans, ftn_int m, ftn_int n, T* alpha, T* A, ftn_int lda, T* x, ftn_int incx, 
                         T* beta, T* y, ftn_int incy, int stream_id);

        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T* alpha, T* A, ftn_int lda,
                         T* B, ftn_int ldb, T* beta, T* C, ftn_int ldc, int stream_id);

        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T* alpha, T* A, ftn_int lda,
                         T* B, ftn_int ldb, T* beta, T* C, ftn_int ldc);
        
        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T* A, ftn_int lda,
                         T* B, ftn_int ldb, T* C, ftn_int ldc, int stream_id);

        template <typename T>
        static void gemm(int transa, int transb, ftn_int m, ftn_int n, ftn_int k, T* A, ftn_int lda,
                         T* B, ftn_int ldb, T* C, ftn_int ldc);
};
#endif

#endif // __LINALG_H__


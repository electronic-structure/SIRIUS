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

/** \file blas.h
 *   
 *  \brief Contains definition of templated blas class.
 */

#ifndef __BLAS_H__
#define __BLAS_H__

template<processing_unit_t> 
class blas;

// CPU
template<> 
class blas<CPU>
{
    public:

        //template <typename T>
        //static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, T* a, int32_t lda, 
        //                 T* b, int32_t ldb, T beta, T* c, int32_t ldc);
        
        //== template <typename T>
        //== static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
        //==                  T* b, int32_t ldb, T* c, int32_t ldc);
                         
        //== template<typename T>
        //== static void hemm(int side, int uplo, int32_t m, int32_t n, T alpha, T* a, int32_t lda, 
        //==                  T* b, int32_t ldb, T beta, T* c, int32_t ldc);

        //== template<typename T>
        //== static void gemv(int trans, int32_t m, int32_t n, T alpha, T* a, int32_t lda, T* x, int32_t incx, 
        //==                  T beta, T* y, int32_t incy);

        //== /// generic interface to p?gemm
        //== template <typename T>
        //== static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, 
        //==                  dmatrix<T>& a, int32_t ia, int32_t ja, dmatrix<T>& b, int32_t ib, int32_t jb, T beta, 
        //==                  dmatrix<T>& c, int32_t ic, int32_t jc);

        //== /// simple interface to p?gemm: all matrices start form (0, 0) corner
        //== template <typename T>
        //== static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
        //==                  T alpha, dmatrix<T>& a, dmatrix<T>& b, T beta, dmatrix<T>& c);
};

#ifdef _GPU_
template<> 
class blas<GPU>
{
    private:
        static double_complex zone;
        static double_complex zzero;

    public:

        template<typename T>
        static void gemv(int trans, int32_t m, int32_t n, T* alpha, T* a, int32_t lda, T* x, int32_t incx, 
                         T* beta, T* y, int32_t incy, int stream_id);

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* beta, T* c, int32_t ldc);

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* beta, T* c, int32_t ldc, int stream_id);
        
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* c, int32_t ldc);
        
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* c, int32_t ldc, int stream_id);
};
#endif

#endif // __BLAS_H__

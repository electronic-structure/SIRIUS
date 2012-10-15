#ifndef __LINALG_H__
#define __LINALG_H__

#include <stdint.h>
#include <iostream>
#include <complex>
#include "config.h"
#include "linalg_cpu.h"
#include "linalg_gpu.h"

//
// gemm
//
template<implementation impl, typename T> 
void gemm(int transa, int transb, int4 m, int4 n, int4 k, T alpha, T* a, int4 lda, T* b, int4 ldb, T beta, 
          T* c, int4 ldc);

template<> void gemm<cpu,real8>(int transa, int transb, int4 m, int4 n, int4 k, real8 alpha, real8* a, 
                                int4 lda, real8* b, int4 ldb, real8 beta, real8* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, 
                   c, &ldc, (int4)1, (int4)1);
}

template<> void gemm<cpu,complex16>(int transa, int transb, int4 m, int4 n, int4 k, complex16 alpha, complex16* a, 
                                    int4 lda, complex16* b, int4 ldb, complex16 beta, complex16* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, 
                   c, &ldc, (int4)1, (int4)1);
}

//
// hemm
//
template<implementation impl, typename T> 
void hemm(int side, int uplo, int4 m, int4 n, T alpha, T* a, int4 lda, T* b, int4 ldb, T beta, T* c, int4 ldc);

template<> void hemm<cpu,complex16>(int side, int uplo, int4 m, int4 n, complex16 alpha, complex16* a, int4 lda, 
                                    complex16* b, int4 ldb, complex16 beta, complex16* c, int4 ldc)

{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, 
                   c, &ldc, (int4)1, (int4)1);
}

//
// heev
//
template<implementation impl, typename T> 
int heev(int4 n, T* a, int4 lda, typename data_type_wrapper<T>::real_type_* eval);

template<> int heev<cpu,complex16>(int4 n, complex16* a, int4 lda, double* eval)
{
   int ispec = 1;
   int n1 = -1;
   int nb = FORTRAN(ilaenv)(&ispec, "ZHETRD", "U",  &n, &n1, &n1, &n1, (int4)6, (int4)1);
   int lwork = (nb + 1) * n;
   std::vector<double> work(lwork * 2);
   std::vector<double> rwork(3* n + 2);
   int info;
   FORTRAN(zheev)("V", "U", &n, a, &lda, eval, &work[0], &lwork, &rwork[0], &info, (int4)1, (int4)1);
   
   return info;
}

//
// hegvx
//
template<implementation impl, typename T>
int hegvx(int4 n, int4 nv, real8 abstol, T* a, T* b, typename data_type_wrapper<T>::real_type_* eval, T* z, int4 ldz);

template<> int hegvx<cpu,complex16>(int4 n, int4 nv, real8 abstol, complex16* a, complex16* b,
                                    real8* eval, complex16* z, int4 ldz)
{
   int4 ispec = 1;
   int4 n1 = -1;
   int4 nb = FORTRAN(ilaenv)(&ispec, "ZHETRD", "U",  &n, &n1, &n1, &n1, (int4)6, (int4)1);
   int4 lwork = (nb + 1) * n;
   std::vector<int4> iwork(5 * n);
   std::vector<int4> ifail(n);
   std::vector<real8> w(n);
   std::vector<real8> rwork(7 * n);
   std::vector< complex16 > work(lwork);
   n1 = 1;
   real8 vl = 0.0;
   real8 vu = 0.0;
   int4 m;
   int4 info;
   
   FORTRAN(zhegvx)(&n1, "V", "I", "U", &n, a, &n, b, &n, &vl, &vu, &n1, 
                   &nv, &abstol, &m, &w[0], z, &ldz, &work[0], &lwork, &rwork[0], 
                   &iwork[0], &ifail[0], &info, (int4)1, (int4)1, (int4)1);

   memcpy(eval, &w[0], nv * sizeof(real8));

   return info;
}


/*
template<implementation impl> 
void zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, complex16 alpha, 
           complex16 *a, int32_t lda, complex16 *b, int32_t ldb, complex16 beta, 
           complex16 *c, int32_t ldc) 
{
    if (impl == cpu)
    {
        const char *trans[] = {"N", "T", "C"};
        FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, 
            c, &ldc, (int32_t)1, (int32_t)1);
    }
    
    if (impl == gpu)
    {    
        gpu_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

template<implementation impl> 
void zhemm(int side, int uplo, int32_t m, int32_t n, complex16 alpha,
           complex16 *a, int32_t lda, complex16 *b, int32_t ldb, complex16 beta,
           complex16 *c, int32_t ldc)
{
    if (impl == cpu) 
    {
        const char *sidestr[] = {"L", "R"};
        const char *uplostr[] = {"U", "L"};
        FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, 
            &ldc, (int32_t)1, (int32_t)1);
    }
}    

template<implementation impl> 
void zhegv(int32_t n, int32_t nv, double abstol, complex16 *a, complex16 *b,
           double *eval, complex16 *z, int32_t ldz)
{
    if (impl == cpu) 
    {
        int ispec = 1;
        int n1 = -1;
        int nb = FORTRAN(ilaenv)(&ispec, "ZHETRD", "U",  &n, &n1, &n1, &n1, (int32_t)6, (int32_t)1);
        int lwork = (nb + 1) * n;
        std::vector<int> iwork(5 * n);
        std::vector<int> ifail(n);
        std::vector<double> w(n);
        std::vector<double> rwork(7 * n);
        std::vector< std::complex<double> > work(lwork);
        n1 = 1;
        double vl = 0.0;
        double vu = 0.0;
        int m;
        int info;
        FORTRAN(zhegvx)(&n1, "V", "I", "U", &n, a, &n, b, &n, &vl, &vu, &n1, 
            &nv, &abstol, &m, &w[0], z, &ldz, &work[0], &lwork, &rwork[0], 
            &iwork[0], &ifail[0], &info, (int32_t)1, (int32_t)1, (int32_t)1);
        if (info)
        {
           std::cout << "zhegvx diagonalization failed" << std::endl
                     << "info = " << info << std::endl
                     << "matrix size = " << n << std::endl;
           exit(0);
        }
        memcpy(eval, &w[0], nv * sizeof(double));
    }
    if (impl == gpu)
    {
        gpu_zhegvx(n, nv, abstol, a, b, eval, z, ldz);
    }
}

template<implementation impl> 
void zheev(int32_t n, complex16 *a, int32_t lda, double *eval)
{
    if (impl == cpu)
    {
        int ispec = 1;
        int n1 = -1;
        int nb = FORTRAN(ilaenv)(&ispec, "ZHETRD", "U",  &n, &n1, &n1, &n1, (int32_t)6, (int32_t)1);
        int lwork = (nb + 1) * n;
        std::vector<double> work(lwork * 2);
        std::vector<double> rwork(3* n + 2);
        int info;
        FORTRAN(zheev)("V", "U", &n, a, &lda, eval, &work[0], &lwork, &rwork[0], &info, (int32_t)1, (int32_t)1);
        if (info)
        {
           std::cout << "zheev diagonalization failed" << std::endl
                     << "info = " << info << std::endl
                     << "matrix size = " << n << std::endl;
           exit(0);
        }
    }
}

template<implementation impl> 
void zgemv(int transa, int32_t m, int32_t n, complex16 alpha, complex16 *a, int32_t lda, complex16 *x,
           int32_t incx, complex16 beta, complex16 *y, int32_t incy)
{
    if (impl == cpu) 
    {
        const char *trans[] = {"N", "T", "C"};
        FORTRAN(zgemv)(trans[transa], &m, &n, &alpha, a, &lda, x, &incx, 
            &beta, y, &incy, (int32_t)1);
    }
}

*/
/*void dgemm(int transa, int transb, int4 m, int4 n, int4 k, real8 alpha, real8* a, int4 lda, real8* b, 
           int4 ldb, real8 beta, real8* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, (int4)1);
}*/

#endif // __LINALG_H__


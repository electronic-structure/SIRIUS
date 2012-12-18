#ifndef __LINALG_H__
#define __LINALG_H__

/** \file linalg.h

    \brief Linear algebra interface

*/

#include <stdint.h>
#include <iostream>
#include <complex>
#include "config.h"
#include "linalg_cpu.h"
#include "linalg_gpu.h"

/// Matrix matrix multimplication
template<implementation impl, typename T> 
void gemm(int transa, int transb, int4 m, int4 n, int4 k, T alpha, T* a, int4 lda, T* b, int4 ldb, T beta, T* c, 
          int4 ldc);

/// Specialization of matrix matrix multiplication for real8 type on a CPU
template<> void gemm<cpu, real8>(int transa, int transb, int4 m, int4 n, int4 k, real8 alpha, real8* a, int4 lda, 
                                 real8* b, int4 ldb, real8 beta, real8* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, 
                   (int4)1);
}

/// Specialization of matrix matrix multiplication for complex16 type on a CPU
template<> void gemm<cpu, complex16>(int transa, int transb, int4 m, int4 n, int4 k, complex16 alpha, complex16* a, 
                                     int4 lda, complex16* b, int4 ldb, complex16 beta, complex16* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, 
                   (int4)1);
}

/// Hermitian matrix matrix multimplication
template<implementation impl, typename T> 
void hemm(int side, int uplo, int4 m, int4 n, T alpha, T* a, int4 lda, T* b, int4 ldb, T beta, T* c, int4 ldc);

template<> void hemm<cpu, complex16>(int side, int uplo, int4 m, int4 n, complex16 alpha, complex16* a, int4 lda, 
                                     complex16* b, int4 ldb, complex16 beta, complex16* c, int4 ldc)

{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, (int4)1);
}

template<typename T> 
int gtsv(int n, int nrhs, T* dl, T* d, T* du, T* b, int ldb);

template<> int gtsv<double>(int n, int nrhs, double *dl, double *d, double *du, double *b, int ldb)
{
    int info;

    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;
}

template<> int gtsv<complex16>(int n, int nrhs, complex16* dl, complex16* d, complex16* du, complex16* b, int ldb)
{
    int4 info;                   

    FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;                
}

template <typename T> int gesv(int4 n, int4 nrhs, T* a, int4 lda, T* b, int4 ldb);

template<> int gesv<real8>(int4 n, int4 nrhs, real8* a, int4 lda, real8* b, int4 ldb)
{
    int4 info;
    std::vector<int4> ipiv(n);

    FORTRAN(dgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

template<> int gesv<complex16>(int4 n, int4 nrhs, complex16* a, int4 lda, complex16* b, int4 ldb)
{
    int4 info;
    std::vector<int4> ipiv(n);

    FORTRAN(zgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}


template <eigen_value_solver_t solver>
void descinit(int4* desc, int4 m, int4 n, int4 mb, int4 nb, int4 irsrc, int4 icsrc, int4 ictxt, int4 lld)
{
    if (solver == scalapack)
    {
        int4 info;

        FORTRAN(descinit)(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);

        if (info) error(__FILE__, __LINE__, "error in descinit");
    }
}

template <eigen_value_solver_t solver>
int generalized_eigenproblem(int4 matrix_size, int num_ranks_row, int num_ranks_col, int blacs_context, int4 nv, 
                             real8 abstol, complex16* a, int4 lda, complex16* b, int4 ldb, real8* eval, complex16* z, 
                             int4 ldz)
{
    if (solver == lapack)
    {
        int4 ispec = 1;
        int4 n1 = -1;
        int4 nb = FORTRAN(ilaenv)(&ispec, "ZHETRD", "U",  &matrix_size, &n1, &n1, &n1, (int4)6, (int4)1);
        int4 lwork = (nb + 1) * matrix_size;
        std::vector<int4> iwork(5 * matrix_size);
        std::vector<int4> ifail(matrix_size);
        std::vector<real8> w(matrix_size);
        std::vector<real8> rwork(7 * matrix_size);
        std::vector<complex16> work(lwork);
        n1 = 1;
        real8 vl = 0.0;
        real8 vu = 0.0;
        int4 m;
        int4 info;
        
        FORTRAN(zhegvx)(&n1, "V", "I", "U", &matrix_size, a, &lda, b, &ldb, &vl, &vu, &n1, 
                        &nv, &abstol, &m, &w[0], z, &ldz, &work[0], &lwork, &rwork[0], 
                        &iwork[0], &ifail[0], &info, (int4)1, (int4)1, (int4)1);

        if (m != nv) error(__FILE__, __LINE__, "Not all eigen-values are found.", fatal_err);

        memcpy(eval, &w[0], nv * sizeof(real8));

        return info;
    }

    if (solver == scalapack)
    {
        int desca[9];
        descinit<solver>(desca, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, lda);

        int descb[9];
        descinit<solver>(descb, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, ldb); 

        int descz[9];
        descinit<solver>(descz, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, ldz); 

        int4 lwork = -1;
        complex16 z1;
        int4 lrwork;
        real8 d1;
        int4 liwork;
        int4 i1;

        int4 ione = 1;
        
        real8 orfac = -1.0;

        int4 m;
        int4 nz;
        std::vector<int4> ifail(matrix_size);
        std::vector<int4> iclustr(2 * num_ranks_row * num_ranks_col);
        std::vector<real8> gap(num_ranks_row * num_ranks_col);
        std::vector<real8> w(matrix_size);
        int4 info;

        // make q work size query (lwork = -1)
        FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, a, &ione, &ione, desca, b, &ione, &ione, descb, &d1, &d1, 
                         &ione, &nv, &abstol, &m, &nz, &w[0], &orfac, z, &ione, &ione, descz, &z1, &lwork, &d1, &lrwork, 
                         &i1, &liwork, &ifail[0], &iclustr[0], &gap[0], &info, (int4)1, (int4)1, (int4)1); 

        lwork = ((int)real(z1) + 100) * 2;
        lrwork = ((int)d1 + 100) * 2;
        liwork = (i1 + 100) * 2;

        std::vector<complex16> work(lwork);
        std::vector<real8> rwork(lrwork);
        std::vector<int4> iwork(liwork);

        FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, a, &ione, &ione, desca, b, &ione, &ione, descb, &d1, &d1, 
                         &ione, &nv, &abstol, &m, &nz, &w[0], &orfac, z, &ione, &ione, descz, &work[0], &lwork, &rwork[0], 
                         &lrwork, &iwork[0], &liwork, &ifail[0], &iclustr[0], &gap[0], &info, (int4)1, (int4)1, (int4)1); 

        if (info)
        {
            std::stringstream s;
            s << "pzhegvx returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        if ((m != nv) || (nz != nv))
            error(__FILE__, __LINE__, "Not all eigen-vectors or eigen-values are found.", fatal_err);

        memcpy(eval, &w[0], nv * sizeof(real8));

        return info;
    }

    return 0;
}

template <eigen_value_solver_t solver>
int standard_eigenproblem(int4 matrix_size, int4 num_ranks_row, int4 num_ranks_col, int blacs_context, 
                          complex16* a, int4 lda, real8* eval, complex16* z, int4 ldz)
{                
    if (solver == scalapack)
    {
        int desca[9];
        descinit<solver>(desca, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, lda);
        
        int descz[9];
        descinit<solver>(descz, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, ldz);
        
        int4 izero = 0;
        int4 nb = scalapack_nb;
        int np0 = FORTRAN(numroc)(&matrix_size, &nb, &izero, &izero, &num_ranks_row);                
        int mq0 = FORTRAN(numroc)(&matrix_size, &nb, &izero, &izero, &num_ranks_col);

        int lwork = matrix_size + (np0 + mq0 + scalapack_nb) * scalapack_nb;
        std::vector<complex16> work(lwork);

        int lrwork = 1 + 9 * matrix_size + 3 * np0 * mq0;
        std::vector<real8> rwork(lrwork);

        int liwork = 7 * matrix_size + 8 * num_ranks_col + 2;
        std::vector<int4> iwork(liwork);

        int4 info;

        int4 ione = 1;
        FORTRAN(pzheevd)("V", "U", &matrix_size, a, &ione, &ione, desca, eval, z, &ione, &ione, descz, &work[0], 
                         &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info, (int4)1, (int4)1);

        if (info)
        {
            std::stringstream s;
            s << "pzheevd returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        return info;
    }
    
    if (solver == lapack)
    {
        //*int ispec = 1;
        //*int n1 = -1;
        //*int nb = FORTRAN(ilaenv)(&ispec, "ZHETRD", "U",  &matrix_size, &n1, &n1, &n1, (int4)6, (int4)1);
        //*int lwork = (nb + 1) * matrix_size;
        //*std::vector<complex16> work(lwork);
        //*std::vector<double> rwork(3 * matrix_size + 2);
        //*int info;
        //*FORTRAN(zheev)("V", "U", &matrix_size, a, &lda, eval, &work[0], &lwork, &rwork[0], &info, (int4)1, (int4)1);

        int4 lwork = 2 * matrix_size + matrix_size * matrix_size;
        int4 lrwork = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
        int4 liwork = 3 + 5 * matrix_size;

        std::vector<complex16> work(lwork);
        std::vector<real8> rwork(lrwork);
        std::vector<int4> iwork(liwork);

        int4 info;

        FORTRAN(zheevd)("V", "U", &matrix_size, a, &lda, eval, &work[0], &lwork, &rwork[0], &lrwork, &iwork[0],
                        &liwork, &info, (int4)1, (int4)1);
        
        for (int i = 0; i < matrix_size; i++)
            memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(complex16));
        
        if (info)
        {
            std::stringstream s;
            s << "zheev returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        return info;
    }
    
    return 0;
}

#endif // __LINALG_H__


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

template<processing_unit_t> struct linalg;

template<> struct linalg<cpu>
{
    template <typename T>
    static inline void gemm(int transa, int transb, int4 m, int4 n, int4 k, T alpha, T* a, int4 lda, T* b, int4 ldb, 
                            T beta, T* c, int4 ldc);
    
    template <typename T>
    static inline void gemm(int transa, int transb, int4 m, int4 n, int4 k, T* a, int4 lda, T* b, int4 ldb, T* c, 
                            int4 ldc);
};

template<> inline void linalg<cpu>::gemm<real8>(int transa, int transb, int4 m, int4 n, int4 k, real8 alpha, 
                                                real8* a, int4 lda, real8* b, int4 ldb, real8 beta, real8* c, 
                                                int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, 
                   (int4)1);
}

template<> inline void linalg<cpu>::gemm<real8>(int transa, int transb, int4 m, int4 n, int4 k, real8* a, int4 lda, 
                                                real8* b, int4 ldb, real8* c, int4 ldc)
{
    gemm(transa, transb, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

template<> inline void linalg<cpu>::gemm<complex16>(int transa, int transb, int4 m, int4 n, int4 k, complex16 alpha, 
                                                    complex16* a, int4 lda, complex16* b, int4 ldb, complex16 beta, 
                                                    complex16* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, 
                   (int4)1);
}

template<> inline void linalg<cpu>::gemm<complex16>(int transa, int transb, int4 m, int4 n, int4 k, complex16* a, 
                                                    int4 lda, complex16* b, int4 ldb, complex16* c, int4 ldc)
{
    gemm(transa, transb, m, n, k, complex16(1, 0), a, lda, b, ldb, complex16(0, 0), c, ldc);
}

/// Matrix matrix multimplication
template<processing_unit_t device, typename T> 
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
template<processing_unit_t device, typename T> 
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




template<eigen_value_solver_t> struct EigenValueSolver;

template<> struct EigenValueSolver<lapack>
{

};















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
int pjlaenv(int4 ictxt, int4 ispec, const std::string& name, const std::string& opts, int4 n1, int4 n2, int4 n3, 
            int4 n4)
{
    if (solver == scalapack)
    {
        return FORTRAN(pjlaenv)(&ictxt, &ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int4)name.length(),
                                (int4)opts.length());
    }
    
    return 0;
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
        int4 desca[9];
        descinit<solver>(desca, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, lda);

        int4 descb[9];
        descinit<solver>(descb, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, ldb); 

        int4 descz[9];
        descinit<solver>(descz, matrix_size, matrix_size, scalapack_nb, scalapack_nb, 0, 0, blacs_context, ldz); 

        int4 izero = 0;
        int4 ione = 1;

        int4 nb = scalapack_nb;
        int4 np0 = FORTRAN(numroc)(&matrix_size, &nb, &izero, &izero, &num_ranks_row);                
        int4 nq0 = FORTRAN(numroc)(&matrix_size, &nb, &izero, &izero, &num_ranks_col);                
        int4 mq0 = FORTRAN(numroc)(&matrix_size, &nb, &izero, &izero, &num_ranks_col);
        
        int4 ictxt = desca[1];
        int4 anb = pjlaenv<solver>(ictxt, 3, "PZHETTRD", "L", 0, 0, 0, 0);
        int4 sqnpc = (int4)pow(real8(num_ranks_row * num_ranks_col), 0.5);
        int4 nps = std::max(FORTRAN(numroc)(&matrix_size, &ione, &izero, &izero, &sqnpc), 2 * anb);

        int4 lwork = matrix_size + (np0 + mq0 + nb) * nb;
        lwork = std::max(lwork, matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
        lwork = std::max(lwork, 2 * np0 * nb + nq0 * nb + nb * nb);

        int4 neig = 10;
        int4 max3 = std::max(neig, std::max(nb, 2));
        mq0 = FORTRAN(numroc)(&max3, &nb, &izero, &izero, &num_ranks_col);
        int4 np = num_ranks_row * num_ranks_col;
        int4 lrwork = 4 * matrix_size + std::max(5 * matrix_size, np0 * mq0) + 
                      FORTRAN(iceil)(&neig, &np) * matrix_size + neig * matrix_size;

        int4 nnp = std::max(matrix_size, std::max(np + 1, 4));
        int4 liwork = 6 * nnp;
        
        real8 orfac = -1.0;

        int4 m;
        int4 nz;
        real8 d1;
        std::vector<int4> ifail(matrix_size);
        std::vector<int4> iclustr(2 * np);
        std::vector<real8> gap(np);
        std::vector<real8> w(matrix_size);
        int4 info;

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


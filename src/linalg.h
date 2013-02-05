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

template<processing_unit_t> struct blas;

template<> struct blas<cpu>
{
    template <typename T>
    static inline void gemm(int transa, int transb, int4 m, int4 n, int4 k, T alpha, T* a, int4 lda, T* b, int4 ldb, 
                            T beta, T* c, int4 ldc);
    
    template <typename T>
    static inline void gemm(int transa, int transb, int4 m, int4 n, int4 k, T* a, int4 lda, T* b, int4 ldb, T* c, 
                            int4 ldc);
    template<typename T>
    static inline void hemm(int side, int uplo, int4 m, int4 n, T alpha, T* a, int4 lda, T* b, int4 ldb, T beta, T* c, 
                            int4 ldc);
};

template<> inline void blas<cpu>::gemm<real8>(int transa, int transb, int4 m, int4 n, int4 k, real8 alpha, 
                                              real8* a, int4 lda, real8* b, int4 ldb, real8 beta, real8* c, 
                                              int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, 
                   (int4)1);
}

template<> inline void blas<cpu>::gemm<real8>(int transa, int transb, int4 m, int4 n, int4 k, real8* a, int4 lda, 
                                              real8* b, int4 ldb, real8* c, int4 ldc)
{
    gemm(transa, transb, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

template<> inline void blas<cpu>::gemm<complex16>(int transa, int transb, int4 m, int4 n, int4 k, complex16 alpha, 
                                                  complex16* a, int4 lda, complex16* b, int4 ldb, complex16 beta, 
                                                  complex16* c, int4 ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, 
                   (int4)1);
}

template<> inline void blas<cpu>::gemm<complex16>(int transa, int transb, int4 m, int4 n, int4 k, complex16* a, 
                                                    int4 lda, complex16* b, int4 ldb, complex16* c, int4 ldc)
{
    gemm(transa, transb, m, n, k, complex16(1, 0), a, lda, b, ldb, complex16(0, 0), c, ldc);
}

template<> inline void blas<cpu>::hemm<complex16>(int side, int uplo, int4 m, int4 n, complex16 alpha, complex16* a, 
                                                  int4 lda, complex16* b, int4 ldb, complex16 beta, complex16* c, 
                                                  int4 ldc)
{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int4)1, (int4)1);
}


template<linalg_t> struct linalg;

template<> struct linalg<lapack>
{
    static int4 ilaenv(int4 ispec, const std::string& name, const std::string& opts, int4 n1, int4 n2, int4 n3, 
                       int4 n4)
    {
        return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int4)name.length(), 
                               (int4)opts.length());
    }
    
    template <typename T> 
    static int gesv(int4 n, int4 nrhs, T* a, int4 lda, T* b, int4 ldb);

    template <typename T> 
    static int gtsv(int4 n, int4 nrhs, T* dl, T* d, T* du, T* b, int4 ldb);
};

template<> int linalg<lapack>::gesv<real8>(int4 n, int4 nrhs, real8* a, int4 lda, real8* b, int4 ldb)
{
    int4 info;
    std::vector<int4> ipiv(n);

    FORTRAN(dgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

template<> int linalg<lapack>::gesv<complex16>(int4 n, int4 nrhs, complex16* a, int4 lda, complex16* b, int4 ldb)
{
    int4 info;
    std::vector<int4> ipiv(n);

    FORTRAN(zgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

template<> int linalg<lapack>::gtsv<real8>(int4 n, int4 nrhs, real8* dl, real8* d, real8* du, real8* b, int4 ldb)
{
    int info;

    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;
}

template<> int linalg<lapack>::gtsv<complex16>(int4 n, int4 nrhs, complex16* dl, complex16* d, complex16* du, 
                                               complex16* b, int4 ldb)
{
    int4 info;                   

    FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;               
}

template<> struct linalg<scalapack>
{
    /// Create BLACS context
    static int create_blacs_context(MPI_Comm comm)
    {
        return Csys2blacs_handle(comm);
    }

    /// create grid of MPI ranks
    static void gridmap(int* blacs_context, int* map, int ld, int nrow, int ncol)
    {
        Cblacs_gridmap(blacs_context, map, ld, nrow, ncol);
    }

    static void gridinfo(int blacs_context, int* nrow, int* ncol, int* irow, int* icol)
    {
        Cblacs_gridinfo(blacs_context, nrow, ncol, irow, icol);
    }

    static void free_blacs_context(int blacs_context)
    {
        Cfree_blacs_system_handle(blacs_context);
    }

    static void descinit(int4* desc, int4 m, int4 n, int4 mb, int4 nb, int4 irsrc, int4 icsrc, int4 ictxt, int4 lld)
    {
        int4 info;
    
        FORTRAN(descinit)(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);
    
        if (info) error(__FILE__, __LINE__, "error in descinit");
    }

    static int pjlaenv(int4 ictxt, int4 ispec, const std::string& name, const std::string& opts, int4 n1, int4 n2, 
                       int4 n3, int4 n4)
    {
        return FORTRAN(pjlaenv)(&ictxt, &ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int4)name.length(),
                                (int4)opts.length());
    }

    static int4 numroc(int4 n, int4 nb, int4 iproc, int4 isrcproc, int4 nprocs)
    {
        return FORTRAN(numroc)(&n, &nb, &iproc, &isrcproc, &nprocs); 
    }

    static int4 indxl2g(int4 indxloc, int4 nb, int4 iproc, int4 isrcproc, int4 nprocs)
    {
        return FORTRAN(indxl2g)(&indxloc, &nb, &iproc, &isrcproc, &nprocs);
    }

    static int4 iceil(int4 inum, int4 idenom)
    {
        return FORTRAN(iceil)(&inum, &idenom);
    }

    static void pztranc(int4 m, int4 n, complex16 alpha, complex16* a, int4 ia, int4 ja, int4* desca, 
                        complex16 beta, complex16* c, int4 ic, int4 jc, int4* descc)
    {
        FORTRAN(pztranc)(&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }
};

/// Declaration of an eigenproblem class to be specialized for a particular solver
template<linalg_t> struct eigenproblem;

/// Lapack eigenvalue specialization
template<> struct eigenproblem<lapack>
{
    static std::vector<int4> get_work_sizes(int id, int4 matrix_size)
    {
        std::vector<int4> work_sizes(3);
        
        switch (id)
        {
            case 0: // zheevd
            {
                work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
                work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
                work_sizes[2] = 3 + 5 * matrix_size;
                break;
            }
            case 1: // zhegvx
            {
                int4 nb = linalg<lapack>::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
                work_sizes[0] = (nb + 1) * matrix_size; // lwork
                work_sizes[1] = 7 * matrix_size; // lrwork
                work_sizes[2] = 5 * matrix_size; // liwork
                break;
            }
            default:
                error(__FILE__, __LINE__, "wrong eigen value solver id");
        }

        return work_sizes;
    }
    
    static int standard(int4 matrix_size, complex16* a, int4 lda, real8* eval, complex16* z, int4 ldz)
    {
        std::vector<int4> work_sizes = get_work_sizes(0, matrix_size);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int4> iwork(work_sizes[2]);
        int4 info;

        FORTRAN(zheevd)("V", "U", &matrix_size, a, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                        &iwork[0], &work_sizes[2], &info, (int4)1, (int4)1);
        
        for (int i = 0; i < matrix_size; i++)
            memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(complex16));
        
        if (info)
        {
            std::stringstream s;
            s << "zheevd returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        return info;

    }

    static int generalized(int4 matrix_size, int4 nv, real8 abstol, complex16* a, int4 lda, complex16* b, int4 ldb, 
                           real8* eval, complex16* z, int4 ldz)
    {
        assert(nv <= matrix_size);

        std::vector<int4> work_sizes = get_work_sizes(1, matrix_size);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int4> iwork(work_sizes[2]);
        std::vector<int4> ifail(matrix_size);
        std::vector<real8> w(matrix_size);
        real8 vl = 0.0;
        real8 vu = 0.0;
        int4 m;
        int4 info;
       
        int4 ione = 1;
        FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size, a, &lda, b, &ldb, &vl, &vu, &ione, &nv, &abstol, &m, 
                        &w[0], z, &ldz, &work[0], &work_sizes[0], &rwork[0], &iwork[0], &ifail[0], &info, (int4)1, 
                        (int4)1, (int4)1);

        if (m != nv) error(__FILE__, __LINE__, "Not all eigen-values are found.", fatal_err);

        if (info)
        {
            std::stringstream s;
            s << "zhegvx returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        memcpy(eval, &w[0], nv * sizeof(real8));

        return info;
    }
};

template<> struct eigenproblem<scalapack>
{
    static std::vector<int4> get_work_sizes(int id, int4 matrix_size, int4 nb, int4 nprow, int4 npcol, int blacs_context)
    {
        std::vector<int4> work_sizes(3);
        
        int4 nn = std::max(matrix_size, std::max(nb, 2));
        
        switch (id)
        {
            case 0: // pzheevd
            {
                int4 np0 = linalg<scalapack>::numroc(nn, nb, 0, 0, nprow);
                int4 mq0 = linalg<scalapack>::numroc(nn, nb, 0, 0, npcol);

                work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;

                work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;

                work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;

                break;
            }
            case 1: // pzhegvx
            {
                int4 neig = std::max(1024, nb);

                int4 nmax3 = std::max(neig, std::max(nb, 2));
                
                int4 np = nprow * npcol;

                // due to the mess in the documentation, take the maximum of np0, nq0, mq0
                int4 nmpq0 = std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, nprow), 
                                      std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, npcol),
                                               linalg<scalapack>::numroc(nmax3, nb, 0, 0, npcol))); 

                int4 anb = linalg<scalapack>::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
                int4 sqnpc = (int4)pow(real8(np), 0.5);
                int4 nps = std::max(linalg<scalapack>::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);

                work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
                work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
                work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);

                work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) + 
                                linalg<scalapack>::iceil(neig, np) * nn + neig * matrix_size;

                int4 nnp = std::max(matrix_size, std::max(np + 1, 4));
                work_sizes[2] = 6 * nnp;

                break;
            }
            default:
            {
                error(__FILE__, __LINE__, "wrong eigen value solver id");
            }
        }

        return work_sizes;
    }

    static int standard(int4 matrix_size, int4 nb, int4 num_ranks_row, int4 num_ranks_col, int blacs_context, 
                        complex16* a, int4 lda, real8* eval, complex16* z, int4 ldz)
    {                
        int desca[9];
        linalg<scalapack>::descinit(desca, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, lda);
        
        int descz[9];
        linalg<scalapack>::descinit(descz, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, ldz);
        
        std::vector<int4> work_sizes = get_work_sizes(0, matrix_size, nb, num_ranks_row, num_ranks_col, blacs_context);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int4> iwork(work_sizes[2]);
        int4 info;

        int4 ione = 1;
        FORTRAN(pzheevd)("V", "U", &matrix_size, a, &ione, &ione, desca, eval, z, &ione, &ione, descz, &work[0], 
                         &work_sizes[0], &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &info, (int4)1, (int4)1);

        if (info)
        {
            std::stringstream s;
            s << "pzheevd returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        return info;
    }

    static int generalized(int4 matrix_size, int4 nb, int num_ranks_row, int num_ranks_col, int blacs_context, 
                           int4 nv, real8 abstol, complex16* a, int4 lda, complex16* b, int4 ldb, real8* eval, 
                           complex16* z, int4 ldz)
    {
        assert(nv <= matrix_size);
        
        int4 desca[9];
        linalg<scalapack>::descinit(desca, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, lda);

        int4 descb[9];
        linalg<scalapack>::descinit(descb, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, ldb); 

        int4 descz[9];
        linalg<scalapack>::descinit(descz, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, ldz); 

        std::vector<int4> work_sizes = get_work_sizes(1, matrix_size, nb, num_ranks_row, num_ranks_col, blacs_context);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int4> iwork(work_sizes[2]);
        
        std::vector<int4> ifail(matrix_size);
        std::vector<int4> iclustr(2 * num_ranks_row * num_ranks_col);
        std::vector<real8> gap(num_ranks_row * num_ranks_col);
        std::vector<real8> w(matrix_size);
        
        real8 orfac = 1e-6;
        int4 ione = 1;
        
        int4 m;
        int4 nz;
        real8 d1;
        int4 info;

        FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, a, &ione, &ione, desca, b, &ione, &ione, descb, &d1, &d1, 
                         &ione, &nv, &abstol, &m, &nz, &w[0], &orfac, z, &ione, &ione, descz, &work[0], &work_sizes[0], 
                         &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &ifail[0], &iclustr[0], &gap[0], &info, 
                         (int4)1, (int4)1, (int4)1); 

        if (info)
        {
            if ((info / 2) % 2)
            {
                std::stringstream s;
                s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                  << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                int k = num_ranks_row * num_ranks_col;
                for (int i = 0; i < num_ranks_row * num_ranks_col - 1; i++)
                {
                    if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                    {
                        k = i + 1;
                        break;
                    }
                }
               
                s << "number of eigenvalue clusters : " << k << std::endl;
                for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                error(__FILE__, __LINE__, s, fatal_err);
            }

            std::stringstream s;
            s << "pzhegvx returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        if ((m != nv) || (nz != nv))
            error(__FILE__, __LINE__, "Not all eigen-vectors or eigen-values are found.", fatal_err);

        memcpy(eval, &w[0], nv * sizeof(real8));

        return info;
    }
};

template<> struct eigenproblem<elpa>
{
    static void standard(int4 matrix_size, int4 nb, int4 num_ranks_row, int4 num_ranks_col, int blacs_context, 
                        complex16* a, int4 lda, real8* eval, complex16* z, int4 ldz)
    {                
        eigenproblem<scalapack>::standard(matrix_size, nb, num_ranks_row, num_ranks_col, blacs_context, a, lda, eval, 
                                          z, ldz);
    }

    static void generalized(int4 matrix_size, int4 nb, int4 na_rows, int4 num_ranks_row, int4 rank_row,
                            int4 na_cols, int4 num_ranks_col, int4 rank_col, int blacs_context, 
                            int4 nv, complex16* a, int4 lda, complex16* b, int4 ldb, real8* eval, 
                            complex16* z, int4 ldz, MPI_Comm comm_row, MPI_Comm comm_col, MPI_Comm comm_all)
    {
        assert(nv <= matrix_size);

        int4 mpi_comm_rows = MPI_Comm_c2f(comm_row);
        int4 mpi_comm_cols = MPI_Comm_c2f(comm_col);
        int4 mpi_comm_all = MPI_Comm_c2f(comm_all);

        FORTRAN(elpa_cholesky_complex)(&matrix_size, b, &ldb, &nb, &mpi_comm_rows, &mpi_comm_cols);
        FORTRAN(elpa_invert_trm_complex)(&matrix_size, b, &ldb, &nb, &mpi_comm_rows, &mpi_comm_cols);
       
        mdarray<complex16, 2> tmp1(na_rows, na_cols);
        mdarray<complex16, 2> tmp2(na_rows, na_cols);

        FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, b, &ldb, a, &lda, &nb, &mpi_comm_rows, 
                                        &mpi_comm_cols, tmp1.get_ptr(), &na_rows, 1, 1);

        int4 descc[9];
        linalg<scalapack>::descinit(descc, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, lda);

        linalg<scalapack>::pztranc(matrix_size, matrix_size, complex16(1, 0), tmp1.get_ptr(), 1, 1, descc, 
                                   complex16(0, 0), tmp2.get_ptr(), 1, 1, descc);

        FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, b, &ldb, tmp2.get_ptr(), &na_rows, &nb, 
                                        &mpi_comm_rows, &mpi_comm_cols, a, &lda, 1, 1);

        linalg<scalapack>::pztranc(matrix_size, matrix_size, complex16(1, 0), a, 1, 1, descc, complex16(0, 0), 
                                   tmp1.get_ptr(), 1, 1, descc);

        for (int i = 0; i < na_cols; i++)
        {
            int4 n_col = linalg<scalapack>::indxl2g(i + 1, nb,  rank_col, 0, num_ranks_col);
            int4 n_row = linalg<scalapack>::numroc(n_col, nb, rank_row, 0, num_ranks_row);
            for (int j = n_row; j < na_rows; j++) 
            {
                assert(j < na_rows);
                assert(i < na_cols);
                a[j + i * lda] = tmp1(j, i);
            }
        }
        
        std::vector<double> w(matrix_size);
        FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nv, a, &lda, &w[0], tmp1.get_ptr(), &na_rows, &nb,
                                               &mpi_comm_rows, &mpi_comm_cols, &mpi_comm_all);

        linalg<scalapack>::pztranc(matrix_size, matrix_size, complex16(1, 0), b, 1, 1, descc, complex16(0, 0), 
                                   tmp2.get_ptr(), 1, 1, descc);

        FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nv, tmp2.get_ptr(), &na_rows, tmp1.get_ptr(), &na_rows,
                                        &nb, &mpi_comm_rows, &mpi_comm_cols, z, &ldz, 1, 1);

        memcpy(eval, &w[0], nv * sizeof(real8));
    }
};

#endif // __LINALG_H__


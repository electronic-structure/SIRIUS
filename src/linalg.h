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

template<processing_unit_t> struct blas;

// CPU
template<> struct blas<cpu>
{
    template <typename T>
    static inline void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, T* a, int32_t lda, T* b, int32_t ldb, 
                            T beta, T* c, int32_t ldc);
    
    template <typename T>
    static inline void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, T* b, int32_t ldb, T* c, 
                            int32_t ldc);
    template<typename T>
    static inline void hemm(int side, int uplo, int32_t m, int32_t n, T alpha, T* a, int32_t lda, T* b, int32_t ldb, T beta, T* c, 
                            int32_t ldc);
};

template<> inline void blas<cpu>::gemm<real8>(int transa, int transb, int32_t m, int32_t n, int32_t k, real8 alpha, 
                                              real8* a, int32_t lda, real8* b, int32_t ldb, real8 beta, real8* c, 
                                              int32_t ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<> inline void blas<cpu>::gemm<real8>(int transa, int transb, int32_t m, int32_t n, int32_t k, real8* a, int32_t lda, 
                                              real8* b, int32_t ldb, real8* c, int32_t ldc)
{
    gemm(transa, transb, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

template<> inline void blas<cpu>::gemm<complex16>(int transa, int transb, int32_t m, int32_t n, int32_t k, complex16 alpha, 
                                                  complex16* a, int32_t lda, complex16* b, int32_t ldb, complex16 beta, 
                                                  complex16* c, int32_t ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<> inline void blas<cpu>::gemm<complex16>(int transa, int transb, int32_t m, int32_t n, int32_t k, complex16* a, 
                                                    int32_t lda, complex16* b, int32_t ldb, complex16* c, int32_t ldc)
{
    gemm(transa, transb, m, n, k, complex16(1, 0), a, lda, b, ldb, complex16(0, 0), c, ldc);
}

template<> inline void blas<cpu>::hemm<complex16>(int side, int uplo, int32_t m, int32_t n, complex16 alpha, complex16* a, 
                                                  int32_t lda, complex16* b, int32_t ldb, complex16 beta, complex16* c, 
                                                  int32_t ldc)
{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, (int32_t)1);
}



// GPU
template<> struct blas<gpu>
{
    template <typename T>
    static inline void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* alpha, T* a, int32_t lda, T* b, int32_t ldb, 
                            T* beta, T* c, int32_t ldc);
};

template<> inline void blas<gpu>::gemm<complex16>(int transa, int transb, int32_t m, int32_t n, int32_t k, complex16* alpha, 
                                                  complex16* a, int32_t lda, complex16* b, int32_t ldb, complex16* beta, 
                                                  complex16* c, int32_t ldc)
{
    cublas_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


template<linalg_t> struct linalg;

template<> struct linalg<lapack>
{
    static int32_t ilaenv(int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2, int32_t n3, 
                       int32_t n4)
    {
        return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(), 
                               (int32_t)opts.length());
    }
    
    template <typename T> 
    static int gesv(int32_t n, int32_t nrhs, T* a, int32_t lda, T* b, int32_t ldb);

    template <typename T> 
    static int gtsv(int32_t n, int32_t nrhs, T* dl, T* d, T* du, T* b, int32_t ldb);
};

template<> int linalg<lapack>::gesv<real8>(int32_t n, int32_t nrhs, real8* a, int32_t lda, real8* b, int32_t ldb)
{
    int32_t info;
    std::vector<int32_t> ipiv(n);

    FORTRAN(dgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

template<> int linalg<lapack>::gesv<complex16>(int32_t n, int32_t nrhs, complex16* a, int32_t lda, complex16* b, int32_t ldb)
{
    int32_t info;
    std::vector<int32_t> ipiv(n);

    FORTRAN(zgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);

    return info;
}

template<> int linalg<lapack>::gtsv<real8>(int32_t n, int32_t nrhs, real8* dl, real8* d, real8* du, real8* b, int32_t ldb)
{
    int info;

    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
            
    return info;
}

template<> int linalg<lapack>::gtsv<complex16>(int32_t n, int32_t nrhs, complex16* dl, complex16* d, complex16* du, 
                                               complex16* b, int32_t ldb)
{
    int32_t info;                   

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

    static void descinit(int32_t* desc, int32_t m, int32_t n, int32_t mb, int32_t nb, int32_t irsrc, int32_t icsrc, int32_t ictxt, int32_t lld)
    {
        int32_t info;
    
        FORTRAN(descinit)(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);
    
        if (info) error(__FILE__, __LINE__, "error in descinit");
    }

    static int pjlaenv(int32_t ictxt, int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2, 
                       int32_t n3, int32_t n4)
    {
        return FORTRAN(pjlaenv)(&ictxt, &ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(),
                                (int32_t)opts.length());
    }

    static int32_t numroc(int32_t n, int32_t nb, int32_t iproc, int32_t isrcproc, int32_t nprocs)
    {
        return FORTRAN(numroc)(&n, &nb, &iproc, &isrcproc, &nprocs); 
    }

    static int32_t indxl2g(int32_t indxloc, int32_t nb, int32_t iproc, int32_t isrcproc, int32_t nprocs)
    {
        return FORTRAN(indxl2g)(&indxloc, &nb, &iproc, &isrcproc, &nprocs);
    }

    static int32_t iceil(int32_t inum, int32_t idenom)
    {
        return FORTRAN(iceil)(&inum, &idenom);
    }

    static void pztranc(int32_t m, int32_t n, complex16 alpha, complex16* a, int32_t ia, int32_t ja, int32_t* desca, 
                        complex16 beta, complex16* c, int32_t ic, int32_t jc, int32_t* descc)
    {
        FORTRAN(pztranc)(&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }
};

/// Declaration of an eigenproblem class to be specialized for a particular solver
template<linalg_t> struct eigenproblem;

/// Lapack eigenvalue specialization
template<> struct eigenproblem<lapack>
{
    static std::vector<int32_t> get_work_sizes(int id, int32_t matrix_size)
    {
        std::vector<int32_t> work_sizes(3);
        
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
                int32_t nb = linalg<lapack>::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
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
    
    static int standard(int32_t matrix_size, complex16* a, int32_t lda, real8* eval, complex16* z, int32_t ldz)
    {
        std::vector<int32_t> work_sizes = get_work_sizes(0, matrix_size);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int32_t> iwork(work_sizes[2]);
        int32_t info;

        FORTRAN(zheevd)("V", "U", &matrix_size, a, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                        &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);
        
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

    static int generalized(int32_t matrix_size, int32_t nv, real8 abstol, complex16* a, int32_t lda, complex16* b, int32_t ldb, 
                           real8* eval, complex16* z, int32_t ldz)
    {
        assert(nv <= matrix_size);

        std::vector<int32_t> work_sizes = get_work_sizes(1, matrix_size);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int32_t> iwork(work_sizes[2]);
        std::vector<int32_t> ifail(matrix_size);
        std::vector<real8> w(matrix_size);
        real8 vl = 0.0;
        real8 vu = 0.0;
        int32_t m;
        int32_t info;
       
        int32_t ione = 1;
        FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size, a, &lda, b, &ldb, &vl, &vu, &ione, &nv, &abstol, &m, 
                        &w[0], z, &ldz, &work[0], &work_sizes[0], &rwork[0], &iwork[0], &ifail[0], &info, (int32_t)1, 
                        (int32_t)1, (int32_t)1);

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
    static std::vector<int32_t> get_work_sizes(int id, int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, int blacs_context)
    {
        std::vector<int32_t> work_sizes(3);
        
        int32_t nn = std::max(matrix_size, std::max(nb, 2));
        
        switch (id)
        {
            case 0: // pzheevd
            {
                int32_t np0 = linalg<scalapack>::numroc(nn, nb, 0, 0, nprow);
                int32_t mq0 = linalg<scalapack>::numroc(nn, nb, 0, 0, npcol);

                work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;

                work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;

                work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;

                break;
            }
            case 1: // pzhegvx
            {
                int32_t neig = std::max(1024, nb);

                int32_t nmax3 = std::max(neig, std::max(nb, 2));
                
                int32_t np = nprow * npcol;

                // due to the mess in the documentation, take the maximum of np0, nq0, mq0
                int32_t nmpq0 = std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, nprow), 
                                      std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, npcol),
                                               linalg<scalapack>::numroc(nmax3, nb, 0, 0, npcol))); 

                int32_t anb = linalg<scalapack>::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
                int32_t sqnpc = (int32_t)pow(real8(np), 0.5);
                int32_t nps = std::max(linalg<scalapack>::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);

                work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
                work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
                work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);

                work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) + 
                                linalg<scalapack>::iceil(neig, np) * nn + neig * matrix_size;

                int32_t nnp = std::max(matrix_size, std::max(np + 1, 4));
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

    static int standard(int32_t matrix_size, int32_t nb, int32_t num_ranks_row, int32_t num_ranks_col, int blacs_context, 
                        complex16* a, int32_t lda, real8* eval, complex16* z, int32_t ldz)
    {                
        int desca[9];
        linalg<scalapack>::descinit(desca, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, lda);
        
        int descz[9];
        linalg<scalapack>::descinit(descz, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, ldz);
        
        std::vector<int32_t> work_sizes = get_work_sizes(0, matrix_size, nb, num_ranks_row, num_ranks_col, blacs_context);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int32_t> iwork(work_sizes[2]);
        int32_t info;

        int32_t ione = 1;
        FORTRAN(pzheevd)("V", "U", &matrix_size, a, &ione, &ione, desca, eval, z, &ione, &ione, descz, &work[0], 
                         &work_sizes[0], &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);

        if (info)
        {
            std::stringstream s;
            s << "pzheevd returned " << info; 
            error(__FILE__, __LINE__, s, fatal_err);
        }

        return info;
    }

    static int generalized(int32_t matrix_size, int32_t nb, int num_ranks_row, int num_ranks_col, int blacs_context, 
                           int32_t nv, real8 abstol, complex16* a, int32_t lda, complex16* b, int32_t ldb, real8* eval, 
                           complex16* z, int32_t ldz)
    {
        assert(nv <= matrix_size);
        
        int32_t desca[9];
        linalg<scalapack>::descinit(desca, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, lda);

        int32_t descb[9];
        linalg<scalapack>::descinit(descb, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, ldb); 

        int32_t descz[9];
        linalg<scalapack>::descinit(descz, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, ldz); 

        std::vector<int32_t> work_sizes = get_work_sizes(1, matrix_size, nb, num_ranks_row, num_ranks_col, blacs_context);
        
        std::vector<complex16> work(work_sizes[0]);
        std::vector<real8> rwork(work_sizes[1]);
        std::vector<int32_t> iwork(work_sizes[2]);
        
        std::vector<int32_t> ifail(matrix_size);
        std::vector<int32_t> iclustr(2 * num_ranks_row * num_ranks_col);
        std::vector<real8> gap(num_ranks_row * num_ranks_col);
        std::vector<real8> w(matrix_size);
        
        real8 orfac = 1e-6;
        int32_t ione = 1;
        
        int32_t m;
        int32_t nz;
        real8 d1;
        int32_t info;

        FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, a, &ione, &ione, desca, b, &ione, &ione, descb, &d1, &d1, 
                         &ione, &nv, &abstol, &m, &nz, &w[0], &orfac, z, &ione, &ione, descz, &work[0], &work_sizes[0], 
                         &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &ifail[0], &iclustr[0], &gap[0], &info, 
                         (int32_t)1, (int32_t)1, (int32_t)1); 

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
    static void standard(int32_t matrix_size, int32_t nb, int32_t num_ranks_row, int32_t num_ranks_col, int blacs_context, 
                        complex16* a, int32_t lda, real8* eval, complex16* z, int32_t ldz)
    {                
        eigenproblem<scalapack>::standard(matrix_size, nb, num_ranks_row, num_ranks_col, blacs_context, a, lda, eval, 
                                          z, ldz);
    }

    static void generalized(int32_t matrix_size, int32_t nb, int32_t na_rows, int32_t num_ranks_row, int32_t rank_row,
                            int32_t na_cols, int32_t num_ranks_col, int32_t rank_col, int blacs_context, 
                            int32_t nv, complex16* a, int32_t lda, complex16* b, int32_t ldb, real8* eval, 
                            complex16* z, int32_t ldz, MPI_Comm comm_row, MPI_Comm comm_col, MPI_Comm comm_all)
    {
        assert(nv <= matrix_size);

        int32_t mpi_comm_rows = MPI_Comm_c2f(comm_row);
        int32_t mpi_comm_cols = MPI_Comm_c2f(comm_col);
        int32_t mpi_comm_all = MPI_Comm_c2f(comm_all);

        sirius::Timer *t;

        t = new sirius::Timer("elpa::ort");
        FORTRAN(elpa_cholesky_complex)(&matrix_size, b, &ldb, &nb, &mpi_comm_rows, &mpi_comm_cols);
        FORTRAN(elpa_invert_trm_complex)(&matrix_size, b, &ldb, &nb, &mpi_comm_rows, &mpi_comm_cols);
       
        mdarray<complex16, 2> tmp1(na_rows, na_cols);
        mdarray<complex16, 2> tmp2(na_rows, na_cols);

        FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, b, &ldb, a, &lda, &nb, &mpi_comm_rows, 
                                        &mpi_comm_cols, tmp1.get_ptr(), &na_rows, 1, 1);

        int32_t descc[9];
        linalg<scalapack>::descinit(descc, matrix_size, matrix_size, nb, nb, 0, 0, blacs_context, lda);

        linalg<scalapack>::pztranc(matrix_size, matrix_size, complex16(1, 0), tmp1.get_ptr(), 1, 1, descc, 
                                   complex16(0, 0), tmp2.get_ptr(), 1, 1, descc);

        FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, b, &ldb, tmp2.get_ptr(), &na_rows, &nb, 
                                        &mpi_comm_rows, &mpi_comm_cols, a, &lda, 1, 1);

        linalg<scalapack>::pztranc(matrix_size, matrix_size, complex16(1, 0), a, 1, 1, descc, complex16(0, 0), 
                                   tmp1.get_ptr(), 1, 1, descc);

        for (int i = 0; i < na_cols; i++)
        {
            int32_t n_col = linalg<scalapack>::indxl2g(i + 1, nb,  rank_col, 0, num_ranks_col);
            int32_t n_row = linalg<scalapack>::numroc(n_col, nb, rank_row, 0, num_ranks_row);
            for (int j = n_row; j < na_rows; j++) 
            {
                assert(j < na_rows);
                assert(i < na_cols);
                a[j + i * lda] = tmp1(j, i);
            }
        }
        delete t;
        
        t = new sirius::Timer("elpa::diag");
        std::vector<double> w(matrix_size);
        FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nv, a, &lda, &w[0], tmp1.get_ptr(), &na_rows, &nb,
                                               &mpi_comm_rows, &mpi_comm_cols, &mpi_comm_all);
        delete t;

        t = new sirius::Timer("elpa::bt");
        linalg<scalapack>::pztranc(matrix_size, matrix_size, complex16(1, 0), b, 1, 1, descc, complex16(0, 0), 
                                   tmp2.get_ptr(), 1, 1, descc);

        FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nv, tmp2.get_ptr(), &na_rows, tmp1.get_ptr(), &na_rows,
                                        &nb, &mpi_comm_rows, &mpi_comm_cols, z, &ldz, 1, 1);
        delete t;

        memcpy(eval, &w[0], nv * sizeof(real8));
    }
};

/// magma eigenvalue specialization
template<> struct eigenproblem<magma>
{
    static std::vector<int32_t> get_work_sizes(int id, int32_t matrix_size)
    {
        std::vector<int32_t> work_sizes(3);
        
        //* switch (id)
        //* {
        //*     case 0: // zheevd
        //*     {
        //*         work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
        //*         work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
        //*         work_sizes[2] = 3 + 5 * matrix_size;
        //*         break;
        //*     }
        //*     case 1: // zhegvx
        //*     {
        //*         int32_t nb = linalg<lapack>::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
        //*         work_sizes[0] = (nb + 1) * matrix_size; // lwork
        //*         work_sizes[1] = 7 * matrix_size; // lrwork
        //*         work_sizes[2] = 5 * matrix_size; // liwork
        //*         break;
        //*     }
        //*     default:
        //*         error(__FILE__, __LINE__, "wrong eigen value solver id");
        //* }

        return work_sizes;
    }
    
    static int standard(int32_t matrix_size, complex16* a, int32_t lda, real8* eval, complex16* z, int32_t ldz)
    {
        //* std::vector<int32_t> work_sizes = get_work_sizes(0, matrix_size);
        //* 
        //* std::vector<complex16> work(work_sizes[0]);
        //* std::vector<real8> rwork(work_sizes[1]);
        //* std::vector<int32_t> iwork(work_sizes[2]);
        //* int32_t info;

        //* FORTRAN(zheevd)("V", "U", &matrix_size, a, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
        //*                 &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);
        //* 
        //* for (int i = 0; i < matrix_size; i++)
        //*     memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(complex16));
        //* 
        //* if (info)
        //* {
        //*     std::stringstream s;
        //*     s << "zheevd returned " << info; 
        //*     error(__FILE__, __LINE__, s, fatal_err);
        //* }

        //* return info;
        return 0;
    }

    static void generalized(int32_t matrix_size, int32_t nv, complex16* a, int32_t lda, complex16* b, int32_t ldb, 
                           real8* eval, complex16* z, int32_t ldz)
    {
        assert(nv <= matrix_size);
        
        magma_zhegvdx_2stage_wrapper(matrix_size, nv, a, lda, b, ldb, eval);

        for (int i = 0; i < nv; i++) memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(complex16));
    }
};

#endif // __LINALG_H__


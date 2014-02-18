#ifndef __LINALG_H__
#define __LINALG_H__

/** \file linalg.h

    \brief Linear algebra interface

*/

#include <stdint.h>
#include <string.h>
#include <iostream>
#include <complex>
#include <vector>
#include "config.h"
#ifdef _GPU_
#include "gpu_interface.h"
#endif
#include "constants.h"
#include "error_handling.h"

/* 
    matrix-vector operations
*/
extern "C" void FORTRAN(zgemv)(const char* trans, int32_t* m, int32_t* n, double_complex* alpha, 
                               double_complex* a, int32_t* lda, double_complex* x, int32_t* incx,
                               double_complex* beta, double_complex* y, int32_t* incy, int32_t translen);

/*
    matrix-matrix operations
*/
extern "C" void FORTRAN(zgemm)(const char* transa, const char* transb, int32_t* m, int32_t* n, int32_t* k, 
                               double_complex* alpha, double_complex* a, int32_t* lda, double_complex* b, int32_t* ldb, 
                               double_complex* beta, double_complex* c, int32_t* ldc, int32_t transalen, int32_t transblen);

extern "C" void FORTRAN(dgemm)(const char* transa, const char* transb, int32_t* m, int32_t* n, int32_t* k, 
                               double* alpha, double* a, int32_t* lda, double* b, int32_t* ldb, 
                               double* beta,double* c, int32_t* ldc, int32_t transalen, int32_t transblen);

extern "C" void FORTRAN(zhemm)(const char *side, const char* uplo, int32_t* m, int32_t* n, 
                               double_complex* alpha, double_complex* a, int32_t* lda, double_complex* b,
                               int32_t* ldb, double_complex* beta, double_complex* c, int32_t* ldc,
                               int32_t sidelen, int32_t uplolen);

/*
    eigen-value problem
*/
extern "C" void FORTRAN(zhegvx)(int32_t* itype, const char* jobz, const char* range, const char* uplo, 
                                int32_t* n, double_complex* a, int32_t* lda, double_complex* b, int32_t* ldb, double* vl, 
                                double* vu, int32_t* il, int32_t* iu, double* abstol, int32_t* m, double* w, double_complex* z,
                                int32_t* ldz, double_complex* work, int32_t* lwork, double* rwork, int32_t* iwork, int32_t* ifail, 
                                int32_t* info, int32_t jobzlen, int32_t rangelen, int32_t uplolen);

extern "C" int32_t FORTRAN(ilaenv)(int32_t* ispec, const char* name, const char* opts, int32_t* n1, int32_t* n2, int32_t* n3, 
                                int32_t* n4, int32_t namelen, int32_t optslen);

extern "C" void FORTRAN(zheev)(const char* jobz, const char* uplo, int32_t* n, double_complex* a,
                               int32_t* lda, double* w, double_complex* work, int32_t* lwork, double* rwork,
                               int32_t* info, int32_t jobzlen, int32_t uplolen);

extern "C" void FORTRAN(zheevd)(const char* jobz, const char* uplo, int32_t* n, double_complex* a,
                                int32_t* lda, double* w, double_complex* work, int32_t* lwork, double* rwork,
                                int32_t* lrwork, int32_t* iwork, int32_t* liwork, int32_t* info, int32_t jobzlen, int32_t uplolen);




extern "C" void FORTRAN(dptsv)(int32_t *n, int32_t *nrhs, double* d, double* *e, double* b, int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgtsv)(int32_t *n, int32_t *nrhs, double *dl, double *d, double *du, double *b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(zgtsv)(int32_t *n, int32_t *nrhs, double_complex* dl, double_complex* d, double_complex* du, double_complex* b, 
                               int32_t *ldb, int32_t *info);

extern "C" void FORTRAN(dgesv)(int32_t* n, int32_t* nrhs, double* a, int32_t* lda, int32_t* ipiv, double* b, int32_t* ldb, int32_t* info);

extern "C" void FORTRAN(zgesv)(int32_t* n, int32_t* nrhs, double_complex* a, int32_t* lda, int32_t* ipiv, double_complex* b, int32_t* ldb, int32_t* info);

extern "C" void FORTRAN(dgetrf)(int32_t* m, int32_t* n, double* a, int32_t* lda, int32_t* ipiv, int32_t* info);

extern "C" void FORTRAN(zgetrf)(int32_t* m, int32_t* n, double_complex* a, int32_t* lda, int32_t* ipiv, int32_t* info);

extern "C" void FORTRAN(dgetri)(int32_t* n, double* a, int32_t* lda, int32_t* ipiv, double* work, int32_t* lwork, int32_t* info);

extern "C" void FORTRAN(zgetri)(int32_t* n, double_complex* a, int32_t* lda, int32_t* ipiv, double_complex* work, int32_t* lwork, int32_t* info);


/* 
    BLACS and ScaLAPACK related functions
*/
#ifdef _SCALAPACK_
extern "C" int Csys2blacs_handle(MPI_Comm SysCtxt);
extern "C" MPI_Comm Cblacs2sys_handle(int BlacsCtxt);
extern "C" void Cblacs_gridinit(int* ConTxt, const char* order, int nprow, int npcol);
extern "C" void Cblacs_gridmap(int* ConTxt, int* usermap, int ldup, int nprow0, int npcol0);
extern "C" void Cblacs_gridinfo(int ConTxt, int* nprow, int* npcol, int* myrow, int* mycol);
extern "C" void Cfree_blacs_system_handle(int ISysCtxt);
extern "C" void Cblacs_barrier(int ConTxt, const char* scope);

extern "C" void FORTRAN(descinit)(int32_t* desc, int32_t* m, int32_t* n, int32_t* mb, int32_t* nb, int32_t* irsrc, int32_t* icsrc, 
                                  int32_t* ictxt, int32_t* lld, int32_t* info);

extern "C" void FORTRAN(pztranc)(int32_t* m, int32_t* n, double_complex* alpha, double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca,
                                 double_complex* beta, double_complex* c, int32_t* ic, int32_t* jc,int32_t* descc);

extern "C" void FORTRAN(pzhegvx)(int32_t* ibtype, const char* jobz, const char* range, const char* uplo, int32_t* n, 
                                 double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                 double_complex* b, int32_t* ib, int32_t* jb, int32_t* descb, 
                                 double* vl, double* vu, 
                                 int32_t* il, int32_t* iu, 
                                 double* abstol, 
                                 int32_t* m, int32_t* nz, double* w, double* orfac, 
                                 double_complex* z, int32_t* iz, int32_t* jz, int32_t* descz, 
                                 double_complex* work, int32_t* lwork, 
                                 double* rwork, int32_t* lrwork, 
                                 int32_t* iwork, int32_t* liwork, 
                                 int32_t* ifail, int32_t* iclustr, double* gap, int32_t* info, 
                                 int32_t jobz_len, int32_t range_len, int32_t uplo_len);

extern "C" void FORTRAN(pzheevd)(const char* jobz, const char* uplo, int32_t* n, 
                                 double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                 double* w, 
                                 double_complex* z, int32_t* iz, int32_t* jz, int32_t* descz, 
                                 double_complex* work, int32_t* lwork, double* rwork, int32_t* lrwork, int32_t* iwork, 
                                 int32_t* liwork, int32_t* info, int32_t jobz_len, int32_t uplo_len);

extern "C" void FORTRAN(pzgemm)(const char* transa, const char* transb, 
                                int32_t* m, int32_t* n, int32_t* k, 
                                double_complex* aplha,
                                double_complex* a, int32_t* ia, int32_t* ja, int32_t* desca, 
                                double_complex* b, int32_t* ib, int32_t* jb, int32_t* descb,
                                double_complex* beta,
                                double_complex* c, int32_t* ic, int32_t* jc, int32_t* descc,
                                int32_t transa_len, int32_t transb_len);

extern "C" int32_t FORTRAN(numroc)(int32_t* n, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" int32_t FORTRAN(indxl2g)(int32_t* indxloc, int32_t* nb, int32_t* iproc, int32_t* isrcproc, int32_t* nprocs);

extern "C" int32_t FORTRAN(pjlaenv)(int32_t* ictxt, int32_t* ispec, const char* name, const char* opts, int32_t* n1, int32_t* n2, 
                                 int32_t* n3, int32_t* n4, int32_t namelen, int32_t optslen);

extern "C" int32_t FORTRAN(iceil)(int32_t* inum, int32_t* idenom);
#endif

#ifdef _ELPA_
extern "C" void FORTRAN(elpa_cholesky_complex)(int32_t* na, double_complex* a, int32_t* lda, int32_t* nblk, int32_t* mpi_comm_rows, 
                                               int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_invert_trm_complex)(int32_t* na, double_complex* a, int32_t* lda, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                 int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_mult_ah_b_complex)(const char* uplo_a, const char* uplo_c, int32_t* na, int32_t* ncb, 
                                                double_complex* a, int32_t* lda, double_complex* b, int32_t* ldb, int32_t* nblk, 
                                                int32_t* mpi_comm_rows, int32_t* mpi_comm_cols, double_complex* c, int32_t* ldc,
                                                int32_t uplo_a_len, int32_t uplo_c_len);

extern "C" void FORTRAN(elpa_solve_evp_complex)(int32_t* na, int32_t* nev, double_complex* a, int32_t* lda, double* ev, double_complex* q, 
                                                int32_t* ldq, int32_t* nblk, int32_t* mpi_comm_rows, int32_t* mpi_comm_cols);

extern "C" void FORTRAN(elpa_solve_evp_complex_2stage)(int32_t* na, int32_t* nev, double_complex* a, int32_t* lda, double* ev, 
                                                       double_complex* q, int32_t* ldq, int32_t* nblk, int32_t* mpi_comm_rows, 
                                                       int32_t* mpi_comm_cols, int32_t* mpi_comm_all);
#endif

template<processing_unit_t> 
class blas;

// CPU
template<> class blas<cpu>
{
    public:

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T beta, T* c, int32_t ldc);
        
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* c, int32_t ldc);
                         
        template<typename T>
        static void hemm(int side, int uplo, int32_t m, int32_t n, T alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T beta, T* c, int32_t ldc);

        template<typename T>
        static void gemv(int trans, int32_t m, int32_t n, T alpha, T* a, int32_t lda, T* x, int32_t incx, 
                         T beta, T* y, int32_t incy);
};

#ifdef _GPU_
template<> class blas<gpu>
{
    private:
        static double_complex zone;
        static double_complex zzero;

    public:

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* beta, T* c, int32_t ldc);
        
        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T* a, int32_t lda, 
                         T* b, int32_t ldb, T* c, int32_t ldc);
};
#endif

#ifdef _SCALAPACK_
template<processing_unit_t> 
class pblas;

template<> class pblas<cpu>
{
    public:

        template <typename T>
        static void gemm(int transa, int transb, int32_t m, int32_t n, int32_t k, T alpha, T* a, int32_t lda, 
                         T* b, int32_t ldb, T beta, T* c, int32_t ldc, int block_size, int blacs_context);
};
#endif


template<linalg_t> 
class linalg;

template<> class linalg<lapack>
{
    public:

        static int32_t ilaenv(int32_t ispec, const std::string& name, const std::string& opts, int32_t n1, int32_t n2, 
                              int32_t n3, int32_t n4)
        {
            return FORTRAN(ilaenv)(&ispec, name.c_str(), opts.c_str(), &n1, &n2, &n3, &n4, (int32_t)name.length(), 
                                   (int32_t)opts.length());
        }
        
        template <typename T> 
        static int gesv(int32_t n, int32_t nrhs, T* a, int32_t lda, T* b, int32_t ldb);

        template <typename T> 
        static int gtsv(int32_t n, int32_t nrhs, T* dl, T* d, T* du, T* b, int32_t ldb);

        template <typename T>
        static int getrf(int32_t m, int32_t n, T* a, int32_t lda, int32_t* ipiv);
        
        template <typename T>
        static int getri(int32_t n, T* a, int32_t lda, int32_t* ipiv, T* work, int32_t lwork);

        template <typename T>
        static void invert_ge(T* mtrx, int size)
        {
            int32_t nb = std::max(ilaenv(1, "dgetri", "U", size, -1, -1, -1), ilaenv(1, "zgetri", "U", size, -1, -1, -1));
            int32_t lwork = size * nb;
            std::vector<T> work(lwork);
            std::vector<int> ipiv(size);
            int info = getrf(size, size, mtrx, size, &ipiv[0]);
            if (info != 0)
            {
                std::stringstream s;
                s << "getrf returned : " << info;
                error_local(__FILE__, __LINE__, s);
            }

            info = getri(size, mtrx, size, &ipiv[0], &work[0], lwork);
            if (info != 0)
            {
                std::stringstream s;
                s << "getri returned : " << info;
                error_local(__FILE__, __LINE__, s);
            }
        }
};

#ifdef _SCALAPACK_
template<> class linalg<scalapack>
{
    public:

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
        
            if (info) error_local(__FILE__, __LINE__, "error in descinit");
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

        static void pztranc(int32_t m, int32_t n, double_complex alpha, double_complex* a, int32_t ia, int32_t ja, int32_t* desca, 
                            double_complex beta, double_complex* c, int32_t ic, int32_t jc, int32_t* descc)
        {
            FORTRAN(pztranc)(&m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
        }
};
#endif

class standard_evp
{
    public:
        virtual ~standard_evp()
        {
        }

        virtual void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {
            error_local(__FILE__, __LINE__, "eigen-value solver is not configured");
        }
};

class standard_evp_lapack: public standard_evp
{
    private:
     
        std::vector<int32_t> get_work_sizes(int32_t matrix_size)
        {
            std::vector<int32_t> work_sizes(3);
            
            work_sizes[0] = 2 * matrix_size + matrix_size * matrix_size;
            work_sizes[1] = 1 + 5 * matrix_size + 2 * matrix_size * matrix_size;
            work_sizes[2] = 3 + 5 * matrix_size;
            return work_sizes;
        }

    public:
        
        standard_evp_lapack()
        {
        }
       
        void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            FORTRAN(zheevd)("V", "U", &matrix_size, a, &lda, eval, &work[0], &work_sizes[0], &rwork[0], &work_sizes[1], 
                            &iwork[0], &work_sizes[2], &info, (int32_t)1, (int32_t)1);
            
            for (int i = 0; i < matrix_size; i++)
                memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(double_complex));
            
            if (info)
            {
                std::stringstream s;
                s << "zheevd returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }
        }
};

#ifdef _PLASMA_
extern "C" void plasma_zheevd_wrapper(int32_t matrix_size, void* a, int32_t lda, void* z,
                                      int32_t ldz, double* eval);
#endif

class standard_evp_plasma: public standard_evp
{
    public:

        standard_evp_plasma()
        {
        }

        #ifdef _PLASMA_
        void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {
            //plasma_set_num_threads(1);
            //omp_set_num_threads(1);
            //printf("before call to plasma_zheevd_wrapper\n");
            plasma_zheevd_wrapper(matrix_size, a, lda, z, lda, eval);
            //printf("after call to plasma_zheevd_wrapper\n");
            //plasma_set_num_threads(8);
            //omp_set_num_threads(8);
        }
        #endif
};


class standard_evp_scalapack: public standard_evp
{
    private:
        int32_t block_size_;
        int num_ranks_row_;
        int num_ranks_col_;
        int blacs_context_;
        
        #ifdef _SCALAPACK_
        std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
                                            int blacs_context)
        {
            std::vector<int32_t> work_sizes(3);
            
            int32_t nn = std::max(matrix_size, std::max(nb, 2));
            
            int32_t np0 = linalg<scalapack>::numroc(nn, nb, 0, 0, nprow);
            int32_t mq0 = linalg<scalapack>::numroc(nn, nb, 0, 0, npcol);
        
            work_sizes[0] = matrix_size + (np0 + mq0 + nb) * nb;
        
            work_sizes[1] = 1 + 9 * matrix_size + 3 * np0 * mq0;
        
            work_sizes[2] = 7 * matrix_size + 8 * npcol + 2;
            
            return work_sizes;
        }
        #endif

    public:

        standard_evp_scalapack(int32_t block_size__, int num_ranks_row__, int num_ranks_col__, int blacs_context__) 
            : block_size_(block_size__), 
              num_ranks_row_(num_ranks_row__), 
              num_ranks_col_(num_ranks_col__), 
              blacs_context_(blacs_context__)
        {
        }

        #ifdef _SCALAPACK_
        void solve(int32_t matrix_size, double_complex* a, int32_t lda, double* eval, double_complex* z, int32_t ldz)
        {

            int desca[9];
            linalg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);
            
            int descz[9];
            linalg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldz);
            
            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size, block_size_, num_ranks_row_, num_ranks_col_, 
                                                             blacs_context_);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            int32_t info;

            int32_t ione = 1;
            FORTRAN(pzheevd)("V", "U", &matrix_size, a, &ione, &ione, desca, eval, z, &ione, &ione, descz, &work[0], 
                             &work_sizes[0], &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &info, (int32_t)1, 
                             (int32_t)1);

            if (info)
            {
                std::stringstream s;
                s << "pzheevd returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }
        }
        #endif
};

class generalized_evp
{
    public:
        virtual ~generalized_evp()
        {
        }

        virtual void solve(int32_t matrix_size, int32_t nevec, double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                           double* eval, double_complex* z, int32_t ldz)
        {
            error_local(__FILE__, __LINE__, "eigen-value solver is not configured");
        }
};

class generalized_evp_lapack: public generalized_evp
{
    private:

        double abstol_;
    
    public:

        generalized_evp_lapack(double abstol__) : abstol_(abstol__)
        {
        }

        void solve(int32_t matrix_size, int32_t nevec, double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                   double* eval, double_complex* z, int32_t ldz)
        {
            assert(nevec <= matrix_size);

            int nb = linalg<lapack>::ilaenv(1, "ZHETRD", "U", matrix_size, 0, 0, 0);
            int lwork = (nb + 1) * matrix_size; // lwork
            int lrwork = 7 * matrix_size; // lrwork
            int liwork = 5 * matrix_size; // liwork
            
            std::vector<double_complex> work(lwork);
            std::vector<double> rwork(lrwork);
            std::vector<int32_t> iwork(liwork);
            std::vector<int32_t> ifail(matrix_size);
            std::vector<double> w(matrix_size);
            double vl = 0.0;
            double vu = 0.0;
            int32_t m;
            int32_t info;
       
            int32_t ione = 1;
            FORTRAN(zhegvx)(&ione, "V", "I", "U", &matrix_size, a, &lda, b, &ldb, &vl, &vu, &ione, &nevec, &abstol_, &m, 
                            &w[0], z, &ldz, &work[0], &lwork, &rwork[0], &iwork[0], &ifail[0], &info, (int32_t)1, 
                            (int32_t)1, (int32_t)1);

            if (m != nevec) error_local(__FILE__, __LINE__, "Not all eigen-values are found.");

            if (info)
            {
                std::stringstream s;
                s << "zhegvx returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }

            memcpy(eval, &w[0], nevec * sizeof(double));
        }


};

class generalized_evp_scalapack: public generalized_evp
{
    private:

        int32_t block_size_;
        int num_ranks_row_;
        int num_ranks_col_;
        int blacs_context_;
        double abstol_;
        
        #ifdef _SCALAPACK_
        std::vector<int32_t> get_work_sizes(int32_t matrix_size, int32_t nb, int32_t nprow, int32_t npcol, 
                                            int blacs_context)
        {
            std::vector<int32_t> work_sizes(3);
            
            int32_t nn = std::max(matrix_size, std::max(nb, 2));
            
            int32_t neig = std::max(1024, nb);

            int32_t nmax3 = std::max(neig, std::max(nb, 2));
            
            int32_t np = nprow * npcol;

            // due to the mess in the documentation, take the maximum of np0, nq0, mq0
            int32_t nmpq0 = std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, nprow), 
                                  std::max(linalg<scalapack>::numroc(nn, nb, 0, 0, npcol),
                                           linalg<scalapack>::numroc(nmax3, nb, 0, 0, npcol))); 

            int32_t anb = linalg<scalapack>::pjlaenv(blacs_context, 3, "PZHETTRD", "L", 0, 0, 0, 0);
            int32_t sqnpc = (int32_t)pow(double(np), 0.5);
            int32_t nps = std::max(linalg<scalapack>::numroc(nn, 1, 0, 0, sqnpc), 2 * anb);

            work_sizes[0] = matrix_size + (2 * nmpq0 + nb) * nb;
            work_sizes[0] = std::max(work_sizes[0], matrix_size + 2 * (anb + 1) * (4 * nps + 2) + (nps + 1) * nps);
            work_sizes[0] = std::max(work_sizes[0], 3 * nmpq0 * nb + nb * nb);

            work_sizes[1] = 4 * matrix_size + std::max(5 * matrix_size, nmpq0 * nmpq0) + 
                            linalg<scalapack>::iceil(neig, np) * nn + neig * matrix_size;

            int32_t nnp = std::max(matrix_size, std::max(np + 1, 4));
            work_sizes[2] = 6 * nnp;

            return work_sizes;
        }
        #endif
    
    public:

        generalized_evp_scalapack(int32_t block_size__, int num_ranks_row__, int num_ranks_col__, int blacs_context__, 
                                  double abstol__) 
            : block_size_(block_size__), 
              num_ranks_row_(num_ranks_row__), 
              num_ranks_col_(num_ranks_col__), 
              blacs_context_(blacs_context__),
              abstol_(abstol__)
        {
        }

        #ifdef _SCALAPACK_
        void solve(int32_t matrix_size, int32_t nevec, double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                   double* eval, double_complex* z, int32_t ldz)
        {
        
            assert(nevec <= matrix_size);
            
            int32_t desca[9];
            linalg<scalapack>::descinit(desca, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            int32_t descb[9];
            linalg<scalapack>::descinit(descb, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldb); 

            int32_t descz[9];
            linalg<scalapack>::descinit(descz, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, ldz); 

            std::vector<int32_t> work_sizes = get_work_sizes(matrix_size, block_size_, num_ranks_row_, num_ranks_col_, 
                                                             blacs_context_);
            
            std::vector<double_complex> work(work_sizes[0]);
            std::vector<double> rwork(work_sizes[1]);
            std::vector<int32_t> iwork(work_sizes[2]);
            
            std::vector<int32_t> ifail(matrix_size);
            std::vector<int32_t> iclustr(2 * num_ranks_row_ * num_ranks_col_);
            std::vector<double> gap(num_ranks_row_ * num_ranks_col_);
            std::vector<double> w(matrix_size);
            
            double orfac = 1e-6;
            int32_t ione = 1;
            
            int32_t m;
            int32_t nz;
            double d1;
            int32_t info;

            FORTRAN(pzhegvx)(&ione, "V", "I", "U", &matrix_size, a, &ione, &ione, desca, b, &ione, &ione, descb, &d1, &d1, 
                             &ione, &nevec, &abstol_, &m, &nz, &w[0], &orfac, z, &ione, &ione, descz, &work[0], &work_sizes[0], 
                             &rwork[0], &work_sizes[1], &iwork[0], &work_sizes[2], &ifail[0], &iclustr[0], &gap[0], &info, 
                             (int32_t)1, (int32_t)1, (int32_t)1); 

            if (info)
            {
                if ((info / 2) % 2)
                {
                    std::stringstream s;
                    s << "eigenvectors corresponding to one or more clusters of eigenvalues" << std::endl  
                      << "could not be reorthogonalized because of insufficient workspace" << std::endl;

                    int k = num_ranks_row_ * num_ranks_col_;
                    for (int i = 0; i < num_ranks_row_ * num_ranks_col_ - 1; i++)
                    {
                        if ((iclustr[2 * i + 1] != 0) && (iclustr[2 * (i + 1)] == 0))
                        {
                            k = i + 1;
                            break;
                        }
                    }
                   
                    s << "number of eigenvalue clusters : " << k << std::endl;
                    for (int i = 0; i < k; i++) s << iclustr[2 * i] << " : " << iclustr[2 * i + 1] << std::endl; 
                    error_local(__FILE__, __LINE__, s);
                }

                std::stringstream s;
                s << "pzhegvx returned " << info; 
                error_local(__FILE__, __LINE__, s);
            }

            if ((m != nevec) || (nz != nevec))
                error_local(__FILE__, __LINE__, "Not all eigen-vectors or eigen-values are found.");

            memcpy(eval, &w[0], nevec * sizeof(double));

        }
        #endif

};

class generalized_evp_elpa: public generalized_evp
{
    private:
        
        int32_t block_size_;
        int32_t na_rows_;
        int32_t num_ranks_row_;
        int32_t rank_row_;
        int32_t na_cols_;
        int32_t num_ranks_col_;
        int32_t rank_col_;
        int blacs_context_;
        MPI_Comm comm_row_;
        MPI_Comm comm_col_;
        MPI_Comm comm_all_;

    public:
        
        generalized_evp_elpa(int32_t block_size__, int32_t na_rows__, int32_t num_ranks_row__, int32_t rank_row__,
                             int32_t na_cols__, int32_t num_ranks_col__, int32_t rank_col__, int blacs_context__, 
                             MPI_Comm comm_row__, MPI_Comm comm_col__, MPI_Comm comm_all__) 
            : block_size_(block_size__), 
              na_rows_(na_rows__), 
              num_ranks_row_(num_ranks_row__), 
              rank_row_(rank_row__),
              na_cols_(na_cols__), 
              num_ranks_col_(num_ranks_col__), 
              rank_col_(rank_col__),
              blacs_context_(blacs_context__), 
              comm_row_(comm_row__), 
              comm_col_(comm_col__), 
              comm_all_(comm_all__)

        {
        }
        
        #ifdef _ELPA_
        void solve(int32_t matrix_size, int32_t nevec, double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                   double* eval, double_complex* z, int32_t ldz)
        {

            assert(nevec <= matrix_size);

            int32_t mpi_comm_rows = MPI_Comm_c2f(comm_row_);
            int32_t mpi_comm_cols = MPI_Comm_c2f(comm_col_);
            int32_t mpi_comm_all = MPI_Comm_c2f(comm_all_);

            sirius::Timer *t;

            t = new sirius::Timer("elpa::ort");
            FORTRAN(elpa_cholesky_complex)(&matrix_size, b, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
            FORTRAN(elpa_invert_trm_complex)(&matrix_size, b, &ldb, &block_size_, &mpi_comm_rows, &mpi_comm_cols);
       
            mdarray<double_complex, 2> tmp1(na_rows_, na_cols_);
            mdarray<double_complex, 2> tmp2(na_rows_, na_cols_);

            FORTRAN(elpa_mult_ah_b_complex)("U", "L", &matrix_size, &matrix_size, b, &ldb, a, &lda, &block_size_, 
                                            &mpi_comm_rows, &mpi_comm_cols, tmp1.get_ptr(), &na_rows_, (int32_t)1, 
                                            (int32_t)1);

            int32_t descc[9];
            linalg<scalapack>::descinit(descc, matrix_size, matrix_size, block_size_, block_size_, 0, 0, 
                                        blacs_context_, lda);

            linalg<scalapack>::pztranc(matrix_size, matrix_size, double_complex(1, 0), tmp1.get_ptr(), 1, 1, descc, 
                                       double_complex(0, 0), tmp2.get_ptr(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("U", "U", &matrix_size, &matrix_size, b, &ldb, tmp2.get_ptr(), &na_rows_, 
                                            &block_size_, &mpi_comm_rows, &mpi_comm_cols, a, &lda, (int32_t)1, 
                                            (int32_t)1);

            linalg<scalapack>::pztranc(matrix_size, matrix_size, double_complex(1, 0), a, 1, 1, descc, double_complex(0, 0), 
                                       tmp1.get_ptr(), 1, 1, descc);

            for (int i = 0; i < na_cols_; i++)
            {
                int32_t n_col = linalg<scalapack>::indxl2g(i + 1, block_size_, rank_col_, 0, num_ranks_col_);
                int32_t n_row = linalg<scalapack>::numroc(n_col, block_size_, rank_row_, 0, num_ranks_row_);
                for (int j = n_row; j < na_rows_; j++) 
                {
                    assert(j < na_rows_);
                    assert(i < na_cols_);
                    a[j + i * lda] = tmp1(j, i);
                }
            }
            delete t;
            
            t = new sirius::Timer("elpa::diag");
            std::vector<double> w(matrix_size);
            FORTRAN(elpa_solve_evp_complex_2stage)(&matrix_size, &nevec, a, &lda, &w[0], tmp1.get_ptr(), &na_rows_, 
                                                   &block_size_, &mpi_comm_rows, &mpi_comm_cols, &mpi_comm_all);
            delete t;

            t = new sirius::Timer("elpa::bt");
            linalg<scalapack>::pztranc(matrix_size, matrix_size, double_complex(1, 0), b, 1, 1, descc, double_complex(0, 0), 
                                       tmp2.get_ptr(), 1, 1, descc);

            FORTRAN(elpa_mult_ah_b_complex)("L", "N", &matrix_size, &nevec, tmp2.get_ptr(), &na_rows_, tmp1.get_ptr(), 
                                            &na_rows_, &block_size_, &mpi_comm_rows, &mpi_comm_cols, z, &ldz, 
                                            (int32_t)1, (int32_t)1);
            delete t;

            memcpy(eval, &w[0], nevec * sizeof(double));
        }
        #endif
};

class generalized_evp_magma: public generalized_evp
{
    private:

    public:
        generalized_evp_magma()
        {
        }

        #ifdef _MAGMA_
        void solve(int32_t matrix_size, int32_t nevec, double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                   double* eval, double_complex* z, int32_t ldz)
        {

            assert(nevec <= matrix_size);
            
            magma_zhegvdx_2stage_wrapper(matrix_size, nevec, a, lda, b, ldb, eval);
            
            for (int i = 0; i < nevec; i++) memcpy(&z[ldz * i], &a[lda * i], matrix_size * sizeof(double_complex));
        }
        #endif
};

#endif // __LINALG_H__


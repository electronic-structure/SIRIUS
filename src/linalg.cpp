#include "linalg.h"

template<> 
void blas<cpu>::gemm<double>(int transa, int transb, int32_t m, int32_t n, int32_t k, double alpha, 
                             double* a, int32_t lda, double* b, int32_t ldb, double beta, double* c, 
                             int32_t ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(dgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<> 
void blas<cpu>::gemm<double>(int transa, int transb, int32_t m, int32_t n, int32_t k, double* a, int32_t lda, 
                             double* b, int32_t ldb, double* c, int32_t ldc)
{
    gemm(transa, transb, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
}

template<> 
void blas<cpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex alpha, double_complex* a, int32_t lda, double_complex* b, 
                                     int32_t ldb, double_complex beta, double_complex* c, int32_t ldc)
{
    const char *trans[] = {"N", "T", "C"};

    FORTRAN(zgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<> 
void blas<cpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex* c, int32_t ldc)
{
    gemm(transa, transb, m, n, k, complex_one, a, lda, b, ldb, complex_zero, c, ldc);
}

template<> 
void blas<cpu>::hemm<double_complex>(int side, int uplo, int32_t m, int32_t n, double_complex alpha, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex beta, double_complex* c, int32_t ldc)
{
    const char *sidestr[] = {"L", "R"};
    const char *uplostr[] = {"U", "L"};
    FORTRAN(zhemm)(sidestr[side], uplostr[uplo], &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, (int32_t)1, 
                   (int32_t)1);
}

template<>
void blas<cpu>::gemv<double_complex>(int trans, int32_t m, int32_t n, double_complex alpha, double_complex* a, 
                                     int32_t lda, double_complex* x, int32_t incx, double_complex beta, 
                                     double_complex* y, int32_t incy)
{
    const char *trans_c[] = {"N", "T", "C"};

    FORTRAN(zgemv)(trans_c[trans], &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, 1);
}

#ifdef _GPU_
double_complex blas<gpu>::zone = double_complex(1, 0);
double_complex blas<gpu>::zzero = double_complex(0, 0);

template<> 
void blas<gpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* alpha, double_complex* a, int32_t lda, double_complex* b, 
                                     int32_t ldb, double_complex* beta, double_complex* c, int32_t ldc)
{
    cublas_zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<> 
void blas<gpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                                     double_complex* a, int32_t lda, double_complex* b, int32_t ldb, 
                                     double_complex* c, int32_t ldc)
{
    cublas_zgemm(transa, transb, m, n, k, &zone, a, lda, b, ldb, &zzero, c, ldc);
}
#endif

#ifdef _SCALAPACK_
int linalg<scalapack>::cyclic_block_size_ = -1;

//== template<> 
//== void pblas<cpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
//==                                       double_complex* a, int32_t lda, double_complex* b, int32_t ldb, double_complex beta, 
//==                                       double_complex* c, int32_t ldc, int blacs_context)
//== {
//==     const char *trans[] = {"N", "T", "C"};
//==     int nrow_a = (transa == 0) ? m : k;
//==     int ncol_a = (transa == 0) ? k : m;
//==     int nrow_b = (transb == 0) ? k : n;
//==     int ncol_b = (transb == 0) ? n : k;
//==     int nrow_c = m;
//==     int ncol_c = n;
//== 
//==     int block_size = linalg<scalapack>::cyclic_block_size();
//== 
//==     int desca[9];
//==     linalg<scalapack>::descinit(desca, nrow_a, ncol_a, block_size, block_size, 0, 0, blacs_context, lda);
//==     
//==     int descb[9];
//==     linalg<scalapack>::descinit(descb, nrow_b, ncol_b, block_size, block_size, 0, 0, blacs_context, ldb);
//==     
//==     int descc[9];
//==     linalg<scalapack>::descinit(descc, nrow_c, ncol_c, block_size, block_size, 0, 0, blacs_context, ldc);
//== 
//==     int32_t ione = 1;
//==     FORTRAN(pzgemm)(trans[transa], trans[transb], &m, &n, &k, &alpha, a, &ione, &ione, desca, b, &ione, &ione, descb,
//==                     &beta, c, &ione, &ione, descc, 1, 1);
//== }

template<> 
void pblas<cpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
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
void pblas<cpu>::gemm<double_complex>(int transa, int transb, int32_t m, int32_t n, int32_t k, double_complex alpha, 
                                      dmatrix<double_complex>& a, dmatrix<double_complex>& b, double_complex beta, 
                                      dmatrix<double_complex>& c)
{
    pblas<cpu>::gemm<double_complex>(transa, transb, m, n, k, alpha, a, 0, 0, b, 0, 0, beta, c, 0, 0);
}

#endif


template<> 
int linalg<lapack>::gesv<double>(int32_t n, int32_t nrhs, double* a, int32_t lda, double* b, int32_t ldb)
{
    int32_t info;
    std::vector<int32_t> ipiv(n);
    FORTRAN(dgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);
    return info;
}

template<> 
int linalg<lapack>::gesv<double_complex>(int32_t n, int32_t nrhs, double_complex* a, int32_t lda, 
                                         double_complex* b, int32_t ldb)
{
    int32_t info;
    std::vector<int32_t> ipiv(n);
    FORTRAN(zgesv)(&n, &nrhs, a, &lda, &ipiv[0], b, &ldb, &info);
    return info;
}

template<> 
int linalg<lapack>::gtsv<double>(int32_t n, int32_t nrhs, double* dl, double* d, double* du, double* b, int32_t ldb)
{
    int info;
    FORTRAN(dgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;
}

template<> 
int linalg<lapack>::gtsv<double_complex>(int32_t n, int32_t nrhs, double_complex* dl, double_complex* d, double_complex* du, 
                                         double_complex* b, int32_t ldb)
{
    int32_t info;                   
    FORTRAN(zgtsv)(&n, &nrhs, dl, d, du, b, &ldb, &info);
    return info;               
}

template<> 
int linalg<lapack>::getrf<double>(int32_t m, int32_t n, double* a, int32_t lda, int32_t* ipiv)
{
    int32_t info;
    FORTRAN(dgetrf)(&m, &n, a, &lda, ipiv, &info);
    return info;
}
    
template<> 
int linalg<lapack>::getrf<double_complex>(int32_t m, int32_t n, double_complex* a, int32_t lda, int32_t* ipiv)
{
    int32_t info;
    FORTRAN(zgetrf)(&m, &n, a, &lda, ipiv, &info);
    return info;
}

template<> 
int linalg<lapack>::getri<double>(int32_t n, double* a, int32_t lda, int32_t* ipiv, double* work, int32_t lwork)
{
    int32_t info;
    FORTRAN(dgetri)(&n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}

template<> 
int linalg<lapack>::getri<double_complex>(int32_t n, double_complex* a, int32_t lda, int32_t* ipiv, double_complex* work, 
                                          int32_t lwork)
{
    int32_t info;
    FORTRAN(zgetri)(&n, a, &lda, ipiv, work, &lwork, &info);
    return info;
}


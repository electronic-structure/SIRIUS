/** \file rocsolver.hpp
 *
 *  \brief Interface to CUDA eigen-solver library.
 *
 */

#ifndef __ROCSOLVER_HPP__
#define __ROCSOLVER_HPP__

#include "acc.hpp"
#include "acc_blas_api.hpp"
#include <rocsolver.h>
#include <rocblas.h>
#include <unistd.h>

namespace rocsolver {

#define CALL_ROCSOLVER(func__, args__)                                                                                 \
    {                                                                                                                  \
        rocblas_status status = func__ args__;                                                                         \
        if (status != rocblas_status::rocblas_status_success) {                                                        \
            char nm[1024];                                                                                             \
            gethostname(nm, 1024);                                                                                     \
            printf("hostname: %s\n", nm);                                                                              \
            printf("Error in %s at line %i of file %s: %s\n", #func__, __LINE__, __FILE__,                             \
                   rocblas_status_to_string(status));                                                                  \
            stack_backtrace();                                                                                         \
        }                                                                                                              \
    }

::acc::blas::handle_t& rocsolver_handle();

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n,
       T* A, int lda, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_dsyevd, (handle, evect, uplo, n, A, lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n,
       T* A, int lda, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_ssyevd, (handle, evect, uplo, n, A, lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n,
       std::complex<T>* A, int lda, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_zheevd,
                   (handle, evect, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n,
       std::complex<T>* A, int lda, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_cheevd,
                   (handle, evect, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda, D, E, info));
}

/// SYmmetric or HEmrmitian GENERALIZED eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_dsygvd, (handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info));
}

/// SYmmetric or HEmrmitian GENERALIZED eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_ssygvd, (handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info));
}

/// SYmmetric or HEmrmitian GENERALIZED eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<T>* A, int lda, std::complex<T>* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_chegvd, (handle, itype, evect, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda,
                                      reinterpret_cast<rocblas_float_complex*>(B), ldb, D, E, info));
}

/// SYmmetric or HEmrmitian GENERALIZED eigenvalue problem
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<T>* A, int lda, std::complex<T>* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_zhegvd, (handle, itype, evect, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda,
                                      reinterpret_cast<rocblas_double_complex*>(B), ldb, D, E, info));
}

} // namespace rocsolver

#endif

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file rocsolver.hpp
 *
 *  \brief Contains implementation of rocsolver wrappers
 */

#ifndef __ROCSOLVER_HPP__
#define __ROCSOLVER_HPP__

#include <rocsolver/rocsolver.h>
#include <rocblas/rocblas.h>
#include <unistd.h>
#include "acc.hpp"
#include "acc_blas_api.hpp"
#include "core/rte/rte.hpp"

namespace sirius {

namespace acc {

/// Interface to ROCM eigensolver.
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
            acc::stack_backtrace();                                                                                    \
        }                                                                                                              \
    }

acc::blas_api::handle_t&
rocsolver_handle();

inline rocblas_operation
get_rocblas_operation(char trans)
{
    rocblas_operation op{rocblas_operation::rocblas_operation_none};
    switch (trans) {
        case 'n':
        case 'N':
            op = rocblas_operation::rocblas_operation_none;
            break;
        case 't':
        case 'T':
            op = rocblas_operation::rocblas_operation_transpose;
            break;
        case 'h':
        case 'H':
            op = rocblas_operation::rocblas_operation_conjugate_transpose;
            break;
        default:
            RTE_THROW("invalid tranpose op.")
    }

    return op;
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem | double
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, T* A, int lda, T* D, T* E,
        int* info)
{
    CALL_ROCSOLVER(rocsolver_dsyevd, (handle, evect, uplo, n, A, lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem | float
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, T* A, int lda, T* D, T* E,
        int* info)
{
    CALL_ROCSOLVER(rocsolver_ssyevd, (handle, evect, uplo, n, A, lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem | complex double
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, std::complex<T>* A, int lda,
        T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_zheevd,
                   (handle, evect, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem | complex float
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, std::complex<T>* A, int lda,
        T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_cheevd,
                   (handle, evect, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda, D, E, info));
}

/// _sy_mmetric or _he_rmitian GENERALIZED eigenvalue problem | double
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_dsygvd, (handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info));
}

/// _sy_mmetric or _he_rmitian GENERALIZED eigenvalue problem | float
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_ssygvd, (handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, info));
}

/// _sy_mmetric or _he_rmitian GENERALIZED eigenvalue problem | complex float
template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<T>* A, int lda, std::complex<T>* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_chegvd, (handle, itype, evect, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda,
                                      reinterpret_cast<rocblas_float_complex*>(B), ldb, D, E, info));
}

/// _sy_mmetric or _he_rmitian GENERALIZED eigenvalue problem | complex double
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<T>* A, int lda, std::complex<T>* B, int ldb, T* D, T* E, int* info)
{
    CALL_ROCSOLVER(rocsolver_zhegvd, (handle, itype, evect, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda,
                                      reinterpret_cast<rocblas_double_complex*>(B), ldb, D, E, info));
}

#if (ROCSOLVER_VERSION_MAJOR > 3) || ((ROCSOLVER_VERSION_MAJOR == 3) && (ROCSOLVER_VERSION_MINOR >= 19))
/// x versions
/// -----------------------------------------------------------------------------------------------------------------
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syheevx(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, T* A, int lda, int il, int iu,
        double abstol, int* nev, T* D, T* Z, int ldz, int* ifail, int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_dsyevx(handle, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, vl, vu, il, iu, abstol, nev,
                     D, Z, ldz, ifail, info);
}

template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syheevx(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, T* A, int lda, int il, int iu,
        double abstol, int* nev, T* D, T* Z, int ldz, int* ifail, int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_ssyevx(handle, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, vl, vu, il, iu, abstol, nev,
                     D, Z, ldz, ifail, info);
}

/// Hermitian | complex double
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syheevx(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, std::complex<double>* A,
        int lda, int il, int iu, double abstol, int* nev, T* D, std::complex<double>* Z, int ldz, int* ifail, int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_zheevx(handle, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                     reinterpret_cast<rocblas_double_complex*>(A), lda, vl, vu, il, iu, abstol, nev, D,
                     reinterpret_cast<rocblas_double_complex*>(Z), ldz, ifail, info);
}

template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syheevx(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, std::complex<float>* A,
        int lda, int il, int iu, double abstol, int* nev, T* D, std::complex<float>* Z, int ldz, int* ifail, int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_cheevx(handle, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                     reinterpret_cast<rocblas_float_complex*>(A), lda, vl, vu, il, iu, abstol, nev, D,
                     reinterpret_cast<rocblas_float_complex*>(Z), ldz, ifail, info);
}

/// x versions
/// -----------------------------------------------------------------------------------------------------------------
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syhegvx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, int il, int iu, double abstol, int* nev, T* D, T* Z, int ldz, int* ifail,
        int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_dsygvx(handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, B, ldb, vl, vu, il,
                     iu, abstol, nev, D, Z, ldz, ifail, info);
}

template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syhegvx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, int il, int iu, double abstol, int* nev, T* D, T* Z, int ldz, int* ifail,
        int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_ssygvx(handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, B, ldb, vl, vu, il,
                     iu, abstol, nev, D, Z, ldz, ifail, info);
}

/// Hermitian | complex double
template <class T>
std::enable_if_t<std::is_same<T, double>::value>
syhegvx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<double>* A, int lda, std::complex<double>* B, int ldb, int il, int iu, double abstol, int* nev,
        T* D, std::complex<double>* Z, int ldz, int* ifail, int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_zhegvx(handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                     reinterpret_cast<rocblas_double_complex*>(A), lda, reinterpret_cast<rocblas_double_complex*>(B),
                     ldb, vl, vu, il, iu, abstol, nev, D, reinterpret_cast<rocblas_double_complex*>(Z), ldz, ifail,
                     info);
}

template <class T>
std::enable_if_t<std::is_same<T, float>::value>
syhegvx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<float>* A, int lda, std::complex<float>* B, int ldb, int il, int iu, double abstol, int* nev, T* D,
        std::complex<float>* Z, int ldz, int* ifail, int* info)
{
    double vl, vu{0}; // ingored if erange = erange_index
    rocsolver_chegvx(handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                     reinterpret_cast<rocblas_float_complex*>(A), lda, reinterpret_cast<rocblas_float_complex*>(B), ldb,
                     vl, vu, il, iu, abstol, nev, D, reinterpret_cast<rocblas_float_complex*>(Z), ldz, ifail, info);
}
#endif // rocsolver >=5.3.0

/// Linear Solvers
void
zgetrs(rocblas_handle handle, char trans, int n, int nrhs, acc_complex_double_t* A, int lda, const int* devIpiv,
       acc_complex_double_t* B, int ldb);

void
zgetrf(rocblas_handle handle, int m, int n, acc_complex_double_t* A, int* devIpiv, int lda, int* devInfo);

} // namespace rocsolver

} // namespace acc

} // namespace sirius

#endif

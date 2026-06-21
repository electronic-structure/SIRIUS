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
#ifdef SIRIUS_ROCM
#ifndef __ROCSOLVER_HPP__
#define __ROCSOLVER_HPP__

#include <rocsolver/rocsolver.h>
#include <rocblas/rocblas.h>
#include <unistd.h>
#include "acc.hpp"
#include "acc_blas_api.hpp"
#include "core/rte/rte.hpp"
#include "core/memory.hpp"

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
            RTE_THROW("invalid transpose op.")
    }

    return op;
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem | double,float
template <class T>
void
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, T* A, int lda, T* D, T* E,
        int* info)
{
    if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_dsyevd, (handle, evect, uplo, n, A, lda, D, E, info));
    } else if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_ssyevd, (handle, evect, uplo, n, A, lda, D, E, info));
    }
}

/// _sy_mmetric or _he_rmitian STANDARD eigenvalue problem | complex double, float
template <class T>
void
syheevd(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, std::complex<T>* A, int lda,
        T* D, T* E, int* info)
{
    if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_zheevd,
                       (handle, evect, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda, D, E, info));

    } else if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_cheevd,
                       (handle, evect, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda, D, E, info));
    }
}

/// _sy_mmetric or _he_rmitian GENERALIZED eigenvalue problem | double, float
template <class T>
int
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, T* D, T* E)
{
    auto& mpd   = get_memory_pool(memory_t::device);
    auto d_info = mpd.get_unique_ptr<int>(1);
    if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_dsygvd, (handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, d_info.get()));
    } else if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_ssygvd, (handle, itype, evect, uplo, n, A, lda, B, ldb, D, E, d_info.get()));
    }
    int info;
    acc::copyout(&info, d_info.get(), 1);
    return info;
}

/// _sy_mmetric or _he_rmitian GENERALIZED eigenvalue problem | complex double, float
template <class T>
int
syhegvd(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<T>* A, int lda, std::complex<T>* B, int ldb, T* D, T* E)
{
    auto& mpd   = get_memory_pool(memory_t::device);
    auto d_info = mpd.get_unique_ptr<int>(1);
    if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_zhegvd, (handle, itype, evect, uplo, n, reinterpret_cast<rocblas_double_complex*>(A),
                                          lda, reinterpret_cast<rocblas_double_complex*>(B), ldb, D, E, d_info.get()));
    } else if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_chegvd, (handle, itype, evect, uplo, n, reinterpret_cast<rocblas_float_complex*>(A),
                                          lda, reinterpret_cast<rocblas_float_complex*>(B), ldb, D, E, d_info.get()));
    }
    int info;
    acc::copyout(&info, d_info.get(), 1);
    return info;
}

#if (ROCSOLVER_VERSION_MAJOR > 3) || ((ROCSOLVER_VERSION_MAJOR == 3) && (ROCSOLVER_VERSION_MINOR >= 19))
/// x versions
/// -----------------------------------------------------------------------------------------------------------------
struct rocsolver_evx_return_type
{
    ///  If info = 0, the first nev elements of ifail are zero. Otherwise,
    ///  contains the indices of those eigenvectors that failed to converge. Not
    ///  referenced if evect is rocblas_evect_none.
    std::vector<int> ifail;
    /// If info = 0, successful exit. If info = i > 0, the algorithm did not
    /// converge. i columns of Z did not converge.
    int info;
    /// The total number of eigenvalues found. If erange is rocblas_erange_all,
    /// nev = n. If erange is rocblas_erange_index, nev = iu - il + 1.
    /// Otherwise, 0 <= nev <= n.
    int nev;
};
template <class T>
rocsolver_evx_return_type
syheevx(rocblas_handle handle, const rocblas_evect evect, rocblas_erange erange, const rocblas_fill uplo, int n, T* A,
        int lda, double vl, double vu, int il, int iu, double abstol, T* D, T* Z, int ldz)
{
    auto& mpd    = get_memory_pool(memory_t::device);
    auto d_info  = mpd.get_unique_ptr<int>(1);
    auto d_nev   = mpd.get_unique_ptr<int>(1);
    auto d_ifail = mpd.get_unique_ptr<int>(n);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_ssyevx, (handle, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, vl, vu,
                                          il, iu, abstol, d_nev.get(), D, Z, ldz, d_ifail.get(), d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_dsyevx, (handle, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, vl, vu,
                                          il, iu, abstol, d_nev.get(), D, Z, ldz, d_ifail.get(), d_info.get()));
    }

    rocsolver_evx_return_type ret;

    acc::copyout(&ret.info, d_info.get(), 1);
    acc::copyout(&ret.nev, d_nev.get(), 1);

    if (evect != rocblas_evect_none) {
        ret.ifail = std::vector<int>(n);
        acc::copyout(ret.ifail.data(), d_ifail.get(), n);
    }

    return ret;
}

/// Hermitian | complex double
template <class T>
rocsolver_evx_return_type
syheevx(rocblas_handle handle, const rocblas_evect evect, const rocblas_fill uplo, int n, std::complex<T>* A, int lda,
        double vu, double vl, int il, int iu, double abstol, T* D, std::complex<T>* Z, int ldz)
{
    auto& mpd    = get_memory_pool(memory_t::device);
    auto d_info  = mpd.get_unique_ptr<int>(1);
    auto d_nev   = mpd.get_unique_ptr<int>(1);
    auto d_ifail = mpd.get_unique_ptr<int>(n);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_cheevx,
                       (handle, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                        reinterpret_cast<rocblas_float_complex*>(A), lda, vl, vu, il, iu, abstol, d_nev.get(), D,
                        reinterpret_cast<rocblas_float_complex*>(Z), ldz, d_ifail.get(), d_info.get()));

    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_zheevx,
                       (handle, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                        reinterpret_cast<rocblas_double_complex*>(A), lda, vl, vu, il, iu, abstol, d_nev.get(), D,
                        reinterpret_cast<rocblas_double_complex*>(Z), ldz, d_ifail.get(), d_info.get()));
    }
    rocsolver_evx_return_type ret;

    acc::copyout(&ret.info, d_info.get(), 1);
    acc::copyout(&ret.nev, d_nev.get(), 1);

    if (evect != rocblas_evect_none) {
        ret.ifail = std::vector<int>(n);
        acc::copyout(ret.ifail.data(), d_ifail.get(), n);
    }

    return ret;
}

/// x versions
/// -----------------------------------------------------------------------------------------------------------------
template <class T>
rocsolver_evx_return_type
syhegvx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        T* A, int lda, T* B, int ldb, double vu, double vl, int il, int iu, double abstol, T* D, T* Z, int ldz)
{
    auto& mpd    = get_memory_pool(memory_t::device);
    auto d_info  = mpd.get_unique_ptr<int>(1);
    auto d_nev   = mpd.get_unique_ptr<int>(1);
    auto d_ifail = mpd.get_unique_ptr<int>(n);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_ssygvx,
                       (handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, B, ldb, vl, vu, il,
                        iu, abstol, d_nev.get(), D, Z, ldz, d_ifail.get(), d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_dsygvx,
                       (handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n, A, lda, B, ldb, vl, vu, il,
                        iu, abstol, d_nev.get(), D, Z, ldz, d_ifail.get(), d_info.get()));
    }

    rocsolver_evx_return_type ret;

    acc::copyout(&ret.info, d_info.get(), 1);
    acc::copyout(&ret.nev, d_nev.get(), 1);

    if (evect != rocblas_evect_none) {
        ret.ifail = std::vector<int>(n);
        acc::copyout(ret.ifail.data(), d_ifail.get(), n);
    }

    return ret;
}

/// Hermitian | complex double
template <class T>
rocsolver_evx_return_type
syhegvx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_fill uplo, int n,
        std::complex<double>* A, int lda, std::complex<double>* B, int ldb, double vl, double vu, int il, int iu,
        double abstol, T* D, std::complex<double>* Z, int ldz)
{
    auto& mpd    = get_memory_pool(memory_t::device);
    auto d_info  = mpd.get_unique_ptr<int>(1);
    auto d_nev   = mpd.get_unique_ptr<int>(1);
    auto d_ifail = mpd.get_unique_ptr<int>(n);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_chegvx,
                       (handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                        reinterpret_cast<rocblas_float_complex*>(A), lda, reinterpret_cast<rocblas_float_complex*>(B),
                        ldb, vl, vu, il, iu, abstol, d_nev.get(), D, reinterpret_cast<rocblas_float_complex*>(Z), ldz,
                        d_ifail.get(), d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_zhegvx,
                       (handle, itype, evect, rocblas_erange::rocblas_erange_index, uplo, n,
                        reinterpret_cast<rocblas_double_complex*>(A), lda, reinterpret_cast<rocblas_double_complex*>(B),
                        ldb, vl, vu, il, iu, abstol, d_nev.get(), D, reinterpret_cast<rocblas_double_complex*>(Z), ldz,
                        d_ifail.get(), d_info.get()));
    }

    rocsolver_evx_return_type ret;

    acc::copyout(&ret.info, d_info.get(), 1);
    acc::copyout(&ret.nev, d_nev.get(), 1);

    if (evect != rocblas_evect_none) {
        ret.ifail = std::vector<int>(n);
        acc::copyout(ret.ifail.data(), d_ifail.get(), n);
    }

    return ret;
}

#endif // rocsolver >=3.19.0

// rocm 7.2.0
#if (ROCSOLVER_VERSION_MAJOR > 3) || ((ROCSOLVER_VERSION_MAJOR == 3) && (ROCSOLVER_VERSION_MINOR >= 30))
struct rocblas_evdx_return_type
{
    int info;
    int nev;
};
/// dx versions (divide-and-conquer + subset selection)
/// -----------------------------------------------------------------------------------------------------------------
template <class T>
rocblas_evdx_return_type
syheevdx(rocblas_handle handle, const rocblas_evect evect, const rocblas_erange erange, const rocblas_fill uplo, int n,
         T* A, int lda, T vl, T vu, int il, int iu, T* w, T* E, int ldz)
{
    int info;
    int nev;

    auto& mpd   = get_memory_pool(memory_t::device);
    auto d_info = mpd.get_unique_ptr<int>(1);
    auto d_nev  = mpd.get_unique_ptr<int>(1);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_ssyevdx,
                       (handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, d_nev.get(), w, E, ldz, d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_dsyevdx,
                       (handle, evect, erange, uplo, n, A, lda, vl, vu, il, iu, d_nev.get(), w, E, ldz, d_info.get()));
    }
    acc::copyout(&info, d_info.get(), 1);
    acc::copyout(&nev, d_nev.get(), 1);

    return {.info = info, .nev = nev};
}

/// Hermitian | complex double
template <class T>
rocblas_evdx_return_type
syheevdx(rocblas_handle handle, const rocblas_evect evect, const rocblas_erange erange, const rocblas_fill uplo, int n,
         std::complex<T>* A, int lda, T vl, T vu, int il, int iu, T* w, std::complex<T>* E, int ldz)
{
    int info;
    int nev;

    auto& mpd   = get_memory_pool(memory_t::device);
    auto d_info = mpd.get_unique_ptr<int>(1);
    auto d_nev  = mpd.get_unique_ptr<int>(1);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_cheevdx,
                       (handle, evect, erange, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda, vl, vu, il,
                        iu, d_nev.get(), w, reinterpret_cast<rocblas_float_complex*>(E), ldz, d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_zheevdx,
                       (handle, evect, erange, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda, vl, vu, il,
                        iu, d_nev.get(), w, reinterpret_cast<rocblas_double_complex*>(E), ldz, d_info.get()));
    }
    acc::copyout(&info, d_info.get(), 1);
    acc::copyout(&nev, d_nev.get(), 1);

    return {.info = info, .nev = nev};
}

/// General eigenproblem
template <class T>
rocblas_evdx_return_type
syhegvdx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_erange erange,
         const rocblas_fill uplo, int n, T* A, int lda, T* B, int ldb, T vl, T vu, int il, int iu, T* w, T* Z, int ldz)
{
    int info;
    int nev;

    auto& mpd   = get_memory_pool(memory_t::device);
    auto d_info = mpd.get_unique_ptr<int>(1);
    auto d_nev  = mpd.get_unique_ptr<int>(1);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_ssygvdx, (handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                                           d_nev.get(), w, Z, ldz, d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_dsygvdx, (handle, itype, evect, erange, uplo, n, A, lda, B, ldb, vl, vu, il, iu,
                                           d_nev.get(), w, Z, ldz, d_info.get()));
    }

    acc::copyout(&info, d_info.get(), 1);
    acc::copyout(&nev, d_nev.get(), 1);

    return {.info = info, .nev = nev};
}

// complex
template <class T>
rocblas_evdx_return_type
syhegvdx(rocblas_handle handle, const rocblas_eform itype, const rocblas_evect evect, const rocblas_erange erange,
         const rocblas_fill uplo, int n, std::complex<T>* A, int lda, std::complex<T>* B, int ldb, T vl, T vu, int il,
         int iu, T* w, std::complex<T>* Z, int ldz)
{
    int info;
    int nev;

    auto& mpd   = get_memory_pool(memory_t::device);
    auto d_info = mpd.get_unique_ptr<int>(1);
    auto d_nev  = mpd.get_unique_ptr<int>(1);

    if constexpr (std::is_same_v<T, float>) {
        CALL_ROCSOLVER(rocsolver_chegvdx,
                       (handle, itype, evect, erange, uplo, n, reinterpret_cast<rocblas_float_complex*>(A), lda,
                        reinterpret_cast<rocblas_float_complex*>(B), ldb, vl, vu, il, iu, d_nev.get(), w,
                        reinterpret_cast<rocblas_float_complex*>(Z), ldz, d_info.get()));
    } else if constexpr (std::is_same_v<T, double>) {
        CALL_ROCSOLVER(rocsolver_zhegvdx,
                       (handle, itype, evect, erange, uplo, n, reinterpret_cast<rocblas_double_complex*>(A), lda,
                        reinterpret_cast<rocblas_double_complex*>(B), ldb, vl, vu, il, iu, d_nev.get(), w,
                        reinterpret_cast<rocblas_double_complex*>(Z), ldz, d_info.get()));
    }

    acc::copyout(&info, d_info.get(), 1);
    acc::copyout(&nev, d_nev.get(), 1);

    return {.info = info, .nev = nev};
}

#endif // rocsolver >= 3.30.0 (aka rocm 7.0.1)

/// Linear Solvers
void
zgetrs(rocblas_handle handle, char trans, int n, int nrhs, acc_complex_double_t* A, int lda, const int* devIpiv,
       acc_complex_double_t* B, int ldb);

void
zgetrf(rocblas_handle handle, int m, int n, acc_complex_double_t* A, int* devIpiv, int lda, int* devInfo);

} // namespace rocsolver

} // namespace acc

} // namespace sirius

#endif // __ROCSOLVER_HPP__
#endif // SIRIUS_ROCM

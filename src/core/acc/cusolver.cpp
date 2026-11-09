/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef SIRIUS_CUDA
#include "cusolver_api.hpp"
#include "core/memory.hpp"

namespace sirius {
namespace acc {
namespace cusolver {

cusolverDnHandle_t&
cusolver_handle()
{
    static cusolverDnHandle_t handle;
    return handle;
}

void
create_handle()
{
    CALL_CUSOLVER(cusolverDnCreate, (&cusolver_handle()));
}

void
destroy_handle()
{
    CALL_CUSOLVER(cusolverDnDestroy, (cusolver_handle()));
}

template <typename T>
int
potrf(int n__, T* A__, int lda__)
{
    int64_t n   = n__;
    int64_t lda = lda__;
    size_t d_lwork{0};
    size_t h_lwork{0};

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    /* work size */
    CALL_CUSOLVER(cusolverDnXpotrf_bufferSize, (cusolver_handle(), NULL, uplo, n, type_wrapper<T>::type, A__, lda,
                                                type_wrapper<T>::type, &d_lwork, &h_lwork));

    auto d_work = get_memory_pool(memory_t::device).get_unique_ptr<T>(d_lwork);
    mdarray<int, 1> info({1});
    info.allocate(get_memory_pool(memory_t::device));

    void* hwork{nullptr};

    CALL_CUSOLVER(cusolverDnXpotrf,
                  (cusolver_handle(), NULL, uplo, n, type_wrapper<T>::type, A__, lda, type_wrapper<T>::type,
                   d_work.get(), d_lwork, hwork, h_lwork, info.at(memory_t::device)));

    info.copy_to(memory_t::host);
    return info[0];
}

template <typename T>
int
trtri(int n__, T* A__, int lda__)
{
    int64_t n   = n__;
    int64_t lda = lda__;
    size_t d_lwork{0};
    size_t h_lwork{0};
    /* work size */
    CALL_CUSOLVER(cusolverDnXtrtri_bufferSize, (cusolver_handle(), CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n,
                                                type_wrapper<T>::type, A__, lda, &d_lwork, &h_lwork));

    auto h_work = get_memory_pool(memory_t::host).get_unique_ptr<char>(h_lwork + 1);
    auto d_work = get_memory_pool(memory_t::device).get_unique_ptr<char>(d_lwork);
    mdarray<int, 1> info({1});
    info.allocate(get_memory_pool(memory_t::device));

    CALL_CUSOLVER(cusolverDnXtrtri,
                  (cusolver_handle(), CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n, type_wrapper<T>::type, A__, lda,
                   d_work.get(), d_lwork, h_work.get(), h_lwork, info.at(memory_t::device)));
    info.copy_to(memory_t::host);
    return info[0];
}

template int
potrf<float>(int n__, float* A__, int lda__);
template int
potrf<double>(int n__, double* A__, int lda__);
template int
potrf<std::complex<float>>(int n__, std::complex<float>* A__, int lda__);
template int
potrf<std::complex<double>>(int n__, std::complex<double>* A__, int lda__);
template int
trtri<float>(int n__, float* A__, int lda__);
template int
trtri<double>(int n__, double* A__, int lda__);
template int
trtri<std::complex<float>>(int n__, std::complex<float>* A__, int lda__);
template int
trtri<std::complex<double>>(int n__, std::complex<double>* A__, int lda__);

} // namespace cusolver
} // namespace acc
} // namespace sirius

#endif

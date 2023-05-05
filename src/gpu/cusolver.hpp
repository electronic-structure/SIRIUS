/** \file cusolver.hpp
 *
 *  \brief Interface to CUDA eigen-solver library.
 *
 */

#ifndef __CUSOLVER_HPP__
#define __CUSOLVER_HPP__

#include "acc.hpp"
#include "SDDK/memory.hpp"
#include <cusolverDn.h>

namespace cusolver {

inline void error_message(cusolverStatus_t status)
{
    switch (status) {
        case CUSOLVER_STATUS_NOT_INITIALIZED: {
            std::printf("the CUDA Runtime initialization failed\n");
            break;
        }
        case CUSOLVER_STATUS_ALLOC_FAILED: {
            std::printf("the resources could not be allocated\n");
            break;
        }
        case CUSOLVER_STATUS_ARCH_MISMATCH: {
            std::printf("the device only supports compute capability 2.0 and above\n");
            break;
        }
        case CUSOLVER_STATUS_INVALID_VALUE: {
            std::printf("An unsupported value or parameter was passed to the function\n");
            break;
        }
        case CUSOLVER_STATUS_EXECUTION_FAILED: {
            std::printf("The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n");
            break;
        }
        case CUSOLVER_STATUS_INTERNAL_ERROR: {
            std::printf("An internal cuSolver operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n");
            break;
        }
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: {
            std::printf("The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n");
            break;
        }
        default: {
            std::printf("cusolver status unknown\n");
        }
    }
}

#define CALL_CUSOLVER(func__, args__)                                                    \
{                                                                                        \
    cusolverStatus_t status;                                                             \
    if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                           \
        cusolver::error_message(status);                                                 \
        char nm[1024];                                                                   \
        gethostname(nm, 1024);                                                           \
        std::printf("hostname: %s\n", nm);                                               \
        std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
        stack_backtrace();                                                               \
    }                                                                                    \
}

cusolverDnHandle_t& cusolver_handle();
void create_handle();
void destroy_handle();

template <typename>
struct type_wrapper;

template<>
struct type_wrapper<float>
{
    static constexpr cudaDataType type = CUDA_R_32F;
};

template<>
struct type_wrapper<double>
{
    static constexpr cudaDataType type = CUDA_R_64F;
};

template<>
struct type_wrapper<std::complex<float>>
{
    static constexpr cudaDataType type = CUDA_C_32F;
};

template<>
struct type_wrapper<std::complex<double>>
{
    static constexpr cudaDataType type = CUDA_C_64F;
};

template <typename T>
int potrf(int n__, void* A__, int lda__)
{
    int64_t n = n__;
    int64_t lda = lda__;
    size_t d_lwork{0};
    size_t h_lwork{0};
    /* work size */
    CALL_CUSOLVER(cusolverDnXpotrf_bufferSize,
        (cusolver_handle(), NULL, CUBLAS_FILL_MODE_UPPER, n, type_wrapper<T>::type, A__, lda,
         type_wrapper<T>::type, &d_lwork, &h_lwork));

    auto h_work = get_memory_pool(sddk::memory_t::host).get_unique_ptr<char>(h_lwork + 1);
    auto d_work = get_memory_pool(sddk::memory_t::device).get_unique_ptr<char>(d_lwork);
    sddk::mdarray<int, 1> info(1);
    info.allocate(get_memory_pool(sddk::memory_t::device));

    CALL_CUSOLVER(cusolverDnXpotrf,
        (cusolver_handle(), NULL, CUBLAS_FILL_MODE_UPPER, n, type_wrapper<T>::type,
         A__, lda, type_wrapper<T>::type, d_work.get(), d_lwork, h_work.get(), h_lwork,
         info.at(sddk::memory_t::device)));
    info.copy_to(sddk::memory_t::host);
    return info[0];
}

template <typename T>
int trtri(int n__, void* A__, int lda__)
{
    int64_t n = n__;
    int64_t lda = lda__;
    size_t d_lwork{0};
    size_t h_lwork{0};
    /* work size */
    CALL_CUSOLVER(cusolverDnXtrtri_bufferSize,
        (cusolver_handle(), CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n, type_wrapper<T>::type, A__, lda,
         &d_lwork, &h_lwork));

    auto h_work = get_memory_pool(sddk::memory_t::host).get_unique_ptr<char>(h_lwork + 1);
    auto d_work = get_memory_pool(sddk::memory_t::device).get_unique_ptr<char>(d_lwork);
    sddk::mdarray<int, 1> info(1);
    info.allocate(get_memory_pool(sddk::memory_t::device));

    CALL_CUSOLVER(cusolverDnXtrtri,
        (cusolver_handle(), CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT, n, type_wrapper<T>::type,
         A__, lda, d_work.get(), d_lwork, h_work.get(), h_lwork, info.at(sddk::memory_t::device)));
    info.copy_to(sddk::memory_t::host);
    return info[0];
}

} // namespace cusolver

#endif

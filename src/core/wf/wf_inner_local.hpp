#ifndef __WF_INNER_LOCAL_HPP__
#define __WF_INNER_LOCAL_HPP__

#if defined(SIRIUS_ROCM) || defined(SIRIUS_CUDA)

#include <complex>
#include "core/acc/acc.hpp"
#include "core/memory.hpp"
#include "core/acc/acc_blas.hpp"
#include "core/typedefs.hpp"

#if defined(SIRIUS_ROCM)
#include <rocblas/rocblas.h>
#endif

#if defined(SIRIUS_CUDA)
#include <cublas_v2.h>
#endif

namespace sirius {

template <typename T, typename F>
void
inner_product_local_gpu(T const* wf1, int ld1, T const* wf2, int ld2, int n, int num_bands, int reduced,
                        std::vector<F>& result)
{
    auto mem_pool = get_memory_pool(memory_t::device);
    auto d_work   = mem_pool.get_unique_ptr<T>(num_bands);
    auto stream   = acc::blas::stream_handle(0);
    std::vector<T> result_local(num_bands, 0);

#if defined(SIRIUS_ROCM)
    CALL_GPU_BLAS(rocblas_zdotc_strided_batched, (stream, n, reinterpret_cast<const rocblas_double_complex*>(wf1), 1,
                                                  ld1, reinterpret_cast<const rocblas_double_complex*>(wf2), 1, ld2,
                                                  num_bands, reinterpret_cast<rocblas_double_complex*>(d_work.get())));
#elif defined(SIRIUS_CUDA)
    T beta{0};
    T alpha{1};
    // https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched
    CALL_GPU_BLAS(cublasZgemmStridedBatched,
                  (stream, CUBLAS_OP_C, CUBLAS_OP_N, 1, 1, n, reinterpret_cast<const cuDoubleComplex*>(&alpha),
                   reinterpret_cast<const cuDoubleComplex*>(wf1), n, ld1, reinterpret_cast<const cuDoubleComplex*>(wf2),
                   n, ld2, reinterpret_cast<const cuDoubleComplex*>(&beta),
                   reinterpret_cast<cuDoubleComplex*>(d_work.get()), 1, 1, num_bands));
#endif

    acc::copyout(result_local.data(), d_work.get(), num_bands);
    if constexpr (std::is_same_v<F, std::complex<double>>) {
        /* non Γ-point case */
    } else if constexpr (std::is_same_v<F, double>) {
        /* Γ-point case */
        // #if defined(SIRIUS_ROCM)
        //         rocblas_ddot_strided_batched(stream, n, reinterpret_cast<const rocblas_double*>(wf1), 1, ld1,
        //                                      reinterpret_cast<const rocblas_double*>(wf2), 1, ld2, num_bands,
        //                                      reinterpret_cast<rocblas_double*>(d_work.get()));
        // #elif defined(SIRIUS_CUDA)
        //         T beta{0};
        //         T alpha{1};
        //         // https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched
        //         cublasDgemmStridedBatched(stream, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, n, &alpha, wf1, n, ld1, wf2, n, ld2, &beta,
        //                                   d_work.get(), 1, 1, num_bands);
        // #endif
        //         acc::copyout(result_local.data(), d_work.get(), num_bands);
        for (int i = 0; i < num_bands; ++i) {
            result_local[i] = F{2} * result_local[i];
        }

        if (reduced == 1) {
            /* rank owns G=0 */
            std::vector<std::complex<double>> wf1_h_g0(num_bands);
            std::vector<std::complex<double>> wf2_h_g0(num_bands);
            // copy G=0 to host
            acc::copyout(wf1_h_g0.data(), num_bands, wf1, ld1, 1, num_bands);
            acc::copyout(wf2_h_g0.data(), num_bands, wf2, ld2, 1, num_bands);
            for (int i = 0; i < num_bands; ++i) {
                result_local[i] -= wf1_h_g0[i] * wf2_h_g0[i];
            }
        } else if (reduced == 2) {
            /* rank doesn't own G=0  */
            // nothing todo
        }
    }

    // add to `result` (because of spin-components)
    for (int i = 0; i < num_bands; ++i) {
        if constexpr (is_real_v<F>) {
            result[i] += std::real(result_local[i]);
        } else {
            result[i] += result_local[i];
        }
    }
}

} // namespace sirius
#endif // defined(SIRIUS_ROCM) || defined(SIRIUS_CUDA)

#endif /* __WF_INNER_LOCAL_HPP__ */

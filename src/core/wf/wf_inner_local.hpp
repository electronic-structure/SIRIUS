#ifndef __WF_INNER_LOCAL_HPP__
#define __WF_INNER_LOCAL_HPP__

#if defined(SIRIUS_ROCM) || defined(SIRIUS_CUDA)

#include <complex>
#include <type_traits>
#include <vector>
#include "core/acc/acc.hpp"
#include "core/acc/acc_blas.hpp"
#include "core/typedefs.hpp"

namespace sirius {

template <typename T, typename F>
void
inner_product_local_gpu(std::complex<T> const* wf1, int ld1, std::complex<T> const* wf2, int ld2, int n, int num_bands,
                        int reduced, std::vector<F>& result)
{
    using C = std::complex<T>;

    std::vector<C> result_local(num_bands, 0);

    if constexpr (std::is_same_v<T, double>) {
        acc::blas::zdotc_strided_batched(n, reinterpret_cast<acc_complex_double_t const*>(wf1), ld1,
                                         reinterpret_cast<acc_complex_double_t const*>(wf2), ld2, num_bands,
                                         reinterpret_cast<acc_complex_double_t*>(result_local.data()), 0 /* stream id */);
    } else {
        acc::blas::cdotc_strided_batched(n, reinterpret_cast<acc_complex_float_t const*>(wf1), ld1,
                                         reinterpret_cast<acc_complex_float_t const*>(wf2), ld2, num_bands,
                                         reinterpret_cast<acc_complex_float_t*>(result_local.data()), 0 /* stream id */);
    }

    if constexpr (is_real_v<F>) {
        /* Γ-point case: only half of the G-vectors are stored */
        for (int i = 0; i < num_bands; ++i) {
            result_local[i] *= T{2};
        }

        if (reduced == 1) {
            /* rank owns G=0, which was counted twice */
            std::vector<C> wf1_h_g0(num_bands);
            std::vector<C> wf2_h_g0(num_bands);
            // copy G=0 to host
            acc::copyout(wf1_h_g0.data(), 1, wf1, ld1, 1, num_bands);
            acc::copyout(wf2_h_g0.data(), 1, wf2, ld2, 1, num_bands);
            for (int i = 0; i < num_bands; ++i) {
                result_local[i] -= std::conj(wf1_h_g0[i]) * wf2_h_g0[i];
            }
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

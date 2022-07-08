// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file generate_fv_states.hpp
 *
 *  \brief Contains implementation of sirius::K_point::generate_fv_states method.
 */

#include "k_point.hpp"

namespace sirius {

template <typename T>
void K_point<T>::generate_fv_states()
{
    PROFILE("sirius::K_point::generate_fv_states");

    if (!ctx_.full_potential()) {
        return;
    }

    sddk::mdarray<std::complex<T>, 2> alm(num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size(), sddk::memory_t::host);
    sddk::mdarray<std::complex<T>, 2> tmp(unit_cell_.max_mt_aw_basis_size(), ctx_.num_fv_states());

    if (ctx_.processing_unit() == sddk::device_t::GPU) {
        fv_eigen_vectors_slab().pw_coeffs(0).allocate(sddk::memory_t::device);
        fv_eigen_vectors_slab().pw_coeffs(0).copy_to(sddk::memory_t::device, 0, ctx_.num_fv_states());
        alm.allocate(sddk::memory_t::device);
        tmp.allocate(sddk::memory_t::device);
    }

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto location = fv_eigen_vectors_slab().spl_num_atoms().location(ia);
        /* number of alm coefficients for atom */
        int mt_aw_size = unit_cell_.atom(ia).mt_aw_basis_size();
        int mt_lo_size = unit_cell_.atom(ia).mt_lo_basis_size();
        /* generate matching coefficients for all G-vectors */
        alm_coeffs_loc_->generate<false>(unit_cell_.atom(ia), alm);

        std::complex<T>* tmp_ptr_gpu{nullptr};

        auto la = sddk::linalg_t::none;
        auto mt = sddk::memory_t::none;
        switch (ctx_.processing_unit()) {
            case sddk::device_t::CPU: {
                la = sddk::linalg_t::blas;
                mt = sddk::memory_t::host;
                break;
            }
            case sddk::device_t::GPU: {
                alm.copy_to(sddk::memory_t::device, 0, mt_aw_size * num_gkvec_loc());
                la = sddk::linalg_t::gpublas;
                mt = sddk::memory_t::device;
                tmp_ptr_gpu = tmp.at(sddk::memory_t::device);
                break;
            }
        }

        sddk::mdarray<std::complex<T>, 2> tmp1(tmp.at(sddk::memory_t::host), tmp_ptr_gpu, mt_aw_size, ctx_.num_fv_states());

        /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for a single atom */
        sddk::linalg(la).gemm('T', 'N', mt_aw_size, ctx_.num_fv_states(), num_gkvec_loc(),
            &sddk::linalg_const<std::complex<T>>::one(), alm.at(mt), alm.ld(),
            fv_eigen_vectors_slab().pw_coeffs(0).prime().at(mt),
            fv_eigen_vectors_slab().pw_coeffs(0).prime().ld(),
            &sddk::linalg_const<std::complex<T>>::zero(), tmp1.at(mt), tmp1.ld());

        switch (ctx_.processing_unit()) {
            case sddk::device_t::CPU: {
                break;
            }
            case sddk::device_t::GPU: {
                tmp1.copy_to(sddk::memory_t::host);
                break;
            }
        }

        comm_.reduce(tmp1.at(sddk::memory_t::host), static_cast<int>(tmp1.size()), location.rank);
// TODO: remove __PRINT_OBJECT_CHECKSUM
#ifdef __PRINT_OBJECT_CHECKSUM
        auto z1 = tmp1.checksum();
        DUMP("checksum(tmp1): %18.10f %18.10f", std::real(z1), std::imag(z1));
#endif

        if (location.rank == comm_.rank()) {
            int offset1 = fv_states().offset_mt_coeffs(location.local_index);
            int offset2 = fv_eigen_vectors_slab().offset_mt_coeffs(location.local_index);
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                /* aw block */
                std::memcpy(fv_states().mt_coeffs(0).prime().at(sddk::memory_t::host, offset1, i),
                            tmp1.at(sddk::memory_t::host, 0, i), mt_aw_size * sizeof(std::complex<T>));
                /* lo block */
                if (mt_lo_size) {
                    std::memcpy(fv_states().mt_coeffs(0).prime().at(sddk::memory_t::host, offset1 + mt_aw_size, i),
                                fv_eigen_vectors_slab().mt_coeffs(0).prime().at(sddk::memory_t::host, offset2, i),
                                mt_lo_size * sizeof(std::complex<T>));
                }
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        /* G+k block */
        std::memcpy(fv_states().pw_coeffs(0).prime().at(sddk::memory_t::host, 0, i),
                    fv_eigen_vectors_slab().pw_coeffs(0).prime().at(sddk::memory_t::host, 0, i),
                    num_gkvec_loc() * sizeof(std::complex<T>));
    }

    if (ctx_.processing_unit() == sddk::device_t::GPU) {
        fv_eigen_vectors_slab().pw_coeffs(0).deallocate(sddk::memory_t::device);
    }
}

template void K_point<double>::generate_fv_states();
#ifdef USE_FP32
template void K_point<float>::generate_fv_states();
#endif

} // namespace sirius

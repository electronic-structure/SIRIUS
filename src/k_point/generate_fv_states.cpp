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
#include "lapw/generate_alm_block.hpp"

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

    auto const& uc = ctx_.unit_cell();

    auto bs = ctx_.cyclic_block_size();
    sddk::dmatrix<std::complex<T>> alm_fv(uc.mt_aw_basis_size(), ctx_.num_fv_states(),
            ctx_.blacs_grid(), bs, bs);


    int atom_begin{0};
    int mt_aw_offset{0};

    /* loop over blocks of atoms */
    for (auto na : utils::split_in_blocks(uc.num_atoms(), 64)) {
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        //std::vector<int> offsets_aw(na);
        for (int i = 0; i < na; i++) {
            int ia     = atom_begin + i;
            auto& type = uc.atom(ia).type();
            num_mt_aw += type.mt_aw_basis_size();
        }

        /* generate complex conjugated Alm coefficients for a block of atoms */
        auto alm = generate_alm_block<false, T>(ctx_, atom_begin, na, this->alm_coeffs_loc());

        /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for the block of atoms */
        spla::pgemm_ssb(num_mt_aw, ctx_.num_fv_states(), this->gkvec().count(), SPLA_OP_TRANSPOSE, 1.0,
                alm.at(sddk::memory_t::host), alm.ld(),
                &fv_eigen_vectors_slab_new().pw_coeffs(sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(0)),
                fv_eigen_vectors_slab_new().ld(),
                0.0, alm_fv.at(sddk::memory_t::host), alm_fv.ld(), mt_aw_offset, 0, alm_fv.spla_distribution(),
                ctx_.spla_context());

        atom_begin += na;
        mt_aw_offset += num_mt_aw;
    }

    std::vector<int> num_mt_apw_coeffs(uc.num_atoms());
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        num_mt_apw_coeffs[ia] = uc.atom(ia).mt_aw_basis_size();
    }
    wf::Wave_functions_mt<T> alm_fv_slab(this->comm(), num_mt_apw_coeffs, wf::num_spins(1),
            wf::num_bands(ctx_.num_fv_states()), sddk::memory_t::host);

    auto& one = sddk::linalg_const<std::complex<T>>::one();
    auto& zero = sddk::linalg_const<std::complex<T>>::zero();

    auto layout_in = alm_fv.grid_layout(0, 0, uc.mt_aw_basis_size(), ctx_.num_fv_states());
    auto layout_out = alm_fv_slab.grid_layout_mt(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
    costa::transform(layout_in, layout_out, 'N', one, zero, this->comm().mpi_comm());

    check_wf_diff("fv_eigen_vectors_slab", fv_eigen_vectors_slab(), fv_eigen_vectors_slab_new());

    #pragma omp parallel for
    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        /* G+k block */
        auto in_ptr = &fv_eigen_vectors_slab_new().pw_coeffs(sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(i));
        auto out_ptr = &fv_states_new_->pw_coeffs(sddk::memory_t::host, 0, wf::spin_index(0), wf::band_index(i));
        std::copy(in_ptr, in_ptr + gkvec().count(), out_ptr);

        for (int ialoc = 0; ialoc < alm_fv_slab.spl_num_atoms().local_size(); ialoc++) {
            int ia = alm_fv_slab.spl_num_atoms()[ialoc];
            int num_mt_aw = uc.atom(ia).type().mt_aw_basis_size();
            for (int xi = 0; xi < num_mt_aw; xi++) {
                fv_states_new_->mt_coeffs(sddk::memory_t::host, xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i)) =
                    alm_fv_slab.mt_coeffs(sddk::memory_t::host, xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i));
            }
            for (int xi = 0; xi < uc.atom(ia).type().mt_lo_basis_size(); xi++) {
                fv_states_new_->mt_coeffs(sddk::memory_t::host, num_mt_aw + xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i)) =
                    fv_eigen_vectors_slab_new().mt_coeffs(sddk::memory_t::host, xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i));
            }
        }
    }

    check_wf_diff("fv_eigen_states",*fv_states_, *fv_states_new_);
}

template void K_point<double>::generate_fv_states();
#ifdef USE_FP32
template void K_point<float>::generate_fv_states();
#endif

} // namespace sirius

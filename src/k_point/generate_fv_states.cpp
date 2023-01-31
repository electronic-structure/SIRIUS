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

    auto const& uc = ctx_.unit_cell();

    auto pcs = env::print_checksum();

    auto bs = ctx_.cyclic_block_size();
    la::dmatrix<std::complex<T>> alm_fv(uc.mt_aw_basis_size(), ctx_.num_fv_states(),
            ctx_.blacs_grid(), bs, bs);

    int atom_begin{0};
    int mt_aw_offset{0};

    /* loop over blocks of atoms */
    for (auto na : utils::split_in_blocks(uc.num_atoms(), 64)) {
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        for (int i = 0; i < na; i++) {
            int ia     = atom_begin + i;
            auto& type = uc.atom(ia).type();
            num_mt_aw += type.mt_aw_basis_size();
        }

        /* generate complex conjugated Alm coefficients for a block of atoms */
        auto alm = generate_alm_block<false, T>(ctx_, atom_begin, na, this->alm_coeffs_loc());
        auto cs = alm.checksum();
        if (pcs) {
            utils::print_checksum("alm", cs, RTE_OUT(this->out(0)));
        }

        /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for the block of atoms */
        spla::pgemm_ssb(num_mt_aw, ctx_.num_fv_states(), this->gkvec().count(), SPLA_OP_TRANSPOSE, 1.0,
                alm.at(sddk::memory_t::host), alm.ld(),
                &fv_eigen_vectors_slab().pw_coeffs(0, wf::spin_index(0), wf::band_index(0)),
                fv_eigen_vectors_slab().ld(),
                0.0, alm_fv.at(sddk::memory_t::host), alm_fv.ld(), mt_aw_offset, 0, alm_fv.spla_distribution(),
                ctx_.spla_context());

        atom_begin += na;
        mt_aw_offset += num_mt_aw;
    }

    std::vector<int> num_mt_apw_coeffs(uc.num_atoms());
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        num_mt_apw_coeffs[ia] = uc.atom(ia).mt_aw_basis_size();
    }
    wf::Wave_functions_mt<T> alm_fv_slab(this->comm(), num_mt_apw_coeffs, wf::num_mag_dims(0),
            wf::num_bands(ctx_.num_fv_states()), sddk::memory_t::host);

    auto& one = la::constant<std::complex<T>>::one();
    auto& zero = la::constant<std::complex<T>>::zero();

    auto layout_in = alm_fv.grid_layout(0, 0, uc.mt_aw_basis_size(), ctx_.num_fv_states());
    auto layout_out = alm_fv_slab.grid_layout_mt(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
    costa::transform(layout_in, layout_out, 'N', one, zero, this->comm().native());

    #pragma omp parallel for
    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        /* G+k block */
        auto in_ptr = &fv_eigen_vectors_slab().pw_coeffs(0, wf::spin_index(0), wf::band_index(i));
        auto out_ptr = &fv_states_->pw_coeffs(0, wf::spin_index(0), wf::band_index(i));
        std::copy(in_ptr, in_ptr + gkvec().count(), out_ptr);

        for (int ialoc = 0; ialoc < alm_fv_slab.spl_num_atoms().local_size(); ialoc++) {
            int ia = alm_fv_slab.spl_num_atoms()[ialoc];
            int num_mt_aw = uc.atom(ia).type().mt_aw_basis_size();
            /* aw part of the muffin-tin coefficients */
            for (int xi = 0; xi < num_mt_aw; xi++) {
                fv_states_->mt_coeffs(xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i)) =
                    alm_fv_slab.mt_coeffs(xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i));
            }
            /* lo part of muffin-tin coefficients */
            for (int xi = 0; xi < uc.atom(ia).type().mt_lo_basis_size(); xi++) {
                fv_states_->mt_coeffs(num_mt_aw + xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i)) =
                    fv_eigen_vectors_slab().mt_coeffs(xi, wf::atom_index(ialoc), wf::spin_index(0), wf::band_index(i));
            }
        }
    }
    if (pcs) {
        auto z1 = fv_states_->checksum_pw(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        auto z2 = fv_states_->checksum_mt(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        utils::print_checksum("fv_states_pw", z1, RTE_OUT(this->out(0)));
        utils::print_checksum("fv_states_mt", z2, RTE_OUT(this->out(0)));

    }
}

template void K_point<double>::generate_fv_states();
#ifdef USE_FP32
template void K_point<float>::generate_fv_states();
#endif

} // namespace sirius

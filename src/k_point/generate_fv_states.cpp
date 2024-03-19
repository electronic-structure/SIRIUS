/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_fv_states.hpp
 *
 *  \brief Contains implementation of sirius::K_point::generate_fv_states method.
 */

#include "k_point.hpp"
#include "lapw/generate_alm_block.hpp"

namespace sirius {

template <typename T>
void
K_point<T>::generate_fv_states()
{
    PROFILE("sirius::K_point::generate_fv_states");

    if (!ctx_.full_potential()) {
        return;
    }

    auto const& uc = ctx_.unit_cell();

    auto pcs = env::print_checksum();

    auto bs = ctx_.cyclic_block_size();
    la::dmatrix<std::complex<T>> alm_fv(uc.mt_aw_basis_size(), ctx_.num_fv_states(), ctx_.blacs_grid(), bs, bs);

    int atom_begin{0};
    int mt_aw_offset{0};

    /* loop over blocks of atoms */
    for (auto na : split_in_blocks(uc.num_atoms(), 64)) {
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        for (int i = 0; i < na; i++) {
            int ia     = atom_begin + i;
            auto& type = uc.atom(ia).type();
            num_mt_aw += type.mt_aw_basis_size();
        }

        /* generate complex conjugated Alm coefficients for a block of atoms */
        auto alm = generate_alm_block<false, T>(ctx_, atom_begin, na, this->alm_coeffs_loc());
        auto cs  = alm.checksum();
        if (pcs) {
            print_checksum("alm", cs, RTE_OUT(this->out(0)));
        }

        /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for the block of atoms */
        spla::pgemm_ssb(num_mt_aw, ctx_.num_fv_states(), this->gkvec().count(), SPLA_OP_TRANSPOSE, 1.0,
                        alm.at(memory_t::host), alm.ld(),
                        &fv_eigen_vectors_slab().pw_coeffs(0, wf::spin_index(0), wf::band_index(0)),
                        fv_eigen_vectors_slab().ld(), 0.0, alm_fv.at(memory_t::host), alm_fv.ld(), mt_aw_offset, 0,
                        alm_fv.spla_distribution(), ctx_.spla_context());

        atom_begin += na;
        mt_aw_offset += num_mt_aw;
    }

    std::vector<int> num_mt_apw_coeffs(uc.num_atoms());
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        num_mt_apw_coeffs[ia] = uc.atom(ia).mt_aw_basis_size();
    }
    wf::Wave_functions_mt<T> alm_fv_slab(this->comm(), num_mt_apw_coeffs, wf::num_mag_dims(0),
                                         wf::num_bands(ctx_.num_fv_states()), memory_t::host);

    auto& one  = la::constant<std::complex<T>>::one();
    auto& zero = la::constant<std::complex<T>>::zero();

    auto layout_in  = alm_fv.grid_layout(0, 0, uc.mt_aw_basis_size(), ctx_.num_fv_states());
    auto layout_out = alm_fv_slab.grid_layout_mt(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
    costa::transform(layout_in, layout_out, 'N', one, zero, this->comm().native());

    #pragma omp parallel for
    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        /* G+k block */
        auto in_ptr  = &fv_eigen_vectors_slab().pw_coeffs(0, wf::spin_index(0), wf::band_index(i));
        auto out_ptr = &fv_states_->pw_coeffs(0, wf::spin_index(0), wf::band_index(i));
        std::copy(in_ptr, in_ptr + gkvec().count(), out_ptr);

        for (auto it : alm_fv_slab.spl_num_atoms()) {
            int num_mt_aw = uc.atom(it.i).type().mt_aw_basis_size();
            /* aw part of the muffin-tin coefficients */
            for (int xi = 0; xi < num_mt_aw; xi++) {
                fv_states_->mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i)) =
                        alm_fv_slab.mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i));
            }
            /* lo part of muffin-tin coefficients */
            for (int xi = 0; xi < uc.atom(it.i).type().mt_lo_basis_size(); xi++) {
                fv_states_->mt_coeffs(num_mt_aw + xi, it.li, wf::spin_index(0), wf::band_index(i)) =
                        fv_eigen_vectors_slab().mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i));
            }
        }
    }
    if (pcs) {
        auto z1 = fv_states_->checksum_pw(memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        auto z2 = fv_states_->checksum_mt(memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        print_checksum("fv_states_pw", z1, RTE_OUT(this->out(0)));
        print_checksum("fv_states_mt", z2, RTE_OUT(this->out(0)));
    }
}

template void
K_point<double>::generate_fv_states();
#ifdef SIRIUS_USE_FP32
template void
K_point<float>::generate_fv_states();
#endif

} // namespace sirius

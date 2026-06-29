/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_lapw_wave_functions.cpp
 *
 *  \brief Contains implementation of sirius::K_point::generate_lapw_wave_functions method.
 */

#include "k_point.hpp"
#include "lapw/generate_alm_block.hpp"

namespace sirius {

template <typename T>
void
K_point<T>::generate_lapw_wave_functions(wf::Wave_functions<T> const& evec__, wf::Wave_functions<T>& wf__, int ispn__)
{
    PROFILE("sirius::K_point::generate_lapw_wave_functions");

    if (!ctx_.full_potential()) {
        return;
    }

    auto const& uc = ctx_.unit_cell();

    auto pcs = env::print_checksum();

    auto bs = ctx_.cyclic_block_size();
    /* store the result of Alm(G) * C_i(G) product */
    la::dmatrix<std::complex<T>> alm_fv(uc.mt_aw_basis_size(), ctx_.num_fv_states(), ctx_.blacs_grid(), bs, bs);

    int atom_begin{0};
    int mt_aw_offset{0};

    auto evec_mg = evec__.memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);

    /* loop over blocks of atoms */
    for (auto na : split_in_blocks(uc.num_atoms(), ctx_.cfg().control().max_atom_chunk_size())) {
        /* actual number of AW radial functions in a block of atoms */
        int num_mt_aw{0};
        for (int i = 0; i < na; i++) {
            int ia     = atom_begin + i;
            auto& type = uc.atom(ia).type();
            num_mt_aw += type.mt_aw_basis_size();
        }

        /* generate complex conjugated Alm coefficients for a block of atoms */
        auto alm = generate_alm_block<true, T>(ctx_, atom_begin, na, this->alm_coeffs_loc());
        if (pcs) {
            auto cs = alm.checksum();
            print_checksum("alm", cs, RTE_OUT(this->out(0)));
        }

        /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for the block of atoms */
        spla::pgemm_ssb(num_mt_aw, ctx_.num_fv_states(), this->gkvec().count(), SPLA_OP_CONJ_TRANSPOSE, 1.0,
                        alm.at(ctx_.processing_unit_memory_t()), alm.ld(),
                        evec__.pw_coeffs(wf::spin_index(0)).at(ctx_.processing_unit_memory_t()), evec__.ld(), 0.0,
                        alm_fv.at(memory_t::host), alm_fv.ld(), mt_aw_offset, 0, alm_fv.spla_distribution(),
                        ctx_.spla_context());

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
        auto in_ptr  = &evec__.pw_coeffs(0, wf::spin_index(0), wf::band_index(i));
        auto out_ptr = &wf__.pw_coeffs(0, wf::spin_index(ispn__), wf::band_index(i));
        std::copy(in_ptr, in_ptr + gkvec().count(), out_ptr);

        for (auto it : alm_fv_slab.spl_num_atoms()) {
            int num_mt_aw = uc.atom(it.i).type().mt_aw_basis_size();
            /* aw part of the muffin-tin coefficients */
            for (int xi = 0; xi < num_mt_aw; xi++) {
                wf__.mt_coeffs(xi, it.li, wf::spin_index(ispn__), wf::band_index(i)) =
                        alm_fv_slab.mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i));
            }
            /* lo part of muffin-tin coefficients: copy from evec */
            for (int xi = 0; xi < uc.atom(it.i).type().mt_lo_basis_size(); xi++) {
                wf__.mt_coeffs(num_mt_aw + xi, it.li, wf::spin_index(ispn__), wf::band_index(i)) =
                        evec__.mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(i));
            }
        }
    }
    if (pcs) {
        auto z1 = wf__.checksum_pw(memory_t::host, wf::spin_index(ispn__), wf::band_range(0, ctx_.num_fv_states()));
        auto z2 = wf__.checksum_mt(memory_t::host, wf::spin_index(ispn__), wf::band_range(0, ctx_.num_fv_states()));
        print_checksum("wf_pw", z1, RTE_OUT(this->out(0)));
        print_checksum("wf_mt", z2, RTE_OUT(this->out(0)));
    }
}

template void
K_point<double>::generate_lapw_wave_functions(wf::Wave_functions<double> const& evec__,
                                              wf::Wave_functions<double>& wf__, int ispn__);
#if defined(SIRIUS_USE_FP32)
template void
K_point<float>::generate_lapw_wave_functions(wf::Wave_functions<float> const& evec__, wf::Wave_functions<float>& wf__,
                                             int ispn__);
#endif

} // namespace sirius

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "non_local_operator.hpp"
#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace sirius {

template <class T>
U_operator<T>::U_operator(Simulation_context const& ctx__, Hubbard_matrix const& um1__, std::array<double, 3> vk__)
    : ctx_(ctx__)
    , vk_(vk__)
{
    if (!ctx_.hubbard_correction()) {
        return;
    }
    /* a pair of "total number, offests" for the Hubbard orbitals idexing */
    auto r                 = ctx_.unit_cell().num_hubbard_wf();
    this->nhwf_            = r.first;
    this->offset_          = um1__.offset();
    this->atomic_orbitals_ = um1__.atomic_orbitals();
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        um_[j] = la::dmatrix<std::complex<T>>(r.first, r.first);
        um_[j].zero();
    }

    /* copy local blocks */
    for (int at_lvl = 0; at_lvl < static_cast<int>(um1__.atomic_orbitals().size()); at_lvl++) {
        const int ia    = um1__.atomic_orbitals(at_lvl).first;
        auto& atom_type = ctx_.unit_cell().atom(ia).type();
        int lo_ind      = um1__.atomic_orbitals(at_lvl).second;
        if (atom_type.lo_descriptor_hub(lo_ind).use_for_calculation()) {
            int lmmax_at = 2 * atom_type.lo_descriptor_hub(lo_ind).l() + 1;
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                for (int m2 = 0; m2 < lmmax_at; m2++) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        um_[j](um1__.offset(at_lvl) + m1, um1__.offset(at_lvl) + m2) = um1__.local(at_lvl)(m1, m2, j);
                    }
                }
            }
        }
    }

    for (int i = 0; i < ctx_.cfg().hubbard().nonlocal().size(); i++) {
        auto nl = ctx_.cfg().hubbard().nonlocal(i);
        int ia  = nl.atom_pair()[0];
        int ja  = nl.atom_pair()[1];
        int il  = nl.l()[0];
        int jl  = nl.l()[1];
        auto Tr = nl.T();

        /* we need to find the index of the radial function corresponding to the atomic level of each atom.  */
        int at1_lvl = um1__.find_orbital_index(ia, nl.n()[0], il);
        int at2_lvl = um1__.find_orbital_index(ja, nl.n()[1], jl);

        auto z1 = std::exp(std::complex<double>(0, twopi * dot(vk_, r3::vector<int>(Tr))));
        for (int is = 0; is < ctx_.num_spins(); is++) {
            for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
                    um_[is](um1__.offset(at1_lvl) + m1, um1__.offset(at2_lvl) + m2) +=
                            z1 * um1__.nonlocal(i)(m1, m2, is);
                }
            }
        }
    }
    for (int is = 0; is < ctx_.num_spins(); is++) {
        auto diff = check_hermitian(um_[is], r.first);
        if (diff > 1e-10) {
            RTE_THROW("um is not Hermitian");
        }
        if (env::print_checksum()) {
            print_checksum("um" + std::to_string(is), um_[is].checksum(r.first, r.first), RTE_OUT(ctx_.out()));
        }
        if (ctx_.processing_unit() == device_t::GPU) {
            um_[is].allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
        }
    }
}

template class U_operator<double>;
#if defined(SIRIUS_USE_FP32)
template class U_operator<float>;
#endif

/** Apply Hubbard U correction
 * \tparam T  Precision type of wave-functions (flat or double).
 * \param [in]  hub_wf   Hubbard atomic wave-functions.
 * \param [in]  phi      Set of wave-functions to which Hubbard correction is applied.
 * \param [out] hphi     Output wave-functions to which the result is added.
 */
template <typename T>
void
apply_U_operator(Simulation_context& ctx__, wf::spin_range spins__, wf::band_range br__,
                 wf::Wave_functions<T> const& hub_wf__, wf::Wave_functions<T> const& phi__, U_operator<T> const& um__,
                 wf::Wave_functions<T>& hphi__)
{
    if (!ctx__.hubbard_correction()) {
        return;
    }

    la::dmatrix<std::complex<T>> dm(hub_wf__.num_wf().get(), br__.size());

    auto mt = ctx__.processing_unit_memory_t();
    auto la = la::lib_t::blas;
    if (is_device_memory(mt)) {
        la = la::lib_t::gpublas;
        dm.allocate(mt);
    }

    /* First calculate the local part of the projections
       dm(i, n) = <phi_i| S |psi_{nk}> */
    wf::inner(ctx__.spla_context(), mt, spins__, hub_wf__, wf::band_range(0, hub_wf__.num_wf().get()), phi__, br__, dm,
              0, 0);

    la::dmatrix<std::complex<T>> Up(hub_wf__.num_wf().get(), br__.size());
    if (is_device_memory(mt)) {
        Up.allocate(mt);
    }

    if (ctx__.num_mag_dims() == 3) {
        Up.zero();
        #pragma omp parallel for schedule(static)
        for (int at_lvl = 0; at_lvl < (int)um__.atomic_orbitals().size(); at_lvl++) {
            const int ia     = um__.atomic_orbitals(at_lvl).first;
            auto const& atom = ctx__.unit_cell().atom(ia);
            if (atom.type().lo_descriptor_hub(um__.atomic_orbitals(at_lvl).second).use_for_calculation()) {
                const int lmax_at = 2 * atom.type().lo_descriptor_hub(um__.atomic_orbitals(at_lvl).second).l() + 1;
                // we apply the hubbard correction. For now I have no papers
                // giving me the formula for the SO case so I rely on QE for it
                // but I do not like it at all
                for (int s1 = 0; s1 < ctx__.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx__.num_spins(); s2++) {
                        // TODO: replace this with matrix matrix multiplication
                        for (int nbd = 0; nbd < br__.size(); nbd++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);
                                    Up(um__.nhwf() * s1 + um__.offset(at_lvl) + m1, nbd) +=
                                            um__(um__.offset(at_lvl) + m2, um__.offset(at_lvl) + m1, ind) *
                                            dm(um__.nhwf() * s2 + um__.offset(at_lvl) + m2, nbd);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        la::wrap(la).gemm('N', 'N', um__.nhwf(), br__.size(), um__.nhwf(), &la::constant<std::complex<T>>::one(),
                          um__.at(mt, 0, 0, spins__.begin().get()), um__.nhwf(), dm.at(mt, 0, 0), dm.ld(),
                          &la::constant<std::complex<T>>::zero(), Up.at(mt, 0, 0), Up.ld());
        if (is_device_memory(mt)) {
            Up.copy_to(memory_t::host);
        }
    }
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto sp  = hub_wf__.actual_spin_index(s);
        auto sp1 = hphi__.actual_spin_index(s);
        wf::transform(ctx__.spla_context(), mt, Up, 0, 0, 1.0, hub_wf__, sp, wf::band_range(0, hub_wf__.num_wf().get()),
                      1.0, hphi__, sp1, br__);
    }
}

template void
apply_U_operator<double>(Simulation_context&, wf::spin_range, wf::band_range, const wf::Wave_functions<double>&,
                         const wf::Wave_functions<double>&, U_operator<double> const&, wf::Wave_functions<double>&);

#ifdef SIRIUS_USE_FP32
template void
apply_U_operator<float>(Simulation_context&, wf::spin_range, wf::band_range, const wf::Wave_functions<float>&,
                        const wf::Wave_functions<float>&, U_operator<float> const&, wf::Wave_functions<float>&);
#endif

/// Apply strain derivative of S-operator to all scalar functions.
void
apply_S_operator_strain_deriv(memory_t mem__, int comp__, Beta_projector_generator<double>& bp__,
                              beta_projectors_coeffs_t<double>& bp_coeffs__,
                              Beta_projector_generator<double>& bp_strain_deriv__,
                              beta_projectors_coeffs_t<double>& bp_strain_deriv_coeffs__,
                              wf::Wave_functions<double>& phi__, Q_operator<double>& q_op__,
                              wf::Wave_functions<double>& ds_phi__)
{
    if (is_device_memory(mem__)) {
        RTE_ASSERT((bp__.pu() == device_t::GPU));
    }
    // NOTE: Beta_projectors_generator knows the target memory!
    using complex_t = std::complex<double>;

    RTE_ASSERT(ds_phi__.num_wf() == phi__.num_wf());
    for (int ichunk = 0; ichunk < bp__.num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */
        bp__.generate(bp_coeffs__, ichunk);
        /* generate derived beta-projectors for a block of atoms */
        bp_strain_deriv__.generate(bp_strain_deriv_coeffs__, ichunk, comp__);

        auto host_mem         = bp__.ctx().host_memory_t();
        auto& spla_ctx        = bp__.ctx().spla_context();
        auto band_range_phi   = wf::band_range(0, phi__.num_wf().get());
        bool result_on_device = bp__.ctx().processing_unit() == device_t::GPU;
        auto dbeta_phi        = inner_prod_beta<complex_t>(spla_ctx, mem__, host_mem, result_on_device,
                                                    bp_strain_deriv_coeffs__, phi__, wf::spin_index(0), band_range_phi);
        auto beta_phi = inner_prod_beta<complex_t>(spla_ctx, mem__, host_mem, result_on_device, bp_coeffs__, phi__,
                                                   wf::spin_index(0), band_range_phi);

        auto band_range = wf::band_range(0, ds_phi__.num_wf().get());
        q_op__.apply(mem__, ichunk, 0, ds_phi__, band_range, bp_coeffs__, dbeta_phi);
        q_op__.apply(mem__, ichunk, 0, ds_phi__, band_range, bp_strain_deriv_coeffs__, beta_phi);
    }
}

} // namespace sirius

// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file diag_full_potential.cpp
 *
 *  \brief Diagonalization of full-potential Hamiltonian.
 */

#include "band.hpp"
#include "residuals.hpp"
#include "wf_inner.hpp"
#include "wf_ortho.hpp"
#include "wf_trans.hpp"
#include "context/simulation_context.hpp"
#include "k_point/k_point.hpp"
#include "utils/profiler.hpp"
#include "davidson.hpp"

namespace sirius {

void
Band::diag_full_potential_first_variation_exact(Hamiltonian_k<double>& Hk__) const
{
    PROFILE("sirius::Band::diag_fv_exact");

    auto& kp = Hk__.kp();

    auto& solver = ctx_.gen_evp_solver();

    /* total eigen-value problem size */
    int ngklo = kp.gklo_basis_size();

    /* block size of scalapack 2d block-cyclic distribution */
    int bs = ctx_.cyclic_block_size();

    sddk::dmatrix<double_complex> h(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, ctx_.mem_pool(solver.host_memory_t()));
    sddk::dmatrix<double_complex> o(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, ctx_.mem_pool(solver.host_memory_t()));

    /* setup Hamiltonian and overlap */
    Hk__.set_fv_h_o(h, o);

    if (ctx_.gen_evp_solver().type() == ev_solver_t::cusolver) {
        auto& mpd = ctx_.mem_pool(sddk::memory_t::device);
        h.allocate(mpd);
        o.allocate(mpd);
        kp.fv_eigen_vectors().allocate(mpd);
    }

    if (ctx_.cfg().control().verification() >= 1) {
        double max_diff = check_hermitian(h, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "H matrix is not hermitian" << std::endl
              << "max error: " << max_diff;
            TERMINATE(s);
        }
        max_diff = check_hermitian(o, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "O matrix is not hermitian" << std::endl
              << "max error: " << max_diff;
            TERMINATE(s);
        }
    }

    if (ctx_.print_checksum()) {
        auto z1 = h.checksum(ngklo, ngklo);
        auto z2 = o.checksum(ngklo, ngklo);
        if (kp.comm().rank() == 0) {
            utils::print_checksum("h_lapw", z1);
            utils::print_checksum("o_lapw", z2);
        }
    }

    assert(kp.gklo_basis_size() > ctx_.num_fv_states());

    std::vector<double> eval(ctx_.num_fv_states());

    ctx_.print_memory_usage(__FILE__, __LINE__);
    if (solver.solve(kp.gklo_basis_size(), ctx_.num_fv_states(), h, o, eval.data(), kp.fv_eigen_vectors())) {
        RTE_THROW("error in generalized eigen-value problem");
    }
    ctx_.print_memory_usage(__FILE__, __LINE__);

    if (ctx_.gen_evp_solver().type() == ev_solver_t::cusolver) {
        h.deallocate(sddk::memory_t::device);
        o.deallocate(sddk::memory_t::device);
        kp.fv_eigen_vectors().deallocate(sddk::memory_t::device);
    }
    kp.set_fv_eigen_values(&eval[0]);

    for (int i = 0; i < ctx_.num_fv_states(); i++) {
        kp.message(4, __function_name__, "eval[%i]=%20.16f\n", i, eval[i]);
    }

    if (ctx_.print_checksum()) {
        auto z1 = kp.fv_eigen_vectors().checksum(kp.gklo_basis_size(), ctx_.num_fv_states());
        if (kp.comm().rank() == 0) {
            utils::print_checksum("fv_eigen_vectors", z1);
        }
    }

    /* remap to slab */
    {
        auto layout_in = kp.fv_eigen_vectors().grid_layout(0, 0, kp.gkvec().num_gvec(), ctx_.num_fv_states());
        auto layout_out = kp.fv_eigen_vectors_slab_new().grid_layout_pw(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        costa::transform(layout_in, layout_out, 'N', sddk::linalg_const<std::complex<double>>::one(),
            sddk::linalg_const<std::complex<double>>::zero(), kp.comm().mpi_comm());
    }
    {
        auto layout_in = kp.fv_eigen_vectors().grid_layout(kp.gkvec().num_gvec(), 0,
                ctx_.unit_cell().mt_lo_basis_size(), ctx_.num_fv_states());
        auto layout_out = kp.fv_eigen_vectors_slab_new().grid_layout_mt(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        costa::transform(layout_in, layout_out, 'N', sddk::linalg_const<std::complex<double>>::one(),
            sddk::linalg_const<std::complex<double>>::zero(), kp.comm().mpi_comm());
    }

    /* renormalize wave-functions */
    if (ctx_.valence_relativity() == relativity_t::iora) {

        std::vector<int> num_mt_coeffs(unit_cell_.num_atoms());
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            num_mt_coeffs[ia] = unit_cell_.atom(ia).mt_lo_basis_size();
        }
        wf::Wave_functions<double> ofv_new(kp.gkvec_sptr(), num_mt_coeffs, wf::num_mag_dims(0),
                wf::num_bands(ctx_.num_fv_states()), sddk::memory_t::host);

        Hk__.apply_fv_h_o(false, false, wf::band_range(0, ctx_.num_fv_states()), kp.fv_eigen_vectors_slab_new(),
                nullptr, &ofv_new);

        //if (ctx_.processing_unit() == sddk::device_t::GPU) {
        //    kp.fv_eigen_vectors_slab().deallocate(sddk::spin_range(0), sddk::memory_t::device);
        //}

        //if (true) {
        //    Wave_functions phi(kp.gkvec_partition(), unit_cell_.num_atoms(),
        //                       [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
        //                       ctx_.preferred_memory_t(), 1);
        //    Wave_functions ofv(kp.gkvec_partition(), unit_cell_.num_atoms(),
        //                       [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
        //                       ctx_.preferred_memory_t(), 1);
        //    phi.allocate(spin_range(0), memory_t::device);
        //    ofv.allocate(spin_range(0), memory_t::device);

        //    for (int i = 0; i < kp.num_gkvec(); i++) {
        //        phi.zero(device_t::CPU, 0, 0, ctx_.num_fv_states());
        //        for (int j = 0; j < ctx_.num_fv_states(); j++) {
        //            phi.pw_coeffs(0).prime(i, j) = 1.0;
        //        }
        //        phi.copy_to(spin_range(0), memory_t::device, 0, ctx_.num_fv_states());
        //        Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), phi, nullptr, &ofv);
        //    }

        //    for (int i = 0; i < unit_cell_.mt_lo_basis_size(); i++) {
        //        phi.zero(device_t::CPU, 0, 0, ctx_.num_fv_states());
        //        for (int j = 0; j < ctx_.num_fv_states(); j++) {
        //            phi.mt_coeffs(0).prime(i, j) = 1.0;
        //        }
        //        phi.copy_to(spin_range(0), memory_t::device, 0, ctx_.num_fv_states());
        //        Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), phi, nullptr, &ofv);
        //    }
        //}

        auto norm1 = wf::inner_diag<double, std::complex<double>>(sddk::memory_t::host, kp.fv_eigen_vectors_slab_new(), ofv_new, wf::spin_range(0),
                wf::num_bands(ctx_.num_fv_states()));

        std::vector<double> norm;
        for (auto e : norm1) {
            norm.push_back(1 / std::sqrt(std::real(e)));
        }

        wf::axpby<double, double>(sddk::memory_t::host, wf::spin_range(0), wf::band_range(0, ctx_.num_fv_states()),
                nullptr, nullptr, norm.data(), &kp.fv_eigen_vectors_slab_new());
    }

    //if (ctx_.cfg().control().verification() >= 2) {
    //    kp.message(1, __function_name__, "%s", "checking application of H and O\n");
    //    /* check application of H and O */
    //    sddk::Wave_functions<double> hphi(kp.gkvec_partition(), unit_cell_.num_atoms(),
    //                        [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
    //                        ctx_.preferred_memory_t());
    //    sddk::Wave_functions<double> ophi(kp.gkvec_partition(), unit_cell_.num_atoms(),
    //                        [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
    //                        ctx_.preferred_memory_t());

    //    if (ctx_.processing_unit() == sddk::device_t::GPU) {
    //        kp.fv_eigen_vectors_slab().allocate(sddk::spin_range(0), sddk::memory_t::device);
    //        kp.fv_eigen_vectors_slab().copy_to(sddk::spin_range(0), sddk::memory_t::device, 0, ctx_.num_fv_states());
    //        hphi.allocate(sddk::spin_range(0), sddk::memory_t::device);
    //        ophi.allocate(sddk::spin_range(0), sddk::memory_t::device);
    //    }

    //    Hk__.apply_fv_h_o(false, false, 0, ctx_.num_fv_states(), kp.fv_eigen_vectors_slab(), &hphi, &ophi);

    //    sddk::dmatrix<double_complex> hmlt(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
    //                                 ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
    //    sddk::dmatrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
    //                                 ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

    //    inner(ctx_.spla_context(), sddk::spin_range(0), kp.fv_eigen_vectors_slab(), 0, ctx_.num_fv_states(),
    //          hphi, 0, ctx_.num_fv_states(), hmlt, 0, 0);
    //    inner(ctx_.spla_context(), sddk::spin_range(0), kp.fv_eigen_vectors_slab(), 0, ctx_.num_fv_states(),
    //          ophi, 0, ctx_.num_fv_states(), ovlp, 0, 0);

    //    double max_diff{0};
    //    for (int i = 0; i < hmlt.num_cols_local(); i++) {
    //        int icol = hmlt.icol(i);
    //        for (int j = 0; j < hmlt.num_rows_local(); j++) {
    //            int jrow = hmlt.irow(j);
    //            if (icol == jrow) {
    //                max_diff = std::max(max_diff, std::abs(hmlt(j, i) - eval[icol]));
    //            } else {
    //                max_diff = std::max(max_diff, std::abs(hmlt(j, i)));
    //            }
    //        }
    //    }
    //    if (max_diff > 1e-9) {
    //        std::stringstream s;
    //        s << "application of Hamiltonian failed, maximum error: " << max_diff;
    //        WARNING(s);
    //    }

    //    max_diff = 0;
    //    for (int i = 0; i < ovlp.num_cols_local(); i++) {
    //        int icol = ovlp.icol(i);
    //        for (int j = 0; j < ovlp.num_rows_local(); j++) {
    //            int jrow = ovlp.irow(j);
    //            if (icol == jrow) {
    //                max_diff = std::max(max_diff, std::abs(ovlp(j, i) - 1.0));
    //            } else {
    //                max_diff = std::max(max_diff, std::abs(ovlp(j, i)));
    //            }
    //        }
    //    }
    //    if (max_diff > 1e-9) {
    //        std::stringstream s;
    //        s << "application of overlap failed, maximum error: " << max_diff;
    //        WARNING(s);
    //    }
    //}
}

void Band::get_singular_components(Hamiltonian_k<double>& Hk__, double itsol_tol__) const
{
    PROFILE("sirius::Band::get_singular_components");

    auto& kp = Hk__.kp();

    int ncomp = kp.singular_components_new().num_wf().get();

    ctx_.message(3, __function_name__, "number of singular components: %i\n", ncomp);

    auto& itso = ctx_.cfg().iterative_solver();

    std::stringstream s;
    std::ostream* out = (kp.comm().rank() == 0) ? &std::cout : &s;

    auto result = davidson<double, double_complex, davidson_evp_t::overlap>(Hk__, wf::num_bands(ncomp), wf::num_mag_dims(0),
            kp.singular_components_new(),
            [&](int i, int ispn){ return itsol_tol__; }, itso.residual_tolerance(), itso.num_steps(), itso.locking(),
            itso.subspace_size(), itso.converge_by_energy(), itso.extra_ortho(), *out, ctx_.verbosity() - 2);

    kp.message(2, __function_name__, "smallest eigen-value of the singular components: %20.16f\n", result.eval[0]);
    for (int i = 0; i < ncomp; i++) {
        kp.message(3, __function_name__, "singular component eigen-value[%i] : %20.16f\n", i, result.eval[i]);
    }
}

void Band::diag_full_potential_first_variation_davidson(Hamiltonian_k<double>& Hk__, double itsol_tol__) const
{
    PROFILE("sirius::Band::diag_fv_davidson");

    auto& kp = Hk__.kp();

    auto& itso = ctx_.cfg().iterative_solver();

    get_singular_components(Hk__, itsol_tol__);

    /* total number of local orbitals */
    int nlo = ctx_.unit_cell().mt_lo_basis_size();

    /* number of singular components */
    int ncomp = kp.singular_components_new().num_wf().get();

    //auto phi_extra = wave_function_factory(ctx_, kp, nlo + ncomp, 1, true);
    //phi_extra->pw_coeffs(0).zero(sddk::memory_t::host, 0, nlo + ncomp);
    //phi_extra->mt_coeffs(0).zero(sddk::memory_t::host, 0, nlo + ncomp);

    auto phi_extra_new = wave_function_factory(ctx_, kp, wf::num_bands(nlo + ncomp), wf::num_mag_dims(0), true);
    phi_extra_new->zero(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, nlo + ncomp));

    /* copy [0, ncomp) from kp.singular_components() to [0, ncomp) in phi_extra */
    wf::copy(sddk::memory_t::host, kp.singular_components_new(), wf::spin_index(0), wf::band_range(0, ncomp),
            *phi_extra_new, wf::spin_index(0), wf::band_range(0, ncomp));

    /* add pure local orbitals to the basis staring from ncomp index */
    if (nlo) {
        for (int ialoc = 0; ialoc < phi_extra_new->spl_num_atoms().local_size(); ialoc++) {
            int ia = phi_extra_new->spl_num_atoms()[ialoc];
            for (int xi = 0; xi < unit_cell_.atom(ia).mt_lo_basis_size(); xi++) {
                phi_extra_new->mt_coeffs(xi, wf::atom_index(ialoc), wf::spin_index(0),
                        wf::band_index(unit_cell_.atom(ia).offset_lo() + xi + ncomp)) = 1.0;
            }
        }
    }
    //if (is_device_memory(ctx_.preferred_memory_t())) {
    //    phi_extra->copy_to(sddk::spin_range(0), sddk::memory_t::device, 0, nlo + ncomp);
    //}
    //if (ctx_.cfg().control().print_checksum()) {
    //    phi_extra->print_checksum(get_device_t(phi_extra->preferred_memory_t()), "extra phi", 0, nlo + ncomp,
    //            RTE_OUT(std::cout));
    //}

    auto tolerance = [&](int j__, int ispn__) -> double {
        return itsol_tol__;
    };

    std::stringstream s;
    std::ostream* out = (kp.comm().rank() == 0) ? &std::cout : &s;
    auto result = davidson<double, std::complex<double>, davidson_evp_t::hamiltonian>(Hk__,
            wf::num_bands(ctx_.num_fv_states()), wf::num_mag_dims(0), kp.fv_eigen_vectors_slab_new(), tolerance,
            itso.residual_tolerance(), itso.num_steps(), itso.locking(), itso.subspace_size(),
            itso.converge_by_energy(), itso.extra_ortho(), *out, ctx_.verbosity() - 2,
            phi_extra_new.get());

    kp.set_fv_eigen_values(&result.eval[0]);
}

void Band::diag_full_potential_second_variation(Hamiltonian_k<double>& Hk__) const
{
    PROFILE("sirius::Band::diag_sv");

    auto& kp = Hk__.kp();

    if (!ctx_.need_sv()) {
        kp.bypass_sv();
        return;
    }

    sddk::mdarray<double, 2> band_energies(ctx_.num_bands(), ctx_.num_spinors());

    std::vector<int> num_mt_coeffs(ctx_.unit_cell().num_atoms());
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        num_mt_coeffs[ia] = ctx_.unit_cell().atom(ia).mt_basis_size();
    }

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<wf::Wave_functions<double>> hpsi;
    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
        hpsi.push_back(wf::Wave_functions<double>(kp.gkvec_sptr(), num_mt_coeffs, 
                    wf::num_mag_dims(0), wf::num_bands(ctx_.num_fv_states()), sddk::memory_t::host));
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2) {
        Hk__.apply_b(kp.fv_states_new(), hpsi);
    } else {
        hpsi[0].zero(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    //== if (ctx_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() == 3)
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    if (ctx_.so_correction()) {
        Hk__.H0().apply_so_correction(kp.fv_states_new(), hpsi);
    }

    int nfv = ctx_.num_fv_states();
    int bs  = ctx_.cyclic_block_size();

    //if (ctx_.processing_unit() == sddk::device_t::GPU) {
    //    kp.fv_states().allocate(sddk::spin_range(0), ctx_.mem_pool(sddk::memory_t::device));
    //    kp.fv_states().copy_to(sddk::spin_range(0), sddk::memory_t::device, 0, nfv);
    //    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
    //        hpsi[i].allocate(sddk::spin_range(0), ctx_.mem_pool(sddk::memory_t::device));
    //        hpsi[i].copy_to(sddk::spin_range(0), sddk::memory_t::device, 0, nfv);
    //    }
    //}

    ctx_.print_memory_usage(__FILE__, __LINE__);

    auto& std_solver = ctx_.std_evp_solver();

    wf::band_range br(0, nfv);
    wf::spin_range sr(0, 1);

    if (ctx_.num_mag_dims() != 3) {
        sddk::dmatrix<double_complex> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);
        if (ctx_.blacs_grid().comm().size() == 1 && ctx_.processing_unit() == sddk::device_t::GPU) {
            h.allocate(ctx_.mem_pool(sddk::memory_t::device));
        }
        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {

            auto cs1 =  kp.fv_states_new().checksum(sddk::memory_t::host, wf::spin_index(0), br);
            auto cs2 =  hpsi[ispn].checksum(sddk::memory_t::host, wf::spin_index(0), br);
            utils::print_checksum("fv_states", cs1, RTE_OUT(std::cout));
            utils::print_checksum("hpsi", cs2, RTE_OUT(std::cout));

            /* compute <wf_i | h * wf_j> */
            wf::inner(ctx_.spla_context(), sddk::memory_t::host, sr, kp.fv_states_new(), br, hpsi[ispn], br, h, 0, 0);

            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp.fv_eigen_value(i));
            }
            auto cs3 = h.checksum(nfv, nfv);
            utils::print_checksum("Hsv", cs3, RTE_OUT(std::cout));
            PROFILE("sirius::Band::diag_sv|stdevp");
            std_solver.solve(nfv, nfv, h, &band_energies(0, ispn), kp.sv_eigen_vectors(ispn));
        }
    } else {
        int nb = ctx_.num_bands();
        sddk::dmatrix<double_complex> h(nb, nb, ctx_.blacs_grid(), bs, bs);
        if (ctx_.blacs_grid().comm().size() == 1 && ctx_.processing_unit() == sddk::device_t::GPU) {
            h.allocate(ctx_.mem_pool(sddk::memory_t::device));
        }
        /* compute <wf_i | h * wf_j> for up-up block */
        wf::inner(ctx_.spla_context(), sddk::memory_t::host, sr, kp.fv_states_new(), br, hpsi[0], br, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        wf::inner(ctx_.spla_context(), sddk::memory_t::host, sr, kp.fv_states_new(), br, hpsi[1], br, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        wf::inner(ctx_.spla_context(), sddk::memory_t::host, sr, kp.fv_states_new(), br, hpsi[2], br, h, 0, nfv);

        if (kp.comm().size() == 1) {
            for (int i = 0; i < nfv; i++) {
                for (int j = 0; j < nfv; j++) {
                    h(nfv + j, i) = std::conj(h(i, nfv + j));
                }
            }
        } else {
            sddk::linalg(sddk::linalg_t::scalapack).tranc(nfv, nfv, h, 0, nfv, h, nfv, 0);
        }

        for (int i = 0; i < nfv; i++) {
            h.add(i, i, kp.fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp.fv_eigen_value(i));
        }
        PROFILE("sirius::Band::diag_sv|stdevp");
        std_solver.solve(nb, nb, h, &band_energies(0, 0), kp.sv_eigen_vectors(0));
    }

//    if (ctx_.processing_unit() == sddk::device_t::GPU) {
//        kp.fv_states().deallocate(sddk::spin_range(0), sddk::memory_t::device);
//        for (int i = 0; i < ctx_.num_mag_comp(); i++) {
//            hpsi[i].deallocate(sddk::spin_range(0), sddk::memory_t::device);
//        }
//    }
    for (int ispn = 0; ispn < ctx_.num_spinors(); ispn++) {
        for (int j = 0; j < ctx_.num_bands(); j++) {
            kp.band_energy(j, ispn, band_energies(j, ispn));
        }
    }
}

//inline int Band::diag_full_potential_single_variation(K_point& kp__, Hamiltonian& hamiltonian__) const
//{
//     if (kp->num_ranks() > 1 && !parameters_.gen_evp_solver()->parallel())
//         error_local(__FILE__, __LINE__, "eigen-value solver is not parallel");
//
//     mdarray<double_complex, 2> h(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//     mdarray<double_complex, 2> o(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//
//     set_o(kp, o);
//
//     std::vector<double> eval(parameters_.num_bands());
//     mdarray<double_complex, 2>& fd_evec = kp->fd_eigen_vectors();
//
//     if (parameters_.num_mag_dims() == 0)
//     {
//         assert(kp->gklo_basis_size() >= parameters_.num_fv_states());
//         set_h<nm>(kp, effective_potential, effective_magnetic_field, h);
//
//         Timer t2("sirius::Band::solve_fd|diag");
//         parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(),
//     kp->gklo_basis_size_col(),
//                                             parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(),
//                                             &eval[0], fd_evec.ptr(), fd_evec.ld());
//     }
//
//     if (parameters_.num_mag_dims() == 1)
//     {
//         assert(kp->gklo_basis_size() >= parameters_.num_fv_states());
//
//         mdarray<double_complex, 2> o1(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//         memcpy(&o1(0, 0), &o(0, 0), o.size() * sizeof(double_complex));
//
//         set_h<uu>(kp, effective_potential, effective_magnetic_field, h);
//
//         Timer t2("sirius::Band::solve_fd|diag");
//         parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(),
//     kp->gklo_basis_size_col(),
//                                             parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(),
//                                             &eval[0], &fd_evec(0, 0), fd_evec.ld());
//         t2.stop();
//
//         set_h<dd>(kp, effective_potential, effective_magnetic_field, h);
//
//         t2.start();
//         parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(),
//     kp->gklo_basis_size_col(),
//                                            parameters_.num_fv_states(), h.ptr(), h.ld(), o1.ptr(), o1.ld(),
//                                            &eval[parameters_.num_fv_states()],
//                                            &fd_evec(0, parameters_.spl_fv_states().local_size()), fd_evec.ld());
//        t2.stop();
//    }
//
//    kp->set_band_energies(&eval[0]);
//    return niter;
//}

} // namespace

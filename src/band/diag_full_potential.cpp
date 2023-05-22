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

    auto pcs = env::print_checksum();

    la::dmatrix<std::complex<double>> h(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, get_memory_pool(solver.host_memory_t()));
    la::dmatrix<std::complex<double>> o(ngklo, ngklo, ctx_.blacs_grid(), bs, bs, get_memory_pool(solver.host_memory_t()));

    /* setup Hamiltonian and overlap */
    Hk__.set_fv_h_o(h, o);

    if (ctx_.gen_evp_solver().type() == la::ev_solver_t::cusolver) {
        auto& mpd = get_memory_pool(sddk::memory_t::device);
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

    if (pcs) {
        auto z1 = h.checksum(ngklo, ngklo);
        auto z2 = o.checksum(ngklo, ngklo);
        utils::print_checksum("h_lapw", z1, ctx_.out());
        utils::print_checksum("o_lapw", z2, ctx_.out());
    }

    RTE_ASSERT(kp.gklo_basis_size() > ctx_.num_fv_states());

    std::vector<double> eval(ctx_.num_fv_states());

    print_memory_usage(ctx_.out(), FILE_LINE);
    if (solver.solve(kp.gklo_basis_size(), ctx_.num_fv_states(), h, o, eval.data(), kp.fv_eigen_vectors())) {
        RTE_THROW("error in generalized eigen-value problem");
    }
    print_memory_usage(ctx_.out(), FILE_LINE);

    if (ctx_.gen_evp_solver().type() == la::ev_solver_t::cusolver) {
        h.deallocate(sddk::memory_t::device);
        o.deallocate(sddk::memory_t::device);
        kp.fv_eigen_vectors().deallocate(sddk::memory_t::device);
    }
    kp.set_fv_eigen_values(&eval[0]);

    {
        rte::ostream out(kp.out(4), std::string(__func__));
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            out << "eval[" << i << "]=" << eval[i] << std::endl;
        }
    }

    if (pcs) {
        auto z1 = kp.fv_eigen_vectors().checksum(kp.gklo_basis_size(), ctx_.num_fv_states());
        utils::print_checksum("fv_eigen_vectors", z1, kp.out(1));
    }

    /* remap to slab */
    {
        /* G+k vector part */
        auto layout_in = kp.fv_eigen_vectors().grid_layout(0, 0, kp.gkvec().num_gvec(), ctx_.num_fv_states());
        auto layout_out = kp.fv_eigen_vectors_slab().grid_layout_pw(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<double>>::one(),
            la::constant<std::complex<double>>::zero(), kp.comm().native());
    }
    {
        /* muffin-tin part */
        auto layout_in = kp.fv_eigen_vectors().grid_layout(kp.gkvec().num_gvec(), 0,
                ctx_.unit_cell().mt_lo_basis_size(), ctx_.num_fv_states());
        auto layout_out = kp.fv_eigen_vectors_slab().grid_layout_mt(wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<double>>::one(),
            la::constant<std::complex<double>>::zero(), kp.comm().native());
    }

    if (pcs) {
        auto z1 = kp.fv_eigen_vectors_slab().checksum_pw(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        auto z2 = kp.fv_eigen_vectors_slab().checksum_mt(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
        utils::print_checksum("fv_eigen_vectors_slab", z1 + z2, kp.out(1));
    }

    /* renormalize wave-functions */
    if (ctx_.valence_relativity() == relativity_t::iora) {

        std::vector<int> num_mt_coeffs(unit_cell_.num_atoms());
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            num_mt_coeffs[ia] = unit_cell_.atom(ia).mt_lo_basis_size();
        }
        wf::Wave_functions<double> ofv_new(kp.gkvec_sptr(), num_mt_coeffs, wf::num_mag_dims(0),
                wf::num_bands(ctx_.num_fv_states()), sddk::memory_t::host);

        {
            auto mem = ctx_.processing_unit() == sddk::device_t::CPU ? sddk::memory_t::host : sddk::memory_t::device;
            auto mg1 = kp.fv_eigen_vectors_slab().memory_guard(mem, wf::copy_to::device);
            auto mg2 = ofv_new.memory_guard(mem, wf::copy_to::host);

            Hk__.apply_fv_h_o(false, false, wf::band_range(0, ctx_.num_fv_states()), kp.fv_eigen_vectors_slab(),
                    nullptr, &ofv_new);
        }

        auto norm1 = wf::inner_diag<double, std::complex<double>>(sddk::memory_t::host, kp.fv_eigen_vectors_slab(),
                ofv_new, wf::spin_range(0), wf::num_bands(ctx_.num_fv_states()));

        std::vector<double> norm;
        for (auto e : norm1) {
            norm.push_back(1 / std::sqrt(std::real(e)));
        }

        wf::axpby<double, double>(sddk::memory_t::host, wf::spin_range(0), wf::band_range(0, ctx_.num_fv_states()),
                nullptr, nullptr, norm.data(), &kp.fv_eigen_vectors_slab());
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

    //    la::dmatrix<std::complex<double>> hmlt(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
    //                                 ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
    //    la::dmatrix<std::complex<double>> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
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

    int ncomp = kp.singular_components().num_wf().get();

    ctx_.out(3, __func__) << "number of singular components: " << ncomp << std::endl;

    auto& itso = ctx_.cfg().iterative_solver();

    std::stringstream s;
    std::ostream* out = (kp.comm().rank() == 0) ? &std::cout : &s;

    auto result = davidson<double, std::complex<double>, davidson_evp_t::overlap>(Hk__, wf::num_bands(ncomp), wf::num_mag_dims(0),
            kp.singular_components(),
            [&](int i, int ispn){ return itsol_tol__; }, itso.residual_tolerance(), itso.num_steps(), itso.locking(),
            itso.subspace_size(), itso.converge_by_energy(), itso.extra_ortho(), *out, ctx_.verbosity() - 2);

    RTE_OUT(kp.out(2)) << "smallest eigen-value of the singular components: " << result.eval[0] << std::endl;
    for (int i = 0; i < ncomp; i++) {
        RTE_OUT(kp.out(3)) << "singular component eigen-value[" << i << "]=" << result.eval[i] << std::endl;
    }
}

void Band::diag_full_potential_first_variation_davidson(Hamiltonian_k<double>& Hk__, double itsol_tol__) const
{
    PROFILE("sirius::Band::diag_fv_davidson");

    auto& kp = Hk__.kp();

    auto& itso = ctx_.cfg().iterative_solver();

    /* number of singular components */
    int ncomp = kp.singular_components().num_wf().get();

    if (ncomp) {
        /* compute eigen-vectors of O^{APW-APW} */
        get_singular_components(Hk__, itsol_tol__);
    }

    /* total number of local orbitals */
    int nlo = ctx_.unit_cell().mt_lo_basis_size();

    auto phi_extra_new = wave_function_factory(ctx_, kp, wf::num_bands(nlo + ncomp), wf::num_mag_dims(0), true);
    phi_extra_new->zero(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, nlo + ncomp));

    if (ncomp) {
        /* copy [0, ncomp) from kp.singular_components() to [0, ncomp) in phi_extra */
        wf::copy(sddk::memory_t::host, kp.singular_components(), wf::spin_index(0), wf::band_range(0, ncomp),
                *phi_extra_new, wf::spin_index(0), wf::band_range(0, ncomp));
    }

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
    if (env::print_checksum()) {
        auto cs = phi_extra_new->checksum(sddk::memory_t::host, wf::band_range(0, nlo + ncomp));
        if (kp.comm().rank() == 0) {
            utils::print_checksum("phi_extra", cs, RTE_OUT(ctx_.out()));
        }
    }

    auto tolerance = [&](int j__, int ispn__) -> double {
        return itsol_tol__;
    };

    std::stringstream s;
    std::ostream* out = (kp.comm().rank() == 0) ? &std::cout : &s;

    auto result = davidson<double, std::complex<double>, davidson_evp_t::hamiltonian>(Hk__,
            wf::num_bands(ctx_.num_fv_states()), wf::num_mag_dims(0), kp.fv_eigen_vectors_slab(), tolerance,
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

    auto pcs = env::print_checksum();

    int nfv = ctx_.num_fv_states();
    int bs  = ctx_.cyclic_block_size();

    sddk::mdarray<double, 2> band_energies(ctx_.num_bands(), ctx_.num_spinors());

    std::vector<int> num_mt_coeffs(ctx_.unit_cell().num_atoms());
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        num_mt_coeffs[ia] = ctx_.unit_cell().atom(ia).mt_basis_size();
    }

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<wf::Wave_functions<double>> hpsi;
    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
        hpsi.push_back(wf::Wave_functions<double>(kp.gkvec_sptr(), num_mt_coeffs, 
                    wf::num_mag_dims(0), wf::num_bands(nfv), ctx_.host_memory_t()));
    }

    if (pcs) {
        auto cs1 = kp.fv_states().checksum_pw(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        auto cs2 = kp.fv_states().checksum_mt(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        if (kp.comm().rank() == 0) {
            utils::print_checksum("psi_pw", cs1, RTE_OUT(ctx_.out()));
            utils::print_checksum("psi_mt", cs2, RTE_OUT(ctx_.out()));
        }
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2) {
        Hk__.apply_b(kp.fv_states(), hpsi);
    } else {
        hpsi[0].zero(sddk::memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
    }

    print_memory_usage(ctx_.out(), FILE_LINE);

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
        Hk__.H0().apply_so_correction(kp.fv_states(), hpsi);
    }

    std::vector<wf::device_memory_guard> mg;
    mg.emplace_back(kp.fv_states().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device));
    for (int i = 0; i < ctx_.num_mag_comp(); i++) {
        mg.emplace_back(hpsi[i].memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device));
    }

    print_memory_usage(ctx_.out(), FILE_LINE);

    auto& std_solver = ctx_.std_evp_solver();

    wf::band_range br(0, nfv);
    wf::spin_range sr(0, 1);

    auto mem = ctx_.processing_unit_memory_t();

    if (ctx_.num_mag_dims() != 3) {
        la::dmatrix<std::complex<double>> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);
        if (ctx_.blacs_grid().comm().size() == 1 && ctx_.processing_unit() == sddk::device_t::GPU) {
            h.allocate(get_memory_pool(sddk::memory_t::device));
        }
        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (pcs) {
                auto cs1 = hpsi[ispn].checksum_pw(mem, wf::spin_index(0), wf::band_range(0, nfv));
                auto cs2 = hpsi[ispn].checksum_mt(mem, wf::spin_index(0), wf::band_range(0, nfv));
                if (kp.comm().rank() == 0) {
                    std::stringstream s1;
                    s1 << "hpsi_pw_" << ispn;
                    utils::print_checksum(s1.str(), cs1, RTE_OUT(ctx_.out()));
                    std::stringstream s2;
                    s2 << "hpsi_mt_" << ispn;
                    utils::print_checksum(s2.str(), cs2, RTE_OUT(ctx_.out()));
                }
            }
            /* compute <wf_i | h * wf_j> */
            wf::inner(ctx_.spla_context(), mem, sr, kp.fv_states(), br, hpsi[ispn], br, h, 0, 0);

            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp.fv_eigen_value(i));
            }
            PROFILE("sirius::Band::diag_sv|stdevp");
            std_solver.solve(nfv, nfv, h, &band_energies(0, ispn), kp.sv_eigen_vectors(ispn));
        }
    } else {
        int nb = ctx_.num_bands();
        la::dmatrix<std::complex<double>> h(nb, nb, ctx_.blacs_grid(), bs, bs);
        if (ctx_.blacs_grid().comm().size() == 1 && ctx_.processing_unit() == sddk::device_t::GPU) {
            h.allocate(get_memory_pool(sddk::memory_t::device));
        }
        /* compute <wf_i | h * wf_j> for up-up block */
        wf::inner(ctx_.spla_context(), mem, sr, kp.fv_states(), br, hpsi[0], br, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        wf::inner(ctx_.spla_context(), mem, sr, kp.fv_states(), br, hpsi[1], br, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        wf::inner(ctx_.spla_context(), mem, sr, kp.fv_states(), br, hpsi[2], br, h, 0, nfv);

        if (kp.comm().size() == 1) {
            for (int i = 0; i < nfv; i++) {
                for (int j = 0; j < nfv; j++) {
                    h(nfv + j, i) = std::conj(h(i, nfv + j));
                }
            }
        } else {
            la::wrap(la::lib_t::scalapack).tranc(nfv, nfv, h, 0, nfv, h, nfv, 0);
        }

        for (int i = 0; i < nfv; i++) {
            h.add(i, i, kp.fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp.fv_eigen_value(i));
        }
        PROFILE("sirius::Band::diag_sv|stdevp");
        std_solver.solve(nb, nb, h, &band_energies(0, 0), kp.sv_eigen_vectors(0));
    }

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
//     mdarray<std::complex<double>, 2> h(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//     mdarray<std::complex<double>, 2> o(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//
//     set_o(kp, o);
//
//     std::vector<double> eval(parameters_.num_bands());
//     mdarray<std::complex<double>, 2>& fd_evec = kp->fd_eigen_vectors();
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
//         mdarray<std::complex<double>, 2> o1(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
//         memcpy(&o1(0, 0), &o(0, 0), o.size() * sizeof(std::complex<double>));
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

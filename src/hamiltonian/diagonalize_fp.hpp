// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file diagonalize_fp.hpp
 *
 *  \brief Diagonalize full-potential LAPW Hamiltonian.
 */

#ifndef __DIAGONALIZE_FP_HPP__
#define __DIAGONALIZE_FP_HPP__

#include "davidson.hpp"
#include "k_point/k_point.hpp"

namespace sirius {

inline void
diagonalize_fp_fv_exact(Hamiltonian_k<float> const&, K_point<float>&)
{
    RTE_THROW("not implemented");
}

inline void
diagonalize_fp_fv_exact(Hamiltonian_k<double> const& Hk__, K_point<double>& kp__)
{
    PROFILE("sirius::diagonalize_fp_fv_exact");

    auto& ctx = Hk__.H0().ctx();

    auto& solver = ctx.gen_evp_solver();

    /* total eigen-value problem size */
    int ngklo = kp__.gklo_basis_size();

    /* block size of scalapack 2d block-cyclic distribution */
    int bs = ctx.cyclic_block_size();

    auto pcs = env::print_checksum();

    la::dmatrix<std::complex<double>> h(ngklo, ngklo, ctx.blacs_grid(), bs, bs,
                                        get_memory_pool(solver.host_memory_t()));
    la::dmatrix<std::complex<double>> o(ngklo, ngklo, ctx.blacs_grid(), bs, bs,
                                        get_memory_pool(solver.host_memory_t()));

    /* setup Hamiltonian and overlap */
    Hk__.set_fv_h_o(h, o);

    if (ctx.gen_evp_solver().type() == la::ev_solver_t::cusolver) {
        auto& mpd = get_memory_pool(memory_t::device);
        h.allocate(mpd);
        o.allocate(mpd);
        kp__.fv_eigen_vectors().allocate(mpd);
    }

    if (ctx.cfg().control().verification() >= 1) {
        double max_diff = check_hermitian(h, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "H matrix is not hermitian" << std::endl << "max error: " << max_diff;
            RTE_THROW(s);
        }
        max_diff = check_hermitian(o, ngklo);
        if (max_diff > 1e-12) {
            std::stringstream s;
            s << "O matrix is not hermitian" << std::endl << "max error: " << max_diff;
            RTE_THROW(s);
        }
    }

    if (pcs) {
        auto z1 = h.checksum(ngklo, ngklo);
        auto z2 = o.checksum(ngklo, ngklo);
        print_checksum("h_lapw", z1, ctx.out());
        print_checksum("o_lapw", z2, ctx.out());
    }

    RTE_ASSERT(kp__.gklo_basis_size() > ctx.num_fv_states());

    std::vector<double> eval(ctx.num_fv_states());

    print_memory_usage(ctx.out(), FILE_LINE);
    if (solver.solve(kp__.gklo_basis_size(), ctx.num_fv_states(), h, o, eval.data(), kp__.fv_eigen_vectors())) {
        RTE_THROW("error in generalized eigen-value problem");
    }
    print_memory_usage(ctx.out(), FILE_LINE);

    if (ctx.gen_evp_solver().type() == la::ev_solver_t::cusolver) {
        h.deallocate(memory_t::device);
        o.deallocate(memory_t::device);
        kp__.fv_eigen_vectors().deallocate(memory_t::device);
    }
    kp__.set_fv_eigen_values(&eval[0]);

    {
        rte::ostream out(kp__.out(4), std::string(__func__));
        for (int i = 0; i < ctx.num_fv_states(); i++) {
            out << "eval[" << i << "]=" << eval[i] << std::endl;
        }
    }

    if (pcs) {
        auto z1 = kp__.fv_eigen_vectors().checksum(kp__.gklo_basis_size(), ctx.num_fv_states());
        print_checksum("fv_eigen_vectors", z1, kp__.out(1));
    }

    /* remap to slab */
    {
        /* G+k vector part */
        auto layout_in = kp__.fv_eigen_vectors().grid_layout(0, 0, kp__.gkvec().num_gvec(), ctx.num_fv_states());
        auto layout_out =
                kp__.fv_eigen_vectors_slab().grid_layout_pw(wf::spin_index(0), wf::band_range(0, ctx.num_fv_states()));
        costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<double>>::one(),
                         la::constant<std::complex<double>>::zero(), kp__.comm().native());
    }
    if (ctx.unit_cell().mt_lo_basis_size()) {
        /* muffin-tin part */
        auto layout_in = kp__.fv_eigen_vectors().grid_layout(kp__.gkvec().num_gvec(), 0,
                                                             ctx.unit_cell().mt_lo_basis_size(), ctx.num_fv_states());
        auto layout_out =
                kp__.fv_eigen_vectors_slab().grid_layout_mt(wf::spin_index(0), wf::band_range(0, ctx.num_fv_states()));
        costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<double>>::one(),
                         la::constant<std::complex<double>>::zero(), kp__.comm().native());
    }

    if (pcs) {
        auto z1 = kp__.fv_eigen_vectors_slab().checksum_pw(memory_t::host, wf::spin_index(0),
                                                           wf::band_range(0, ctx.num_fv_states()));
        auto z2 = kp__.fv_eigen_vectors_slab().checksum_mt(memory_t::host, wf::spin_index(0),
                                                           wf::band_range(0, ctx.num_fv_states()));
        print_checksum("fv_eigen_vectors_slab", z1 + z2, kp__.out(1));
    }

    /* renormalize wave-functions */
    if (ctx.valence_relativity() == relativity_t::iora) {

        std::vector<int> num_mt_coeffs(ctx.unit_cell().num_atoms());
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            num_mt_coeffs[ia] = ctx.unit_cell().atom(ia).mt_lo_basis_size();
        }
        wf::Wave_functions<double> ofv_new(kp__.gkvec_sptr(), num_mt_coeffs, wf::num_mag_dims(0),
                                           wf::num_bands(ctx.num_fv_states()), memory_t::host);

        {
            auto mem = ctx.processing_unit() == device_t::CPU ? memory_t::host : memory_t::device;
            auto mg1 = kp__.fv_eigen_vectors_slab().memory_guard(mem, wf::copy_to::device);
            auto mg2 = ofv_new.memory_guard(mem, wf::copy_to::host);

            Hk__.apply_fv_h_o(false, false, wf::band_range(0, ctx.num_fv_states()), kp__.fv_eigen_vectors_slab(),
                              nullptr, &ofv_new);
        }

        auto norm1 =
                wf::inner_diag<double, std::complex<double>>(memory_t::host, kp__.fv_eigen_vectors_slab(), ofv_new,
                                                             wf::spin_range(0), wf::num_bands(ctx.num_fv_states()));

        std::vector<double> norm;
        for (auto e : norm1) {
            norm.push_back(1 / std::sqrt(std::real(e)));
        }

        wf::axpby<double, double>(memory_t::host, wf::spin_range(0), wf::band_range(0, ctx.num_fv_states()), nullptr,
                                  nullptr, norm.data(), &kp__.fv_eigen_vectors_slab());
    }

    // if (ctx.cfg().control().verification() >= 2) {
    //     kp.message(1, __function_name__, "%s", "checking application of H and O\n");
    //     /* check application of H and O */
    //     sddk::Wave_functions<double> hphi(kp.gkvec_partition(), unit_cell_.num_atoms(),
    //                         [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx.num_fv_states(),
    //                         ctx.preferred_memory_t());
    //     sddk::Wave_functions<double> ophi(kp.gkvec_partition(), unit_cell_.num_atoms(),
    //                         [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx.num_fv_states(),
    //                         ctx.preferred_memory_t());

    //    if (ctx.processing_unit() == sddk::device_t::GPU) {
    //        kp.fv_eigen_vectors_slab().allocate(sddk::spin_range(0), memory_t::device);
    //        kp.fv_eigen_vectors_slab().copy_to(sddk::spin_range(0), memory_t::device, 0, ctx.num_fv_states());
    //        hphi.allocate(sddk::spin_range(0), memory_t::device);
    //        ophi.allocate(sddk::spin_range(0), memory_t::device);
    //    }

    //    Hk__.apply_fv_h_o(false, false, 0, ctx.num_fv_states(), kp.fv_eigen_vectors_slab(), &hphi, &ophi);

    //    la::dmatrix<std::complex<double>> hmlt(ctx.num_fv_states(), ctx.num_fv_states(), ctx.blacs_grid(),
    //                                 ctx.cyclic_block_size(), ctx.cyclic_block_size());
    //    la::dmatrix<std::complex<double>> ovlp(ctx.num_fv_states(), ctx.num_fv_states(), ctx.blacs_grid(),
    //                                 ctx.cyclic_block_size(), ctx.cyclic_block_size());

    //    inner(ctx.spla_context(), sddk::spin_range(0), kp.fv_eigen_vectors_slab(), 0, ctx.num_fv_states(),
    //          hphi, 0, ctx.num_fv_states(), hmlt, 0, 0);
    //    inner(ctx.spla_context(), sddk::spin_range(0), kp.fv_eigen_vectors_slab(), 0, ctx.num_fv_states(),
    //          ophi, 0, ctx.num_fv_states(), ovlp, 0, 0);

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

inline void
get_singular_components(Hamiltonian_k<double> const& Hk__, K_point<double>& kp__, double itsol_tol__)
{
    PROFILE("sirius::get_singular_components");

    auto& ctx = Hk__.H0().ctx();

    int ncomp = kp__.singular_components().num_wf().get();

    RTE_OUT(ctx.out(3)) << "number of singular components: " << ncomp << std::endl;

    auto& itso = ctx.cfg().iterative_solver();

    std::stringstream s;
    std::ostream* out = (kp__.comm().rank() == 0) ? &std::cout : &s;

    auto result = davidson<double, std::complex<double>, davidson_evp_t::overlap>(
            Hk__, kp__, wf::num_bands(ncomp), wf::num_mag_dims(0), kp__.singular_components(),
            [&](int i, int ispn) { return itsol_tol__; }, itso.residual_tolerance(), itso.num_steps(), itso.locking(),
            itso.subspace_size(), itso.converge_by_energy(), itso.extra_ortho(), *out, ctx.verbosity() - 2);

    RTE_OUT(kp__.out(2)) << "smallest eigen-value of the singular components: " << result.eval[0] << std::endl;
    for (int i = 0; i < ncomp; i++) {
        RTE_OUT(kp__.out(3)) << "singular component eigen-value[" << i << "]=" << result.eval[i] << std::endl;
    }
}

inline void
diagonalize_fp_fv_davidson(Hamiltonian_k<float> const&, K_point<float>&, double)
{
    RTE_THROW("not implemented");
}

inline void
diagonalize_fp_fv_davidson(Hamiltonian_k<double> const& Hk__, K_point<double>& kp__, double itsol_tol__)
{
    PROFILE("sirius::diagonalize_fp_fv_davidson");

    auto& ctx = Hk__.H0().ctx();

    auto& itso = ctx.cfg().iterative_solver();

    /* number of singular components */
    int ncomp = kp__.singular_components().num_wf().get();

    if (ncomp) {
        /* compute eigen-vectors of O^{APW-APW} */
        get_singular_components(Hk__, kp__, itsol_tol__);
    }

    /* total number of local orbitals */
    int nlo = ctx.unit_cell().mt_lo_basis_size();

    auto phi_extra_new = wave_function_factory(ctx, kp__, wf::num_bands(nlo + ncomp), wf::num_mag_dims(0), true);
    phi_extra_new->zero(memory_t::host, wf::spin_index(0), wf::band_range(0, nlo + ncomp));

    if (ncomp) {
        /* copy [0, ncomp) from kp.singular_components() to [0, ncomp) in phi_extra */
        wf::copy(memory_t::host, kp__.singular_components(), wf::spin_index(0), wf::band_range(0, ncomp),
                 *phi_extra_new, wf::spin_index(0), wf::band_range(0, ncomp));
    }

    std::vector<int> offset_lo(ctx.unit_cell().num_atoms());
    std::generate(offset_lo.begin(), offset_lo.end(), [n = 0, ia = 0, &ctx]() mutable {
        int offs = n;
        n += ctx.unit_cell().atom(ia++).mt_lo_basis_size();
        return offs;
    });

    /* add pure local orbitals to the basis staring from ncomp index */
    if (nlo) {
        for (auto it : phi_extra_new->spl_num_atoms()) {
            for (int xi = 0; xi < ctx.unit_cell().atom(it.i).mt_lo_basis_size(); xi++) {
                phi_extra_new->mt_coeffs(xi, it.li, wf::spin_index(0), wf::band_index(offset_lo[it.i] + xi + ncomp)) =
                        1.0;
            }
        }
    }
    if (env::print_checksum()) {
        auto cs = phi_extra_new->checksum(memory_t::host, wf::band_range(0, nlo + ncomp));
        if (kp__.comm().rank() == 0) {
            print_checksum("phi_extra", cs, RTE_OUT(ctx.out()));
        }
    }

    auto tolerance = [&](int j__, int ispn__) -> double { return itsol_tol__; };

    std::stringstream s;
    std::ostream* out = (kp__.comm().rank() == 0) ? &std::cout : &s;

    auto result = davidson<double, std::complex<double>, davidson_evp_t::hamiltonian>(
            Hk__, kp__, wf::num_bands(ctx.num_fv_states()), wf::num_mag_dims(0), kp__.fv_eigen_vectors_slab(),
            tolerance, itso.residual_tolerance(), itso.num_steps(), itso.locking(), itso.subspace_size(),
            itso.converge_by_energy(), itso.extra_ortho(), *out, ctx.verbosity() - 2, phi_extra_new.get());

    kp__.set_fv_eigen_values(&result.eval[0]);
}

inline void
diagonalize_fp_sv(Hamiltonian_k<float> const&, K_point<float>&)
{
    RTE_THROW("not implemented");
}

/// Diagonalize second-variational Hamiltonian.
inline void
diagonalize_fp_sv(Hamiltonian_k<double> const& Hk__, K_point<double>& kp)
{
    PROFILE("sirius::diagonalize_fp_sv");

    // auto& kp = Hk__.kp();
    auto& ctx = Hk__.H0().ctx();

    if (!ctx.need_sv()) {
        kp.bypass_sv();
        return;
    }

    auto pcs = env::print_checksum();

    int nfv = ctx.num_fv_states();
    int bs  = ctx.cyclic_block_size();

    mdarray<double, 2> band_energies({ctx.num_bands(), ctx.num_spinors()});

    std::vector<int> num_mt_coeffs(ctx.unit_cell().num_atoms());
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        num_mt_coeffs[ia] = ctx.unit_cell().atom(ia).mt_basis_size();
    }

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<wf::Wave_functions<double>> hpsi;
    for (int i = 0; i < ctx.num_mag_comp(); i++) {
        hpsi.push_back(wf::Wave_functions<double>(kp.gkvec_sptr(), num_mt_coeffs, wf::num_mag_dims(0),
                                                  wf::num_bands(nfv), ctx.host_memory_t()));
    }

    if (pcs) {
        auto cs1 = kp.fv_states().checksum_pw(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        auto cs2 = kp.fv_states().checksum_mt(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
        if (kp.comm().rank() == 0) {
            print_checksum("psi_pw", cs1, RTE_OUT(ctx.out()));
            print_checksum("psi_mt", cs2, RTE_OUT(ctx.out()));
        }
    }

    /* compute product of magnetic field and wave-function */
    if (ctx.num_spins() == 2) {
        Hk__.apply_b(kp.fv_states(), hpsi);
    } else {
        hpsi[0].zero(memory_t::host, wf::spin_index(0), wf::band_range(0, nfv));
    }

    print_memory_usage(ctx.out(), FILE_LINE);

    //== if (ctx.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx.num_mag_dims() == 3)
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    if (ctx.so_correction()) {
        Hk__.H0().apply_so_correction(kp.fv_states(), hpsi);
    }

    std::vector<wf::device_memory_guard> mg;
    mg.emplace_back(kp.fv_states().memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::device));
    for (int i = 0; i < ctx.num_mag_comp(); i++) {
        mg.emplace_back(hpsi[i].memory_guard(ctx.processing_unit_memory_t(), wf::copy_to::device));
    }

    print_memory_usage(ctx.out(), FILE_LINE);

    auto& std_solver = ctx.std_evp_solver();

    wf::band_range br(0, nfv);
    wf::spin_range sr(0, 1);

    auto mem = ctx.processing_unit_memory_t();

    if (ctx.num_mag_dims() != 3) {
        la::dmatrix<std::complex<double>> h(nfv, nfv, ctx.blacs_grid(), bs, bs);
        if (ctx.blacs_grid().comm().size() == 1 && ctx.processing_unit() == device_t::GPU) {
            h.allocate(get_memory_pool(memory_t::device));
        }
        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            if (pcs) {
                auto cs1 = hpsi[ispn].checksum_pw(mem, wf::spin_index(0), wf::band_range(0, nfv));
                auto cs2 = hpsi[ispn].checksum_mt(mem, wf::spin_index(0), wf::band_range(0, nfv));
                if (kp.comm().rank() == 0) {
                    std::stringstream s1;
                    s1 << "hpsi_pw_" << ispn;
                    print_checksum(s1.str(), cs1, RTE_OUT(ctx.out()));
                    std::stringstream s2;
                    s2 << "hpsi_mt_" << ispn;
                    print_checksum(s2.str(), cs2, RTE_OUT(ctx.out()));
                }
            }
            /* compute <wf_i | h * wf_j> */
            wf::inner(ctx.spla_context(), mem, sr, kp.fv_states(), br, hpsi[ispn], br, h, 0, 0);

            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp.fv_eigen_value(i));
            }
            PROFILE("sirius::diagonalize_fp_sv|stdevp");
            std_solver.solve(nfv, nfv, h, &band_energies(0, ispn), kp.sv_eigen_vectors(ispn));
        }
    } else {
        int nb = ctx.num_bands();
        la::dmatrix<std::complex<double>> h(nb, nb, ctx.blacs_grid(), bs, bs);
        if (ctx.blacs_grid().comm().size() == 1 && ctx.processing_unit() == device_t::GPU) {
            h.allocate(get_memory_pool(memory_t::device));
        }
        /* compute <wf_i | h * wf_j> for up-up block */
        wf::inner(ctx.spla_context(), mem, sr, kp.fv_states(), br, hpsi[0], br, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        wf::inner(ctx.spla_context(), mem, sr, kp.fv_states(), br, hpsi[1], br, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        wf::inner(ctx.spla_context(), mem, sr, kp.fv_states(), br, hpsi[2], br, h, 0, nfv);

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
        PROFILE("sirius::diagonalize_fp_sv|stdevp");
        std_solver.solve(nb, nb, h, &band_energies(0, 0), kp.sv_eigen_vectors(0));
    }

    for (int ispn = 0; ispn < ctx.num_spinors(); ispn++) {
        for (int j = 0; j < ctx.num_bands(); j++) {
            kp.band_energy(j, ispn, band_energies(j, ispn));
        }
    }
}

/// Diagonalize a full-potential LAPW Hamiltonian.
template <typename T>
inline void
diagonalize_fp(Hamiltonian_k<T> const& Hk__, K_point<T>& kp__, double itsol_tol__)
{
    auto& ctx = Hk__.H0().ctx();
    print_memory_usage(ctx.out(), FILE_LINE);
    if (ctx.cfg().control().use_second_variation()) {
        /* solve non-magnetic Hamiltonian (so-called first variation) */
        auto& itso = ctx.cfg().iterative_solver();
        if (itso.type() == "exact") {
            diagonalize_fp_fv_exact(Hk__, kp__);
        } else if (itso.type() == "davidson") {
            diagonalize_fp_fv_davidson(Hk__, kp__, itsol_tol__);
        }
        /* generate first-variational states */
        kp__.generate_fv_states();
        /* solve magnetic Hamiltonian */
        diagonalize_fp_sv(Hk__, kp__);
        /* generate spinor wave-functions */
        kp__.generate_spinor_wave_functions();
    } else {
        RTE_THROW("not implemented");
        // diag_full_potential_single_variation();
    }
    print_memory_usage(ctx.out(), FILE_LINE);
}

} // namespace sirius

#endif

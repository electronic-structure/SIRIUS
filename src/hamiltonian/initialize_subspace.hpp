/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file initialize_subspace.hpp
 *
 *  \brief Create intial subspace from atomic-like wave-functions
 */

#ifndef __INITIALIZE_SUBSPACE_HPP__
#define __INITIALIZE_SUBSPACE_HPP__

#include "k_point/k_point_set.hpp"
#include "diagonalize_pp.hpp"

namespace sirius {

/// Initialize the wave-functions subspace at a given k-point.
/** If the number of atomic orbitals is smaller than the number of bands, the rest of the initial wave-functions
 *  are created from the random numbers. */
template <typename T, typename F>
inline void
initialize_subspace(Hamiltonian_k<T> const& Hk__, K_point<T>& kp__, int num_ao__)
{
    PROFILE("sirius::initialize_subspace|kp");

    auto& ctx = Hk__.H0().ctx();

    if (ctx.cfg().control().verification() >= 2) {
        auto eval = diag_S_davidson<T, F>(Hk__, kp__);
        if (eval[0] <= 0) {
            std::stringstream s;
            s << "S-operator matrix is not positive definite\n"
              << "  lowest eigen-value: " << eval[0];
            RTE_WARNING(s);
        } else {
            std::stringstream s;
            s << "S-matrix is OK! Minimum eigen-value : " << eval[0];
            RTE_OUT(ctx.out(1)) << s.str() << std::endl;
        }
    }

    auto pcs = env::print_checksum();

    /* number of non-zero spin components */
    const int num_sc = (ctx.num_mag_dims() == 3) ? 2 : 1;

    /* short notation for number of target wave-functions */
    int num_bands = ctx.num_bands();

    /* number of basis functions */
    int num_phi = std::max(num_ao__, num_bands / num_sc);

    int num_phi_tot = num_phi * num_sc;

    auto& mp = get_memory_pool(ctx.host_memory_t());

    print_memory_usage(ctx.out(), FILE_LINE);

    /* initial basis functions */
    wf::Wave_functions<T> phi(kp__.gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
                              wf::num_bands(num_phi_tot), memory_t::host);

    for (int ispn = 0; ispn < num_sc; ispn++) {
        phi.zero(memory_t::host, wf::spin_index(ispn), wf::band_range(0, num_phi_tot));
    }

    /* generate the initial atomic wavefunctions */
    std::vector<int> atoms(ctx.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0);
    kp__.generate_atomic_wave_functions(
            atoms, [&](int iat) { return &ctx.unit_cell().atom_type(iat).indexb_wfs(); }, *ctx.ri().ps_atomic_wf_, phi);

    /* generate some random noise */
    std::vector<T> tmp(4096);
    random_uint32(true);
    for (int i = 0; i < 4096; i++) {
        tmp[i] = 1e-5 * random<T>();
    }
    PROFILE_START("sirius::initialize_subspace|kp|wf");
    /* fill remaining wave-functions with pseudo-random guess */
    RTE_ASSERT(kp__.num_gkvec() > num_phi + 10);
    #pragma omp parallel
    {
        for (int i = 0; i < num_phi - num_ao__; i++) {
            #pragma omp for schedule(static) nowait
            for (int igk_loc = 0; igk_loc < kp__.num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = kp__.gkvec().offset() + igk_loc; // Hk__.kp().idxgk(igk_loc);
                if (igk == i + 1) {
                    phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(num_ao__ + i)) = 1.0;
                }
                if (igk == i + 2) {
                    phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(num_ao__ + i)) = 0.5;
                }
                if (igk == i + 3) {
                    phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(num_ao__ + i)) = 0.25;
                }
            }
        }
        /* add random noise */
        for (int i = 0; i < num_phi; i++) {
            #pragma omp for schedule(static) nowait
            for (int igk_loc = kp__.gkvec().skip_g0(); igk_loc < kp__.num_gkvec_loc(); igk_loc++) {
                /* global index of G+k vector */
                int igk = kp__.gkvec().offset() + igk_loc;
                phi.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(i)) += tmp[igk & 0xFFF];
            }
        }
    }

    if (ctx.num_mag_dims() == 3) {
        /* make pure spinor up- and dn- wave functions */
        wf::copy(memory_t::host, phi, wf::spin_index(0), wf::band_range(0, num_phi), phi, wf::spin_index(1),
                 wf::band_range(num_phi, num_phi_tot));
    }
    PROFILE_STOP("sirius::initialize_subspace|kp|wf");

    /* allocate wave-functions */
    wf::Wave_functions<T> hphi(kp__.gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
                               wf::num_bands(num_phi_tot), memory_t::host);
    wf::Wave_functions<T> ophi(kp__.gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
                               wf::num_bands(num_phi_tot), memory_t::host);
    /* temporary wave-functions required as a storage during orthogonalization */
    wf::Wave_functions<T> wf_tmp(kp__.gkvec_sptr(), wf::num_mag_dims(ctx.num_mag_dims() == 3 ? 3 : 0),
                                 wf::num_bands(num_phi_tot), memory_t::host);

    int bs = ctx.cyclic_block_size();

    auto& gen_solver = ctx.gen_evp_solver();

    la::dmatrix<F> hmlt(num_phi_tot, num_phi_tot, ctx.blacs_grid(), bs, bs, mp);
    la::dmatrix<F> ovlp(num_phi_tot, num_phi_tot, ctx.blacs_grid(), bs, bs, mp);
    la::dmatrix<F> evec(num_phi_tot, num_phi_tot, ctx.blacs_grid(), bs, bs, mp);

    std::vector<real_type<F>> eval(num_bands);

    print_memory_usage(ctx.out(), FILE_LINE);

    auto mem = ctx.processing_unit() == device_t::CPU ? memory_t::host : memory_t::device;

    std::vector<wf::device_memory_guard> mg;
    mg.emplace_back(kp__.spinor_wave_functions().memory_guard(mem, wf::copy_to::host));
    mg.emplace_back(phi.memory_guard(mem, wf::copy_to::device));
    mg.emplace_back(hphi.memory_guard(mem));
    mg.emplace_back(ophi.memory_guard(mem));
    mg.emplace_back(wf_tmp.memory_guard(mem));

    if (is_device_memory(mem)) {
        auto& mpd = get_memory_pool(mem);
        evec.allocate(mpd);
        hmlt.allocate(mpd);
        ovlp.allocate(mpd);
    }

    print_memory_usage(ctx.out(), FILE_LINE);

    if (pcs) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            auto cs = phi.checksum(mem, wf::spin_index(ispn), wf::band_range(0, num_phi_tot));
            if (kp__.comm().rank() == 0) {
                std::stringstream s;
                s << "initial_phi" << ispn;
                print_checksum(s.str(), cs, RTE_OUT(std::cout));
            }
        }
    }

    for (int ispn_step = 0; ispn_step < ctx.num_spinors(); ispn_step++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        Hk__.template apply_h_s<F>(ctx.num_mag_dims() == 3 ? wf::spin_range(0, 2) : wf::spin_range(ispn_step),
                                   wf::band_range(0, num_phi_tot), phi, &hphi, &ophi);

        /* do some checks */
        //    if (ctx_.cfg().control().verification() >= 1) {

        //        set_subspace_mtrx<T>(0, num_phi_tot, 0, phi, ophi, ovlp);
        //        if (ctx_.cfg().control().verification() >= 2 && ctx_.verbosity() >= 2) {
        //            auto s = ovlp.serialize("overlap", num_phi_tot, num_phi_tot);
        //            if (Hk__.kp().comm().rank() == 0) {
        //                ctx_.out() << s.str() << std::endl;
        //            }
        //        }

        //        double max_diff = check_hermitian(ovlp, num_phi_tot);
        //        if (max_diff > 1e-12) {
        //            std::stringstream s;
        //            s << "overlap matrix is not hermitian, max_err = " << max_diff;
        //            RTE_WARNING(s);
        //        }
        //        std::vector<real_type<T>> eo(num_phi_tot);
        //        auto& std_solver = ctx_.std_evp_solver();
        //        if (std_solver.solve(num_phi_tot, num_phi_tot, ovlp, eo.data(), evec)) {
        //            std::stringstream s;
        //            s << "error in diagonalization";
        //            RTE_WARNING(s);
        //        }
        //        Hk__.kp().message(1, __function_name__, "minimum eigen-value of the overlap matrix: %18.12f\n",
        //        eo[0]); if (eo[0] < 0) {
        //            RTE_WARNING("overlap matrix is not positively defined");
        //        }
        //    }

        /* setup eigen-value problem */
        generate_subspace_matrix(ctx, 0, num_phi_tot, 0, phi, hphi, hmlt);
        generate_subspace_matrix(ctx, 0, num_phi_tot, 0, phi, ophi, ovlp);

        if (pcs) {
            auto cs1 = hmlt.checksum(num_phi_tot, num_phi_tot);
            auto cs2 = ovlp.checksum(num_phi_tot, num_phi_tot);
            if (kp__.comm().rank() == 0) {
                print_checksum("hmlt", cs1, RTE_OUT(std::cout));
                print_checksum("ovlp", cs2, RTE_OUT(std::cout));
            }
        }

        //    if (ctx_.cfg().control().verification() >= 2 && ctx_.verbosity() >= 2) {
        //        auto s1 = hmlt.serialize("hmlt", num_phi_tot, num_phi_tot);
        //        auto s2 = hmlt.serialize("ovlp", num_phi_tot, num_phi_tot);
        //        if (Hk__.kp().comm().rank() == 0) {
        //            ctx_.out() << s1.str() << std::endl << s2.str() << std::endl;
        //        }
        //    }

        /* solve generalized eigen-value problem with the size N and get lowest num_bands eigen-vectors */
        if (gen_solver.solve(num_phi_tot, num_bands, hmlt, ovlp, eval.data(), evec)) {
            RTE_THROW("error in diagonalization");
        }

        //    if (ctx_.print_checksum()) {
        //        auto cs = evec.checksum(num_phi_tot, num_bands);
        //        real_type<T> cs1{0};
        //        for (int i = 0; i < num_bands; i++) {
        //            cs1 += eval[i];
        //        }
        //        if (Hk__.kp().comm().rank() == 0) {
        //            utils::print_checksum("evec", cs);
        //            utils::print_checksum("eval", cs1);
        //        }
        //    }
        {
            rte::ostream out(kp__.out(3), std::string(__func__));
            for (int i = 0; i < num_bands; i++) {
                out << "eval[" << i << "]=" << eval[i] << std::endl;
            }
        }

        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        for (int ispn = 0; ispn < num_sc; ispn++) {
            wf::transform(ctx.spla_context(), mem, evec, 0, 0, 1.0, phi, wf::spin_index(num_sc == 2 ? ispn : 0),
                          wf::band_range(0, num_phi_tot), 0.0, kp__.spinor_wave_functions(),
                          wf::spin_index(num_sc == 2 ? ispn : ispn_step), wf::band_range(0, num_bands));
        }

        for (int j = 0; j < num_bands; j++) {
            kp__.band_energy(j, ispn_step, eval[j]);
        }
    }

    if (pcs) {
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            auto cs = kp__.spinor_wave_functions().checksum(mem, wf::spin_index(ispn), wf::band_range(0, num_bands));
            std::stringstream s;
            s << "initial_spinor_wave_functions_" << ispn;
            if (kp__.comm().rank() == 0) {
                print_checksum(s.str(), cs, RTE_OUT(std::cout));
            }
        }
    }
}

template <typename T>
void
initialize_subspace(K_point_set& kset__, Hamiltonian0<T>& H0__)
{
    PROFILE("sirius::initialize_subspace_kset");

    int N{0};

    auto& ctx = H0__.ctx();

    if (ctx.cfg().iterative_solver().init_subspace() == "lcao") {
        /* get the total number of atomic-centered orbitals */
        N = ctx.unit_cell().num_ps_atomic_wf().first;
    }

    for (auto it : kset__.spl_num_kpoints()) {
        auto kp = kset__.get<T>(it.i);
        auto Hk = H0__(*kp);
        if (ctx.gamma_point() && (ctx.so_correction() == false)) {
            initialize_subspace<T, T>(Hk, *kp, N);
        } else {
            initialize_subspace<T, std::complex<T>>(Hk, *kp, N);
        }
    }

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
        for (int ispn = 0; ispn < ctx.num_spinors(); ispn++) {
            for (int i = 0; i < ctx.num_bands(); i++) {
                kset__.get<T>(ik)->band_energy(i, ispn, 0);
                kset__.get<T>(ik)->band_occupancy(i, ispn, ctx.max_occupancy());
            }
        }
    }
}

} // namespace sirius

#endif

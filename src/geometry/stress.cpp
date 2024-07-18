/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file stress.cpp
 *
 *  \brief Contains implementation of sirius::Stress class.
 */

#include "core/r3/r3.hpp"
#include "core/profiler.hpp"
#include "k_point/k_point.hpp"
#include "non_local_functor.hpp"
#include "dft/energy.hpp"
#include "symmetry/symmetrize_stress_tensor.hpp"
#include "stress.hpp"

namespace sirius {

template <typename T, typename F>
void
Stress::calc_stress_nonloc_aux()
{
    PROFILE("sirius::Stress|nonloc");

    mdarray<real_type<F>, 2> collect_result({9, ctx_.unit_cell().num_atoms()});
    collect_result.zero();

    stress_nonloc_.zero();

    /* if there are no beta projectors then get out there */
    if (ctx_.unit_cell().max_mt_basis_size() == 0) {
        return;
    }

    print_memory_usage(ctx_.out(), FILE_LINE);

    for (auto it : kset_.spl_num_kpoints()) {
        auto kp  = kset_.get<T>(it.i);
        auto mem = ctx_.processing_unit_memory_t();
        auto mg  = kp->spinor_wave_functions().memory_guard(mem, wf::copy_to::device);
        Beta_projectors_strain_deriv<T> bp_strain_deriv(ctx_, kp->gkvec());

        add_k_point_contribution_nonlocal<T, F>(ctx_, bp_strain_deriv, *kp, collect_result);
    }

    #pragma omp parallel
    {
        r3::matrix<double> tmp_stress; // TODO: test pragma omp parallel for reduction(+:stress)

        #pragma omp for
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    tmp_stress(i, j) -= collect_result(j * 3 + i, ia);
                }
            }
        }

        #pragma omp critical
        stress_nonloc_ += tmp_stress;
    }

    ctx_.comm().allreduce(&stress_nonloc_(0, 0), 9);

    stress_nonloc_ *= (1.0 / ctx_.unit_cell().omega());

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_nonloc_);
}

r3::matrix<double>
Stress::calc_stress_total()
{
    calc_stress_kin();
    calc_stress_har();
    calc_stress_ewald();
    calc_stress_vloc();
    calc_stress_core();
    calc_stress_xc();
    calc_stress_us();
    calc_stress_nonloc();
    stress_hubbard_.zero();
    if (ctx_.hubbard_correction()) {
        calc_stress_hubbard();
    }
    stress_total_.zero();

    for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {
            stress_total_(mu, nu) = stress_kin_(mu, nu) + stress_har_(mu, nu) + stress_ewald_(mu, nu) +
                                    stress_vloc_(mu, nu) + stress_core_(mu, nu) + stress_xc_(mu, nu) +
                                    stress_us_(mu, nu) + stress_nonloc_(mu, nu) + stress_hubbard_(mu, nu);
        }
    }
    return stress_total_;
}

r3::matrix<double>
Stress::calc_stress_hubbard()
{
    stress_hubbard_.zero();
    auto r = ctx_.unit_cell().num_hubbard_wf();
    /* if there are no beta projectors then get out there */
    /* TODO : Need to fix the case where pp have no beta projectors */
    if (ctx_.unit_cell().max_mt_basis_size() == 0) {
        RTE_THROW("Hubbard forces : Your pseudo potentials do not have beta projectors. This need a proper fix");
        return stress_hubbard_;
    }

    Q_operator<double> q_op(ctx_);

    auto nhwf = ctx_.unit_cell().num_hubbard_wf().first;

    mdarray<std::complex<double>, 4> dn({nhwf, nhwf, 2, 9});
    if (is_device_memory(ctx_.processing_unit_memory_t())) {
        dn.allocate(ctx_.processing_unit_memory_t());
    }
    for (auto it : kset_.spl_num_kpoints()) {
        auto kp = kset_.get<double>(it.i);
        dn.zero();
        if (is_device_memory(ctx_.processing_unit_memory_t())) {
            dn.zero(ctx_.processing_unit_memory_t());
        }
        auto mg1 = kp->spinor_wave_functions().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
        auto mg2 = kp->hubbard_wave_functions_S().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
        auto mg3 = kp->atomic_wave_functions().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);
        auto mg4 = kp->atomic_wave_functions_S().memory_guard(ctx_.processing_unit_memory_t(), wf::copy_to::device);

        if (ctx_.num_mag_dims() == 3) {
            RTE_THROW("Hubbard stress correction is only implemented for the simple hubbard correction.");
        }

        /* compute the derivative of the occupancies numbers */
        potential_.U().compute_occupancies_stress_derivatives(*kp, q_op, dn);
        for (int dir1 = 0; dir1 < 3; dir1++) {
            for (int dir2 = 0; dir2 < 3; dir2++) {
                for (int at_lvl = 0; at_lvl < static_cast<int>(potential_.hubbard_potential().local().size());
                     at_lvl++) {
                    const int ia1    = potential_.hubbard_potential().atomic_orbitals(at_lvl).first;
                    const auto& atom = ctx_.unit_cell().atom(ia1);
                    const int lo     = potential_.hubbard_potential().atomic_orbitals(at_lvl).second;
                    if (atom.type().lo_descriptor_hub(lo).use_for_calculation()) {
                        const int lmax_at = 2 * atom.type().lo_descriptor_hub(lo).l() + 1;
                        const int offset  = potential_.hubbard_potential().offset(at_lvl);
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    stress_hubbard_(dir1, dir2) -=
                                            (potential_.hubbard_potential().local(at_lvl)(m2, m1, ispn) *
                                             dn(offset + m1, offset + m2, ispn, dir1 + 3 * dir2))
                                                    .real() /
                                            ctx_.unit_cell().omega();
                                }
                            }
                        }
                    }
                }

                double d = 0;
                for (int i = 0; i < ctx_.cfg().hubbard().nonlocal().size(); i++) {
                    auto nl = ctx_.cfg().hubbard().nonlocal(i);
                    int ia  = nl.atom_pair()[0];
                    int ja  = nl.atom_pair()[1];
                    int il  = nl.l()[0];
                    int jl  = nl.l()[1];
                    int in  = nl.n()[0];
                    int jn  = nl.n()[1];
                    auto Tr = nl.T();

                    auto z1           = std::exp(std::complex<double>(0, -twopi * dot(r3::vector<int>(Tr), kp->vk())));
                    const int at_lvl1 = potential_.hubbard_potential().find_orbital_index(ia, in, il);
                    const int at_lvl2 = potential_.hubbard_potential().find_orbital_index(ja, jn, jl);
                    const int offset1 = potential_.hubbard_potential().offset(at_lvl1);
                    const int offset2 = potential_.hubbard_potential().offset(at_lvl2);

                    for (int is = 0; is < ctx_.num_spins(); is++) {
                        for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
                            for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                                auto result1_ = z1 * std::conj(dn(offset2 + m2, offset1 + m1, is, dir1 + 3 * dir2)) *
                                                potential_.hubbard_potential().nonlocal(i)(m1, m2, is);
                                d += std::real(result1_);
                            }
                        }
                    }
                }
                stress_hubbard_(dir1, dir2) -= d / ctx_.unit_cell().omega();
            }
        }
    }

    /* global reduction */
    kset_.comm().allreduce(&stress_hubbard_(0, 0), 9);
    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_hubbard_);

    stress_hubbard_ = -1.0 * stress_hubbard_;

    return stress_hubbard_;
}

r3::matrix<double>
Stress::calc_stress_core()
{
    stress_core_.zero();

    bool empty = true;

    /* check if the core atomic wave functions are set up or not */
    /* if not then the core correction is simply ignored */

    for (int ia = 0; (ia < ctx_.unit_cell().num_atoms()) && empty; ia++) {
        Atom& atom = ctx_.unit_cell().atom(ia);
        if (!atom.type().ps_core_charge_density().empty()) {
            empty = false;
        }
    }

    if (empty) {
        return stress_core_;
    }

    potential_.xc_potential().rg().fft_transform(-1);

    auto q        = ctx_.gvec().shells_len();
    auto const ff = ctx_.ri().ps_core_djl_->values(q, ctx_.comm());
    auto drhoc    = make_periodic_function<true>(ctx_.unit_cell(), ctx_.gvec(), ctx_.phase_factors_t(), ff);

    double sdiag{0};
    int ig0 = ctx_.gvec().skip_g0();

    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        auto G = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));
        auto g = G.length();

        for (int mu : {0, 1, 2}) {
            for (int nu : {0, 1, 2}) {
                stress_core_(mu, nu) -=
                        std::real(std::conj(potential_.xc_potential().rg().f_pw_local(igloc)) * drhoc[igloc]) * G[mu] *
                        G[nu] / g;
            }
        }

        sdiag += std::real(std::conj(potential_.xc_potential().rg().f_pw_local(igloc)) *
                           density_.rho_pseudo_core().f_pw_local(igloc));
    }

    if (ctx_.gvec().reduced()) {
        stress_core_ *= 2;
        sdiag *= 2;
    }
    if (ctx_.comm().rank() == 0) {
        sdiag += std::real(std::conj(potential_.xc_potential().rg().f_pw_local(0)) *
                           density_.rho_pseudo_core().f_pw_local(0));
    }

    for (int mu : {0, 1, 2}) {
        stress_core_(mu, mu) -= sdiag;
    }

    ctx_.comm().allreduce(&stress_core_(0, 0), 9);

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_core_);

    return stress_core_;
}

r3::matrix<double>
Stress::calc_stress_xc()
{
    stress_xc_.zero();

    double e = sirius::energy_exc(density_, potential_) - sirius::energy_vxc(density_, potential_) -
               sirius::energy_bxc(density_, potential_);

    for (int l = 0; l < 3; l++) {
        stress_xc_(l, l) = e / ctx_.unit_cell().omega();
    }

    if (potential_.is_gradient_correction()) {

        r3::matrix<double> t;

        /* factor 2 in the expression for gradient correction comes from the
           derivative of sigm (which is grad(rho) * grad(rho)) */

        if (ctx_.num_spins() == 1) {
            Smooth_periodic_function<double> rhovc(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            rhovc.zero();
            rhovc += density_.rho().rg();
            rhovc += density_.rho_pseudo_core();

            /* transform to PW domain */
            rhovc.fft_transform(-1);

            /* generate pw coeffs of the gradient */
            auto grad_rho = gradient(rhovc);

            /* gradient in real space */
            for (int x : {0, 1, 2}) {
                grad_rho[x].fft_transform(1);
            }

            for (int irloc = 0; irloc < ctx_.spfft<double>().local_slice_size(); irloc++) {
                for (int mu = 0; mu < 3; mu++) {
                    for (int nu = 0; nu < 3; nu++) {
                        t(mu, nu) += 2 * grad_rho[mu].value(irloc) * grad_rho[nu].value(irloc) *
                                     potential_.vsigma(0).value(irloc);
                    }
                }
            }
        } else {
            auto result  = get_rho_up_dn<true>(density_);
            auto& rho_up = *result[0];
            auto& rho_dn = *result[1];

            /* transform to PW domain */
            rho_up.fft_transform(-1);
            rho_dn.fft_transform(-1);

            /* generate pw coeffs of the gradient */
            auto grad_rho_up = gradient(rho_up);
            auto grad_rho_dn = gradient(rho_dn);

            /* gradient in real space */
            for (int x : {0, 1, 2}) {
                grad_rho_up[x].fft_transform(1);
                grad_rho_dn[x].fft_transform(1);
            }

            for (int irloc = 0; irloc < ctx_.spfft<double>().local_slice_size(); irloc++) {
                for (int mu = 0; mu < 3; mu++) {
                    for (int nu = 0; nu < 3; nu++) {
                        t(mu, nu) += grad_rho_up[mu].value(irloc) * grad_rho_up[nu].value(irloc) * 2 *
                                             potential_.vsigma(0).value(irloc) +
                                     (grad_rho_up[mu].value(irloc) * grad_rho_dn[nu].value(irloc) +
                                      grad_rho_dn[mu].value(irloc) * grad_rho_up[nu].value(irloc)) *
                                             potential_.vsigma(1).value(irloc) +
                                     grad_rho_dn[mu].value(irloc) * grad_rho_dn[nu].value(irloc) * 2 *
                                             potential_.vsigma(2).value(irloc);
                    }
                }
            }
        }
        mpi::Communicator(ctx_.spfft<double>().communicator()).allreduce(&t(0, 0), 9);
        t *= (-1.0 / ctx_.fft_grid().num_points());
        stress_xc_ += t;
    }

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_xc_);

    return stress_xc_;
}

r3::matrix<double>
Stress::calc_stress_us()
{
    PROFILE("sirius::Stress|us");

    stress_us_.zero();

    /* check if we have beta projectors. Only for pseudo potentials */
    if (ctx_.unit_cell().max_mt_basis_size() == 0) {
        return stress_us_;
    }

    potential_.fft_transform(-1);

    la::lib_t la{la::lib_t::none};
    memory_t qmem{memory_t::none};

    memory_pool* mp{nullptr};
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            mp   = &get_memory_pool(memory_t::host);
            la   = la::lib_t::blas;
            qmem = memory_t::host;
            break;
        }
        case device_t::GPU: {
            mp   = &get_memory_pool(memory_t::host_pinned);
            la   = la::lib_t::spla;
            qmem = memory_t::host;
            break;
        }
    }

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
        auto& atom_type = ctx_.unit_cell().atom_type(iat);
        if (!atom_type.augment() || atom_type.num_atoms() == 0) {
            continue;
        }

        Augmentation_operator q_deriv(ctx_.unit_cell().atom_type(iat), ctx_.gvec(), *ctx_.ri().aug_,
                                      *ctx_.ri().aug_djl_);

        auto nbf = atom_type.mt_basis_size();

        /* get auxiliary density matrix */
        auto dm = density_.density_matrix_aux(atom_type);

        mdarray<std::complex<double>, 2> phase_factors({atom_type.num_atoms(), ctx_.gvec().count()},
                                                       get_memory_pool(memory_t::host));

        PROFILE_START("sirius::Stress|us|phase_fac");
        #pragma omp parallel for
        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            for (int i = 0; i < atom_type.num_atoms(); i++) {
                phase_factors(i, igloc) = ctx_.gvec_phase_factor(ig, atom_type.atom_id(i));
            }
        }
        PROFILE_STOP("sirius::Stress|us|phase_fac");

        mdarray<double, 2> v_tmp({atom_type.num_atoms(), ctx_.gvec().count() * 2}, *mp);
        mdarray<double, 2> tmp({nbf * (nbf + 1) / 2, atom_type.num_atoms()}, *mp);
        /* over spin components, can be from 1 to 4 */
        for (int nu = 0; nu < 3; nu++) {
            /* generate dQ(G)/dG */
            q_deriv.generate_pw_coeffs_gvec_deriv(nu);
            for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++) {
                for (int mu = 0; mu < 3; mu++) {
                    PROFILE_START("sirius::Stress|us|prepare");
                    int igloc0{0};
                    if (ctx_.comm().rank() == 0) {
                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            v_tmp(ia, 0) = v_tmp(ia, 1) = 0;
                        }
                        igloc0 = 1;
                    }
                    #pragma omp parallel for
                    for (int igloc = igloc0; igloc < ctx_.gvec().count(); igloc++) {
                        auto gvc = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));
                        double g = gvc.length();

                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            auto z = phase_factors(ia, igloc) * potential_.component(ispin).rg().f_pw_local(igloc) *
                                     (-gvc[mu] / g);
                            v_tmp(ia, 2 * igloc)     = z.real();
                            v_tmp(ia, 2 * igloc + 1) = z.imag();
                        }
                    }
                    PROFILE_STOP("sirius::Stress|us|prepare");

                    PROFILE_START("sirius::Stress|us|gemm");
                    la::wrap(la).gemm('N', 'T', nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                      &la::constant<double>::one(), q_deriv.q_pw().at(qmem), q_deriv.q_pw().ld(),
                                      v_tmp.at(memory_t::host), v_tmp.ld(), &la::constant<double>::zero(),
                                      tmp.at(memory_t::host), tmp.ld());
                    PROFILE_STOP("sirius::Stress|us|gemm");

                    for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                        for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                            stress_us_(mu, nu) += tmp(i, ia) * dm(i, ia, ispin) * q_deriv.sym_weight(i);
                        }
                    }
                }
            }
        }
    }

    ctx_.comm().allreduce(&stress_us_(0, 0), 9);
    if (ctx_.gvec().reduced()) {
        stress_us_ *= 2;
    }

    stress_us_ *= (1.0 / ctx_.unit_cell().omega());

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_us_);

    return stress_us_;
}

r3::matrix<double>
Stress::calc_stress_ewald()
{
    PROFILE("sirius::Stress|ewald");

    stress_ewald_.zero();

    double lambda = ctx_.ewald_lambda();

    auto& uc = ctx_.unit_cell();

    int ig0 = ctx_.gvec().skip_g0();
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        int ig = ctx_.gvec().offset() + igloc;

        auto G          = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));
        double g2       = std::pow(G.length(), 2);
        double g2lambda = g2 / 4.0 / lambda;

        std::complex<double> rho(0, 0);

        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            rho += ctx_.gvec_phase_factor(ig, ia) * static_cast<double>(uc.atom(ia).zn());
        }

        double a1 = twopi * std::pow(std::abs(rho) / uc.omega(), 2) * std::exp(-g2lambda) / g2;

        for (int mu : {0, 1, 2}) {
            for (int nu : {0, 1, 2}) {
                stress_ewald_(mu, nu) += a1 * G[mu] * G[nu] * 2 * (g2lambda + 1) / g2;
            }
        }

        for (int mu : {0, 1, 2}) {
            stress_ewald_(mu, mu) -= a1;
        }
    }

    if (ctx_.gvec().reduced()) {
        stress_ewald_ *= 2;
    }

    ctx_.comm().allreduce(&stress_ewald_(0, 0), 9);

    for (int mu : {0, 1, 2}) {
        stress_ewald_(mu, mu) += twopi * std::pow(uc.num_electrons() / uc.omega(), 2) / 4 / lambda;
    }

    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        for (int i = 1; i < uc.num_nearest_neighbours(ia); i++) {
            auto ja = uc.nearest_neighbour(i, ia).atom_id;
            auto d  = uc.nearest_neighbour(i, ia).distance;
            auto rc = uc.nearest_neighbour(i, ia).rc;

            double a1 = (0.5 * uc.atom(ia).zn() * uc.atom(ja).zn() / uc.omega() / std::pow(d, 3)) *
                        (-2 * std::exp(-lambda * std::pow(d, 2)) * std::sqrt(lambda / pi) * d -
                         std::erfc(std::sqrt(lambda) * d));

            for (int mu : {0, 1, 2}) {
                for (int nu : {0, 1, 2}) {
                    stress_ewald_(mu, nu) += a1 * rc[mu] * rc[nu];
                }
            }
        }
    }

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_ewald_);

    return stress_ewald_;
}

void
Stress::print_info(std::ostream& out__, int verbosity__) const
{
    auto print_stress = [&](std::string label__, r3::matrix<double> const& s) {
        out__ << "=== " << label__ << " ===" << std::endl;
        for (int mu : {0, 1, 2}) {
            out__ << ffmt(12, 6) << s(mu, 0) << ffmt(12, 6) << s(mu, 1) << ffmt(12, 6) << s(mu, 2) << std::endl;
        }
    };

    const double au2kbar = 2.94210119E5;
    auto stress_kin      = stress_kin_ * au2kbar;
    auto stress_har      = stress_har_ * au2kbar;
    auto stress_ewald    = stress_ewald_ * au2kbar;
    auto stress_vloc     = stress_vloc_ * au2kbar;
    auto stress_xc       = stress_xc_ * au2kbar;
    auto stress_nonloc   = stress_nonloc_ * au2kbar;
    auto stress_us       = stress_us_ * au2kbar;
    auto stress_hubbard  = stress_hubbard_ * au2kbar;
    auto stress_core     = stress_core_ * au2kbar;

    out__ << "=== stress tensor components [kbar] ===" << std::endl;

    print_stress("stress_kin", stress_kin);

    print_stress("stress_har", stress_har);

    print_stress("stress_ewald", stress_ewald);

    print_stress("stress_vloc", stress_vloc);

    print_stress("stress_xc", stress_xc);

    print_stress("stress_core", stress_core);

    print_stress("stress_nonloc", stress_nonloc);

    print_stress("stress_us", stress_us);

    stress_us = stress_us + stress_nonloc;
    print_stress("stress_us_nl", stress_us);

    if (ctx_.hubbard_correction()) {
        print_stress("stress_hubbard", stress_hubbard);
    }

    auto stress_total = stress_total_ * au2kbar;
    print_stress("stress_total", stress_total);
}

r3::matrix<double>
Stress::calc_stress_har()
{
    PROFILE("sirius::Stress|har");

    stress_har_.zero();

    int ig0 = ctx_.gvec().skip_g0();
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        auto G    = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));
        double g2 = std::pow(G.length(), 2);
        auto z    = density_.rho().rg().f_pw_local(igloc);
        double d  = twopi * (std::pow(z.real(), 2) + std::pow(z.imag(), 2)) / g2;

        for (int mu : {0, 1, 2}) {
            for (int nu : {0, 1, 2}) {
                stress_har_(mu, nu) += d * 2 * G[mu] * G[nu] / g2;
            }
        }
        for (int mu : {0, 1, 2}) {
            stress_har_(mu, mu) -= d;
        }
    }

    if (ctx_.gvec().reduced()) {
        stress_har_ *= 2;
    }

    ctx_.comm().allreduce(&stress_har_(0, 0), 9);

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_har_);

    return stress_har_;
}

template <typename T>
void
Stress::calc_stress_kin_aux()
{
    stress_kin_.zero();

    for (auto it : kset_.spl_num_kpoints()) {
        auto kp = kset_.get<T>(it.i);

        double fact = kp->gkvec().reduced() ? 2.0 : 1.0;
        fact *= kp->weight();

        #pragma omp parallel
        {
            r3::matrix<double> tmp;
            for (int ispin = 0; ispin < ctx_.num_spins(); ispin++) {
                #pragma omp for
                for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                    for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
                        auto Gk = kp->gkvec().gkvec_cart(gvec_index_t::local(igloc));

                        double f = kp->band_occupancy(i, ispin);
                        auto z = kp->spinor_wave_functions().pw_coeffs(igloc, wf::spin_index(ispin), wf::band_index(i));
                        double d = fact * f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                        for (int mu : {0, 1, 2}) {
                            for (int nu : {0, 1, 2}) {
                                tmp(mu, nu) += Gk[mu] * Gk[nu] * d;
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            stress_kin_ += tmp;
        }
    } // ikloc

    ctx_.comm().allreduce(&stress_kin_(0, 0), 9);

    stress_kin_ *= (-1.0 / ctx_.unit_cell().omega());

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_kin_);
}

r3::matrix<double>
Stress::calc_stress_kin()
{
    PROFILE("sirius::Stress|kin");
    if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(SIRIUS_USE_FP32)
        this->calc_stress_kin_aux<float>();
#endif
    } else {
        this->calc_stress_kin_aux<double>();
    }
    return stress_kin_;
}

r3::matrix<double>
Stress::calc_stress_vloc()
{
    PROFILE("sirius::Stress|vloc");

    stress_vloc_.zero();

    auto q                = ctx_.gvec().shells_len();
    auto const ri_vloc    = ctx_.ri().vloc_->values(q, ctx_.comm());
    auto const ri_vloc_dg = ctx_.ri().vloc_djl_->values(q, ctx_.comm());

    auto v  = make_periodic_function<true>(ctx_.unit_cell(), ctx_.gvec(), ctx_.phase_factors_t(), ri_vloc);
    auto dv = make_periodic_function<true>(ctx_.unit_cell(), ctx_.gvec(), ctx_.phase_factors_t(), ri_vloc_dg);

    double sdiag{0};

    int ig0 = ctx_.gvec().skip_g0();
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {

        auto G = ctx_.gvec().gvec_cart(gvec_index_t::local(igloc));

        for (int mu : {0, 1, 2}) {
            for (int nu : {0, 1, 2}) {
                stress_vloc_(mu, nu) +=
                        std::real(std::conj(density_.rho().rg().f_pw_local(igloc)) * dv[igloc]) * G[mu] * G[nu];
            }
        }

        sdiag += std::real(std::conj(density_.rho().rg().f_pw_local(igloc)) * v[igloc]);
    }

    if (ctx_.gvec().reduced()) {
        stress_vloc_ *= 2;
        sdiag *= 2;
    }
    if (ctx_.comm().rank() == 0) {
        sdiag += std::real(std::conj(density_.rho().rg().f_pw_local(0)) * v[0]);
    }

    for (int mu : {0, 1, 2}) {
        stress_vloc_(mu, mu) -= sdiag;
    }

    ctx_.comm().allreduce(&stress_vloc_(0, 0), 9);

    symmetrize_stress_tensor(ctx_.unit_cell().symmetry(), stress_vloc_);

    return stress_vloc_;
}

r3::matrix<double>
Stress::calc_stress_nonloc()
{
    if (ctx_.cfg().parameters().precision_wf() == "fp32") {
#if defined(SIRIUS_USE_FP32)
        if (ctx_.gamma_point()) {
            calc_stress_nonloc_aux<float, float>();
        } else {
            calc_stress_nonloc_aux<float, std::complex<float>>();
        }
#else
        RTE_THROW("Not compiled with FP32 support");
#endif
    } else {
        if (ctx_.gamma_point()) {
            calc_stress_nonloc_aux<double, double>();
        } else {
            calc_stress_nonloc_aux<double, std::complex<double>>();
        }
    }

    return stress_nonloc_;
}

} // namespace sirius

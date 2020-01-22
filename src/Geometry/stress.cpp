// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file stress.cpp
 *
 *  \brief Contains implementation of sirius::Stress class.
 */

#include "SDDK/geometry3d.hpp"
#include "K_point/k_point.hpp"
#include "stress.hpp"
#include "non_local_functor.hpp"
#include "utils/profiler.hpp"

namespace sirius {

using namespace geometry3d;

template <typename T>
void Stress::calc_stress_nonloc_aux()
{
    PROFILE("sirius::Stress|nonloc");

    mdarray<double, 2> collect_result(9, ctx_.unit_cell().num_atoms());
    collect_result.zero();

    stress_nonloc_.zero();

    /* if there are no beta projectors then get out there */
    if (ctx_.unit_cell().mt_lo_basis_size() == 0) {
        return;
    }

    ctx_.print_memory_usage(__FILE__, __LINE__);

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];
        if (is_device_memory(ctx_.preferred_memory_t())) {
            int nbnd = ctx_.num_bands();
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* allocate GPU memory */
                kp->spinor_wave_functions().pw_coeffs(ispn).allocate(ctx_.mem_pool(memory_t::device));
                kp->spinor_wave_functions().pw_coeffs(ispn).copy_to(memory_t::device, 0, nbnd);
            }
        }
        Beta_projectors_strain_deriv bp_strain_deriv(ctx_, kp->gkvec(), kp->igk_loc());

        Non_local_functor<T> nlf(ctx_, bp_strain_deriv);

        nlf.add_k_point_contribution(*kp, collect_result);

        if (is_device_memory(ctx_.preferred_memory_t())) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* deallocate GPU memory */
                kp->spinor_wave_functions().pw_coeffs(ispn).deallocate(memory_t::device);
            }
        }
    }

    #pragma omp parallel
    {
        matrix3d<double> tmp_stress; // TODO: test pragma omp paralell for reduction(+:stress)

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

    symmetrize(stress_nonloc_);
}

template void Stress::calc_stress_nonloc_aux<double>();

template void Stress::calc_stress_nonloc_aux<double_complex>();

matrix3d<double> Stress::calc_stress_total()
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

matrix3d<double> Stress::calc_stress_hubbard()
{
    stress_hubbard_.zero();

    /* if there are no beta projectors then get out there */
    /* TODO : Need to fix the case where pp have no beta projectors */
    if (ctx_.unit_cell().mt_lo_basis_size() == 0) {
        TERMINATE("Hubbard forces : Your pseudo potentials do not have beta projectors. This need a proper fix");
        return stress_hubbard_;
    }

    mdarray<double_complex, 5> dn(potential_.U().max_number_of_orbitals_per_atom(),
                                  potential_.U().max_number_of_orbitals_per_atom(),
                                  2, ctx_.unit_cell().num_atoms(), 9);

    Q_operator q_op(ctx_);

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        dn.zero();
        int ik    = kset_.spl_num_kpoints(ikloc);
        auto kp__ = kset_[ik];
        if (ctx_.num_mag_dims() == 3)
            TERMINATE("Hubbard stress correction is only implemented for the simple hubbard correction.");

        /* compute the derivative of the occupancies numbers */
        potential_.U().compute_occupancies_stress_derivatives(*kp__, q_op, dn);
        for (int dir1 = 0; dir1 < 3; dir1++) {
            for (int dir2 = 0; dir2 < 3; dir2++) {
                for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ia1++) {
                    const auto& atom = ctx_.unit_cell().atom(ia1);
                    if (atom.type().hubbard_correction()) {
                        const int lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    stress_hubbard_(dir1, dir2) -= (potential_.U().U(m2, m1, ispn, ia1) *
                                                                    dn(m1, m2, ispn, ia1, dir1 + 3 * dir2)).real() /
                                                                    ctx_.unit_cell().omega();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* global reduction */
    kset_.comm().allreduce<double, mpi_op_t::sum>(&stress_hubbard_(0, 0), 9);
    symmetrize(stress_hubbard_);

    return stress_hubbard_;
}

matrix3d<double> Stress::calc_stress_core()
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

    potential_.xc_potential().fft_transform(-1);

    auto& ri_dg = ctx_.ps_core_ri_djl();

    auto drhoc = ctx_.make_periodic_function<index_domain_t::local>(
        [&ri_dg](int iat, double g) { return ri_dg.value<int>(iat, g); });
    double sdiag{0};
    int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;

    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        auto G = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);
        auto g = G.length();

        for (int mu : {0, 1, 2}) {
            for (int nu : {0, 1, 2}) {
                stress_core_(mu, nu) -= std::real(std::conj(potential_.xc_potential().f_pw_local(igloc)) *
                    drhoc[igloc]) * G[mu] * G[nu] / g;
            }
        }

        sdiag += std::real(std::conj(potential_.xc_potential().f_pw_local(igloc)) *
                           density_.rho_pseudo_core().f_pw_local(igloc));
    }

    if (ctx_.gvec().reduced()) {
        stress_core_ *= 2;
        sdiag *= 2;
    }
    if (ctx_.comm().rank() == 0) {
        sdiag += std::real(std::conj(potential_.xc_potential().f_pw_local(0)) *
            density_.rho_pseudo_core().f_pw_local(0));
    }

    for (int mu : {0, 1, 2}) {
        stress_core_(mu, mu) -= sdiag;
    }

    ctx_.comm().allreduce(&stress_core_(0, 0), 9);

    symmetrize(stress_core_);

    return stress_core_;
}

matrix3d<double> Stress::calc_stress_xc()
{
    stress_xc_.zero();

    double e = potential_.energy_exc(density_) - potential_.energy_vxc(density_);

    for (int l = 0; l < 3; l++) {
        stress_xc_(l, l) = e / ctx_.unit_cell().omega();
    }

    if (potential_.is_gradient_correction()) {

        Smooth_periodic_function<double> rhovc(ctx_.spfft(), ctx_.gvec_partition());
        rhovc.zero();
        rhovc.add(density_.rho());
        rhovc.add(density_.rho_pseudo_core());

        /* transform to PW domain */
        rhovc.fft_transform(-1);

        /* generate pw coeffs of the gradient */
        auto grad_rho = gradient(rhovc);

        /* gradient in real space */
        for (int x : {0, 1, 2}) {
            grad_rho[x].fft_transform(1);
        }

        matrix3d<double> t;
        for (int irloc = 0; irloc < ctx_.spfft().local_slice_size(); irloc++) {
            for (int mu = 0; mu < 3; mu++) {
                for (int nu = 0; nu < 3; nu++) {
                    t(mu, nu) += grad_rho[mu].f_rg(irloc) * grad_rho[nu].f_rg(irloc) * potential_.vsigma(0).f_rg(irloc);
                }
            }
        }
        Communicator(ctx_.spfft().communicator()).allreduce(&t(0, 0), 9);
        t *= (-2.0 / ctx_.fft_grid().num_points()); // factor 2 comes from the derivative of sigma (which is grad(rho) * grad(rho))
        // with respect to grad(rho) components
        stress_xc_ += t;
    }

    symmetrize(stress_xc_);

    return stress_xc_;
}

matrix3d<double> Stress::calc_stress_us()
{
    PROFILE("sirius::Stress|us");

    stress_us_.zero();

    /* check if we have beta projectors. Only for pseudo potentials */
    if (ctx_.unit_cell().mt_lo_basis_size() == 0) {
        return stress_us_;
    }

    auto& ri    = ctx_.aug_ri();
    auto& ri_dq = ctx_.aug_ri_djl();

    potential_.fft_transform(-1);

    Augmentation_operator_gvec_deriv q_deriv(ctx_.unit_cell().lmax(), ctx_.gvec(), ctx_.comm());

    linalg_t la{linalg_t::none};

    memory_pool* mp{nullptr};
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            mp = &ctx_.mem_pool(memory_t::host);
            la = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            mp = &ctx_.mem_pool(memory_t::host_pinned);
            la = linalg_t::cublasxt;
            break;
        }
    }

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
        auto& atom_type = ctx_.unit_cell().atom_type(iat);
        if (!atom_type.augment() || atom_type.num_atoms() == 0) {
            continue;
        }

        int nbf = atom_type.mt_basis_size();

        /* get auxiliary density matrix */
        auto dm = density_.density_matrix_aux(iat);

        mdarray<double_complex, 2> phase_factors(atom_type.num_atoms(), ctx_.gvec().count(),
                                                 ctx_.mem_pool(memory_t::host));

        PROFILE_START("sirius::Stress|us|phase_fac");
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            for (int i = 0; i < atom_type.num_atoms(); i++) {
                phase_factors(i, igloc) = ctx_.gvec_phase_factor(ig, atom_type.atom_id(i));
            }
        }
        PROFILE_STOP("sirius::Stress|us|phase_fac");

        mdarray<double, 2> v_tmp(atom_type.num_atoms(), ctx_.gvec().count() * 2, *mp);
        mdarray<double, 2> tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms(), *mp);
        /* over spin components, can be from 1 to 4 */
        for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++) {
            for (int nu = 0; nu < 3; nu++) {
                q_deriv.generate_pw_coeffs(atom_type, ri, ri_dq, nu, *mp);

                for (int mu = 0; mu < 3; mu++) {
                    PROFILE_START("sirius::Stress|us|prepare");
                    int igloc0{0};
                    if (ctx_.comm().rank() == 0) {
                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            v_tmp(ia, 0) = v_tmp(ia, 1) = 0;
                        }
                        igloc0 = 1;
                    }
                    #pragma omp parallel for schedule(static)
                    for (int igloc = igloc0; igloc < ctx_.gvec().count(); igloc++) {
                        auto gvc = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);
                        double g = gvc.length();

                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            auto z = phase_factors(ia, igloc) * potential_.component(ispin).f_pw_local(igloc) *
                                     (-gvc[mu] / g);
                            v_tmp(ia, 2 * igloc)     = z.real();
                            v_tmp(ia, 2 * igloc + 1) = z.imag();
                        }
                    }
                    PROFILE_STOP("sirius::Stress|us|prepare");

                    PROFILE_START("sirius::Stress|us|gemm");
                    linalg(la).gemm('N', 'T', nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                        &linalg_const<double>::one(),
                        q_deriv.q_pw().at(memory_t::host), q_deriv.q_pw().ld(),
                        v_tmp.at(memory_t::host), v_tmp.ld(),
                        &linalg_const<double>::zero(),
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

    symmetrize(stress_us_);

    return stress_us_;
}

matrix3d<double> Stress::calc_stress_ewald()
{
    PROFILE("sirius::Stress|ewald");

    stress_ewald_.zero();

    double lambda = ctx_.ewald_lambda();

    auto& uc = ctx_.unit_cell();

    int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        int ig = ctx_.gvec().offset() + igloc;

        auto G          = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);
        double g2       = std::pow(G.length(), 2);
        double g2lambda = g2 / 4.0 / lambda;

        double_complex rho(0, 0);

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
            int ja   = uc.nearest_neighbour(i, ia).atom_id;
            double d = uc.nearest_neighbour(i, ia).distance;

            vector3d<double> v1 = uc.atom(ja).position() - uc.atom(ia).position() +
                                  vector3d<int>(uc.nearest_neighbour(i, ia).translation);
            auto r1    = uc.get_cartesian_coordinates(v1);
            double len = r1.length();

            if (std::abs(d - len) > 1e-12) {
                STOP();
            }

            double a1 = (0.5 * uc.atom(ia).zn() * uc.atom(ja).zn() / uc.omega() / std::pow(len, 3)) *
                        (-2 * std::exp(-lambda * std::pow(len, 2)) * std::sqrt(lambda / pi) * len -
                         std::erfc(std::sqrt(lambda) * len));

            for (int mu : {0, 1, 2}) {
                for (int nu : {0, 1, 2}) {
                    stress_ewald_(mu, nu) += a1 * r1[mu] * r1[nu];
                }
            }
        }
    }

    symmetrize(stress_ewald_);

    return stress_ewald_;
}

void Stress::print_info() const
{
    if (ctx_.comm().rank() == 0) {

        auto print_stress = [&](matrix3d<double> const& s) {
            for (int mu : {0, 1, 2}) {
                std::printf("%12.6f %12.6f %12.6f\n", s(mu, 0), s(mu, 1), s(mu, 2));
            }
        };

        const double au2kbar = 2.94210119E5;
        auto stress_kin      = stress_kin_ * au2kbar;
        auto stress_har      = stress_har_ * au2kbar;
        auto stress_ewald    = stress_ewald_ * au2kbar;
        auto stress_vloc     = stress_vloc_ * au2kbar;
        auto stress_nonloc   = stress_nonloc_ * au2kbar;
        auto stress_us       = stress_us_ * au2kbar;
        auto stress_hubbard  = stress_hubbard_ * au2kbar;

        std::printf("== stress tensor components [kbar] ===\n");

        std::printf("== stress_kin ==\n");
        print_stress(stress_kin);

        std::printf("== stress_har ==\n");
        print_stress(stress_har);

        std::printf("== stress_ewald ==\n");
        print_stress(stress_ewald);

        std::printf("== stress_vloc ==\n");
        print_stress(stress_vloc);

        std::printf("== stress_nonloc ==\n");
        print_stress(stress_nonloc);

        std::printf("== stress_us ==\n");
        print_stress(stress_us);

        stress_us = stress_us + stress_nonloc;
        std::printf("== stress_us_nl ==\n");
        print_stress(stress_us);

        if (ctx_.hubbard_correction()) {
            std::printf("== stress_hubbard ==\n");
            print_stress(stress_hubbard);
        }

        auto stress_total = stress_total_ * au2kbar;
        std::printf("== stress_total ==\n");
        print_stress(stress_total);
    }
}

matrix3d<double> Stress::calc_stress_har()
{
    PROFILE("sirius::Stress|har");

    stress_har_.zero();

    int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
        auto G    = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);
        double g2 = std::pow(G.length(), 2);
        auto z    = density_.rho().f_pw_local(igloc);
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

    symmetrize(stress_har_);

    return stress_har_;
}

matrix3d<double> Stress::calc_stress_kin()
{
    PROFILE("sirius::Stress|kin");

    stress_kin_.zero();

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        #pragma omp parallel
        {
            matrix3d<double> tmp;
            #pragma omp for schedule(static)
            for (int igloc = 0; igloc < kp->num_gkvec_loc(); igloc++) {
                auto Gk = kp->gkvec().gkvec_cart<index_domain_t::local>(igloc);

                double d{0};
                for (int ispin = 0; ispin < ctx_.num_spins(); ispin++) {
                    for (int i = 0; i < kp->num_occupied_bands(ispin); i++) {
                        double f = kp->band_occupancy(i, ispin);
                        auto z   = kp->spinor_wave_functions().pw_coeffs(ispin).prime(igloc, i);
                        d += f * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
                d *= kp->weight();
                if (kp->gkvec().reduced()) {
                    d *= 2;
                }
                for (int mu : {0, 1, 2}) {
                    for (int nu : {0, 1, 2}) {
                        tmp(mu, nu) += Gk[mu] * Gk[nu] * d;
                    }
                }
            } // igloc
            #pragma omp critical
            stress_kin_ += tmp;
        }
    } // ikloc

    ctx_.comm().allreduce(&stress_kin_(0, 0), 9);

    stress_kin_ *= (-1.0 / ctx_.unit_cell().omega());

    symmetrize(stress_kin_);

    return stress_kin_;
}

void Stress::symmetrize(matrix3d<double>& mtrx__) const
{
    if (!ctx_.use_symmetry()) {
        return;
    }

    matrix3d<double> result;

    for (int i = 0; i < ctx_.unit_cell().symmetry().num_mag_sym(); i++) {
        auto R = ctx_.unit_cell().symmetry().magnetic_group_symmetry(i).spg_op.rotation;
        result = result + transpose(R) * mtrx__ * R;
    }

    mtrx__ = result * (1.0 / ctx_.unit_cell().symmetry().num_mag_sym());

    std::vector<std::array<int, 2>> idx = {{0, 1}, {0, 2}, {1, 2}};
    for (auto e : idx) {
        mtrx__(e[0], e[1]) = mtrx__(e[1], e[0]) = 0.5 * (mtrx__(e[0], e[1]) + mtrx__(e[1], e[0]));
    }
}

matrix3d<double> Stress::calc_stress_vloc()
{
    PROFILE("sirius::Stress|vloc");

    stress_vloc_.zero();

    auto& ri_vloc    = ctx_.vloc_ri();
    auto& ri_vloc_dg = ctx_.vloc_ri_djl();

    auto v =
        ctx_.make_periodic_function<index_domain_t::local>([&](int iat, double g) { return ri_vloc.value(iat, g); });

    auto dv =
        ctx_.make_periodic_function<index_domain_t::local>([&](int iat, double g) { return ri_vloc_dg.value(iat, g); });

    double sdiag{0};

    int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;
    for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {

        auto G = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc);

        for (int mu : {0, 1, 2}) {
            for (int nu : {0, 1, 2}) {
                stress_vloc_(mu, nu) +=
                    std::real(std::conj(density_.rho().f_pw_local(igloc)) * dv[igloc]) * G[mu] * G[nu];
            }
        }

        sdiag += std::real(std::conj(density_.rho().f_pw_local(igloc)) * v[igloc]);
    }

    if (ctx_.gvec().reduced()) {
        stress_vloc_ *= 2;
        sdiag *= 2;
    }
    if (ctx_.comm().rank() == 0) {
        sdiag += std::real(std::conj(density_.rho().f_pw_local(0)) * v[0]);
    }

    for (int mu : {0, 1, 2}) {
        stress_vloc_(mu, mu) -= sdiag;
    }

    ctx_.comm().allreduce(&stress_vloc_(0, 0), 9);

    symmetrize(stress_vloc_);

    return stress_vloc_;
}

matrix3d<double> Stress::calc_stress_nonloc()
{
    if (ctx_.gamma_point()) {
        calc_stress_nonloc_aux<double>();
    } else {
        calc_stress_nonloc_aux<double_complex>();
    }

    return stress_nonloc_;
}

} // namespace sirius

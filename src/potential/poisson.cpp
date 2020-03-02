// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file poisson.hpp
 *
 *  \brief Implementation of the Poisson solver for the full-potential muffin-tin case.
 */

#include "potential.hpp"
#include "utils/profiler.hpp"

namespace sirius {

void Potential::poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt__,
                                      mdarray<double_complex, 2>& qit__,
                                      double_complex* rho_pw__)
{
    PROFILE("sirius::Potential::poisson_add_pseudo_pw");

    int lmmax = ctx_.lmmax_rho();
    int ngv = ctx_.gvec().count();

    /* The following term is added to the plane-wave coefficients of the charge density:
     * Integrate[SphericalBesselJ[l,a*x]*p[x,R]*x^2,{x,0,R},Assumptions->{l>=0,n>=0,R>0,a>0}] /
     *  Integrate[p[x,R]*x^(2+l),{x,0,R},Assumptions->{h>=0,n>=0,R>0}]
     * i.e. contributon from pseudodensity to l-th channel of plane wave expansion multiplied by
     * the difference bethween true and interstitial-in-the-mt multipole moments and divided by the
     * moment of the pseudodensity.
     */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        double R = unit_cell_.atom_type(iat).mt_radius();
        int na = unit_cell_.atom_type(iat).num_atoms();

        mdarray<double_complex, 2> pf;
        mdarray<double_complex, 2> qa;
        mdarray<double_complex, 2> qapf;

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                auto& mp = ctx_.mem_pool(memory_t::host);
                pf = mdarray<double_complex, 2>(ngv, na, mp);
                qa = mdarray<double_complex, 2>(lmmax, na, mp);
                qapf = mdarray<double_complex, 2>(lmmax, ngv, mp);
                break;
            }
            case device_t::GPU: {
                auto& mp = ctx_.mem_pool(memory_t::host);
                auto& mpd = ctx_.mem_pool(memory_t::device);
                /* allocate on GPU */
                pf = mdarray<double_complex, 2>(nullptr, ngv, na);
                pf.allocate(mpd);
                /* allocate on CPU & GPU */
                qa = mdarray<double_complex, 2>(lmmax, na, mp);
                qa.allocate(mpd);
                /* allocate on CPU & GPU */
                qapf = mdarray<double_complex, 2>(lmmax, ngv, mp);
                qapf.allocate(mpd);
                break;
            }
        }

        ctx_.generate_phase_factors(iat, pf);

        for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                qa(lm, i) = qmt__(lm, ia) - qit__(lm, ia);
            }
        }

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                linalg(linalg_t::blas).gemm('N', 'C', ctx_.lmmax_rho(), ctx_.gvec().count(),
                    unit_cell_.atom_type(iat).num_atoms(), &linalg_const<double_complex>::one(),
                    qa.at(memory_t::host), qa.ld(), pf.at(memory_t::host), pf.ld(),
                    &linalg_const<double_complex>::zero(), qapf.at(memory_t::host), qapf.ld());
                break;
            }
            case device_t::GPU: {
                qa.copy_to(memory_t::device);
                linalg(linalg_t::gpublas).gemm('N', 'C', ctx_.lmmax_rho(), ctx_.gvec().count(),
                    unit_cell_.atom_type(iat).num_atoms(), &linalg_const<double_complex>::one(),
                    qa.at(memory_t::device), qa.ld(), pf.at(memory_t::device), pf.ld(),
                    &linalg_const<double_complex>::zero(), qapf.at(memory_t::device), qapf.ld());
                qapf.copy_to(memory_t::host);
                break;
            }
        }

        double fourpi_omega = fourpi / unit_cell_.omega();

        /* add pseudo_density to interstitial charge density so that rho(G) has the correct
         * multipole moments in the muffin-tins */
        #pragma omp parallel for schedule(static)
        for (int igloc = ctx_.gvec().skip_g0(); igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;

            double gR = ctx_.gvec().gvec_len(ig) * R;
            double gRn = std::pow(2.0 / gR, pseudo_density_order_ + 1);

            double_complex rho_G(0, 0);
            /* loop over atoms of the same type */
            for (int l = 0, lm = 0; l <= ctx_.lmax_rho(); l++) {
                double_complex zt1(0, 0);
                for (int m = -l; m <= l; m++, lm++) {
                    zt1 += gvec_ylm_(lm, igloc) * qapf(lm, igloc);
                }
                rho_G += fourpi_omega * std::conj(zil_[l]) * zt1 * gamma_factors_R_(l, iat) *
                         sbessel_mt_(l + pseudo_density_order_ + 1, igloc, iat) * gRn;
            }
            rho_pw__[igloc] += rho_G;
        }
        /* for G=0 case */
        if (ctx_.comm().rank() == 0) {
            double_complex rho_G(0, 0);
            for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                rho_G += fourpi_omega * y00 * (qmt__(0, ia) - qit__(0, ia));
            }
            rho_pw__[0] += rho_G;
        }
    }
}

void Potential::poisson(Periodic_function<double> const& rho)
{
    PROFILE("sirius::Potential::poisson");

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (ctx_.full_potential()) {

        /* true multipole moments */
        auto qmt = poisson_vmt(rho);

        if (ctx_.control().print_checksum_) {
            if (ctx_.comm().rank() == 0) {
                utils::print_checksum("qmt", qmt.checksum());
            }
        }

        /* compute multipoles of interstitial density in MT region */
        auto qit = ctx_.sum_fg_fl_yg(ctx_.lmax_rho(), &rho.f_pw_local(0), sbessel_mom_, gvec_ylm_);

        if (ctx_.control().print_checksum_) {
            if (ctx_.comm().rank() == 0) {
                utils::print_checksum("qit", qit.checksum());
            }
        }

        /* add contribution from the pseudo-charge */
        poisson_add_pseudo_pw(qmt, qit, const_cast<double_complex*>(&rho.f_pw_local(0)));

        if (ctx_.control().verification_ >= 2) {
            auto qit = ctx_.sum_fg_fl_yg(ctx_.lmax_rho(), &rho.f_pw_local(0), sbessel_mom_, gvec_ylm_);

            double d = 0.0;
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                    d += std::abs(qmt(lm, ia) - qit(lm, ia));
                }
            }
            ctx_.message(1, __function_name__, "pseudocharge error: %18.10f\n", d);
        }
    }

    /* compute pw coefficients of Hartree potential */
    int ig0{0};
    if (ctx_.gvec().comm().rank() == 0) {
        hartree_potential_->f_pw_local(0) = 0.0;
        ig0 = 1;
    }
    if (!ctx_.molecule()) {
        #pragma omp parallel for schedule(static)
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            hartree_potential_->f_pw_local(igloc) = (fourpi * rho.f_pw_local(igloc) / std::pow(ctx_.gvec().gvec_len(ig), 2));
        }
    } else {
        /* reference paper:

           Supercell technique for total-energy calculations of finite charged and polar systems
           M. R. Jarvis, I. D. White, R. W. Godby, and M. C. Payne
           Phys. Rev. B 56, 14972 – Published 15 December 1997
           DOI:https://doi.org/10.1103/PhysRevB.56.14972
        */
        double R_cut = 0.5 * std::pow(unit_cell_.omega(), 1.0 / 3);
        #pragma omp parallel for schedule(static)
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            hartree_potential_->f_pw_local(igloc) = (fourpi * rho.f_pw_local(igloc) / std::pow(ctx_.gvec().gvec_len(ig), 2)) *
                                                    (1.0 - std::cos(ctx_.gvec().gvec_len(ig) * R_cut));
        }
    }

    //if (ctx_.control().print_checksum_) {
    //    auto z4 = mdarray<double_complex, 1>(&vh->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    //    if (ctx_.comm().rank() == 0) {
    //        DUMP("checksum(vh_pw): %20.14f %20.14f", z4.real(), z4.imag());
    //    }
    //}

    /* boundary condition for muffin-tins */
    if (ctx_.full_potential()) {
        /* compute V_lm at the MT boundary */
        auto vmtlm = ctx_.sum_fg_fl_yg(ctx_.lmax_pot(), &hartree_potential_->f_pw_local(0), sbessel_mt_, gvec_ylm_);

        /* add boundary condition and convert to Rlm */
        PROFILE("sirius::Potential::poisson|bc");
        mdarray<double, 3> rRl(unit_cell_.max_num_mt_points(), ctx_.lmax_pot() + 1, unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            int nmtp = unit_cell_.atom_type(iat).num_mt_points();
            double R = unit_cell_.atom_type(iat).mt_radius();

            #pragma omp parallel for default(shared)
            for (int l = 0; l <= ctx_.lmax_pot(); l++) {
                for (int ir = 0; ir < nmtp; ir++) {
                    rRl(ir, l, iat) = std::pow(unit_cell_.atom_type(iat).radial_grid(ir) / R, l);
                }
            }
        }

        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            int nmtp = unit_cell_.atom(ia).num_mt_points();

            std::vector<double> vlm(ctx_.lmmax_pot());
            SHT::convert(ctx_.lmax_pot(), &vmtlm(0, ia), &vlm[0]);

            #pragma omp parallel for default(shared)
            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) {
                    hartree_potential_->f_mt<index_domain_t::local>(lm, ir, ialoc) +=
                        vlm[lm] * rRl(ir, l, unit_cell_.atom(ia).type_id());
                }
            }
            /* save electronic part of the potential at the point of origin */
#ifdef __VHA_AUX
            vh_el_(ia) = y00 * hartree_potential_->f_mt<index_domain_t::local>(0, 0, ialoc) + unit_cell_.atom(ia).zn() / unit_cell_.atom(ia).radial_grid(0);
#else
            vh_el_(ia) = y00 * hartree_potential_->f_mt<index_domain_t::local>(0, 0, ialoc);
#endif
        }
        ctx_.comm().allgather(vh_el_.at(memory_t::host), unit_cell_.spl_num_atoms().global_offset(),
                              unit_cell_.spl_num_atoms().local_size());
    }

    /* transform Hartree potential to real space */
    hartree_potential_->fft_transform(1);

    if (ctx_.control().print_checksum_) {
        auto cs = hartree_potential_->checksum_rg();
        auto cs1 = hartree_potential_->checksum_pw();
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("vha_rg", cs);
            utils::print_checksum("vha_pw", cs1);
        }
    }

    /* compute contribution from the smooth part of Hartree potential */
    energy_vha_ = sirius::inner(rho, hartree_potential());

#ifndef __VHA_AUX
    /* add nucleus potential and contribution to Hartree energy */
    if (ctx_.full_potential()) {
        double evha_nuc{0};
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            auto& atom = unit_cell_.atom(ia);
            Spline<double> srho(atom.radial_grid());
            for (int ir = 0; ir < atom.num_mt_points(); ir++) {
                double r = atom.radial_grid(ir);
                hartree_potential_->f_mt<index_domain_t::local>(0, ir, ialoc) -= atom.zn() / r / y00;
                srho(ir) = rho.f_mt<index_domain_t::local>(0, ir, ialoc) * r;
            }
            evha_nuc -= atom.zn() * srho.interpolate().integrate(0) / y00;
        }
        ctx_.comm().allreduce(&evha_nuc, 1);
        energy_vha_ += evha_nuc;
    }
#endif

    /* check values at MT boundary */
    //if (true) {
    //    /* compute V_lm at the MT boundary */
    //    auto vmtlm = ctx_.sum_fg_fl_yg(ctx_.lmax_pot(), &hartree_potential_->f_pw_local(0), sbessel_mt_, gvec_ylm_);

    //    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
    //        int ia = unit_cell_.spl_num_atoms(ialoc);
    //        int nmtp = unit_cell_.atom(ia).num_mt_points();

    //        std::vector<double> vlm(ctx_.lmmax_pot());
    //        SHT::convert(ctx_.lmax_pot(), &vmtlm(0, ia), &vlm[0]);

    //        for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
    //            printf("ia=%i lm=%i vlmdiff=%20.16f\n", ia, lm,
    //              std::abs(hartree_potential_->f_mt<index_domain_t::local>(lm, nmtp - 1, ialoc) - vlm[lm]));
    //        }
    //    }

    //}
}

} // namespace sirius

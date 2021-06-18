// Copyright (c) 2013-2020 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file potential.cpp
 *
 *  \brief Generate effective potential.
 */

#include "potential.hpp"
#include "xc_functional.hpp"

namespace sirius {

Potential::Potential(Simulation_context& ctx__)
    : Field4D(ctx__, ctx__.lmmax_pot())
    , unit_cell_(ctx__.unit_cell())
    , comm_(ctx__.comm())
{
    PROFILE("sirius::Potential");

    if (!ctx_.initialized()) {
        TERMINATE("Simulation_context is not initialized");
    }

    lmax_ = std::max(ctx_.lmax_rho(), ctx_.lmax_pot());

    if (lmax_ >= 0) {
        sht_  = std::unique_ptr<SHT>(new SHT(ctx_.processing_unit(), lmax_, ctx_.cfg().settings().sht_coverage()));
        if (ctx_.cfg().control().verification() >= 1)  {
            sht_->check();
        }
        l_by_lm_ = utils::l_by_lm(lmax_);

        /* precompute i^l */
        zil_.resize(lmax_ + 1);
        for (int l = 0; l <= lmax_; l++) {
            zil_[l] = std::pow(double_complex(0, 1), l);
        }

        zilm_.resize(utils::lmmax(lmax_));
        for (int l = 0, lm = 0; l <= lmax_; l++) {
            for (int m = -l; m <= l; m++, lm++) {
                zilm_[lm] = zil_[l];
            }
        }
    }

    /* create list of XC functionals */
    for (auto& xc_label : ctx_.xc_functionals()) {
        xc_func_.push_back(new XC_functional(ctx_.spfft(), ctx_.unit_cell().lattice_vectors(), xc_label, ctx_.num_spins()));
        if (ctx_.cfg().parameters().xc_dens_tre() > 0) {
            xc_func_.back()->set_dens_threshold(ctx_.cfg().parameters().xc_dens_tre());
        }
    }

    using pf = Periodic_function<double>;
    using spf = Smooth_periodic_function<double>;

    hartree_potential_ = std::unique_ptr<pf>(new pf(ctx_, ctx_.lmmax_pot()));
    hartree_potential_->allocate_mt(false);

    xc_potential_ = std::unique_ptr<pf>(new pf(ctx_, ctx_.lmmax_pot()));
    xc_potential_->allocate_mt(false);

    xc_energy_density_ = std::unique_ptr<pf>(new pf(ctx_, ctx_.lmmax_pot()));
    xc_energy_density_->allocate_mt(false);

    if (this->is_gradient_correction()) {
        int nsigma = (ctx_.num_spins() == 1) ? 1 : 3;
        for (int i = 0; i < nsigma ; i++) {
            vsigma_[i] = std::unique_ptr<spf>(new spf(ctx_.spfft(), ctx_.gvec_partition()));
        }
    }

    if (!ctx_.full_potential()) {
        local_potential_ = std::unique_ptr<spf>(new spf(ctx_.spfft(), ctx_.gvec_partition()));
        dveff_ = std::unique_ptr<spf>(new spf(ctx_.spfft(), ctx_.gvec_partition()));
        dveff_->zero();
    }

    vh_el_ = mdarray<double, 1>(unit_cell_.num_atoms());

    if (ctx_.full_potential()) {
        gvec_ylm_ = mdarray<double_complex, 2>(ctx_.lmmax_pot(), ctx_.gvec().count(), memory_t::host, "gvec_ylm_");

        switch (ctx_.valence_relativity()) {
            case relativity_t::iora: {
                rm2_inv_pw_ = mdarray<double_complex, 1>(ctx_.gvec().num_gvec());
            }
            case relativity_t::zora: {
                rm_inv_pw_ = mdarray<double_complex, 1>(ctx_.gvec().num_gvec());
            }
            default: {
                veff_pw_ = mdarray<double_complex, 1>(ctx_.gvec().num_gvec());
            }
        }
    }

    aux_bf_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
    aux_bf_.zero();

    if (ctx_.cfg().parameters().reduce_aux_bf() > 0 && ctx_.cfg().parameters().reduce_aux_bf() < 1) {
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                aux_bf_(x, ia) = 1;
            }
        }
    }

    /* in case of PAW */
    init_PAW();

    if (ctx_.hubbard_correction()) {
        U_ = std::unique_ptr<Hubbard>(new Hubbard(ctx_));
    }

    update();
}

Potential::~Potential()
{
    for (auto& ixc: xc_func_) {
        delete ixc;
    }
}

void Potential::update()
{
    PROFILE("sirius::Potential::update");

    if (!ctx_.full_potential()) {
        local_potential_->zero();
        generate_local_potential();
    } else {
        gvec_ylm_ = ctx_.generate_gvec_ylm(ctx_.lmax_pot());
        sbessel_mt_ = ctx_.generate_sbessel_mt(lmax_ + pseudo_density_order_ + 1);

        /* compute moments of spherical Bessel functions
         *
         * In[]:= Integrate[SphericalBesselJ[l,G*x]*x^(2+l),{x,0,R},Assumptions->{R>0,G>0,l>=0}]
         * Out[]= (Sqrt[\[Pi]/2] R^(3/2+l) BesselJ[3/2+l,G R])/G^(3/2)
         *
         * and use relation between Bessel and spherical Bessel functions:
         * Subscript[j, n](z)=Sqrt[\[Pi]/2]/Sqrt[z]Subscript[J, n+1/2](z) */
        sbessel_mom_ = mdarray<double, 3>(ctx_.lmax_rho() + 1,
                                          ctx_.gvec().count(),
                                          unit_cell_.num_atom_types(),
                                          memory_t::host, "sbessel_mom_");
        sbessel_mom_.zero();
        int ig0{0};
        if (ctx_.comm().rank() == 0) {
            /* for |G| = 0 */
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                sbessel_mom_(0, 0, iat) = std::pow(unit_cell_.atom_type(iat).mt_radius(), 3) / 3.0;
            }
            ig0 = 1;
        }
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            #pragma omp parallel for schedule(static)
            for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
                auto len = ctx_.gvec().gvec_cart<index_domain_t::local>(igloc).length();
                for (int l = 0; l <= ctx_.lmax_rho(); l++) {
                    sbessel_mom_(l, igloc, iat) = std::pow(unit_cell_.atom_type(iat).mt_radius(), l + 2) *
                                                  sbessel_mt_(l + 1, igloc, iat) / len;
                }
            }
        }

        /* compute Gamma[5/2 + n + l] / Gamma[3/2 + l] / R^l
         *
         * use Gamma[1/2 + p] = (2p - 1)!!/2^p Sqrt[Pi] */
        gamma_factors_R_ = mdarray<double, 2>(ctx_.lmax_rho() + 1, unit_cell_.num_atom_types(), memory_t::host, "gamma_factors_R_");
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            for (int l = 0; l <= ctx_.lmax_rho(); l++) {
                long double Rl = std::pow(unit_cell_.atom_type(iat).mt_radius(), l);

                int n_min = (2 * l + 3);
                int n_max = (2 * l + 1) + (2 * pseudo_density_order_ + 2);
                /* split factorial product into two parts to avoid overflow */
                long double f1 = 1.0;
                long double f2 = 1.0;
                for (int n = n_min; n <= n_max; n += 2) {
                    if (f1 < Rl) {
                        f1 *= (n / 2.0);
                    } else {
                        f2 *= (n / 2.0);
                    }
                }
                gamma_factors_R_(l, iat) = static_cast<double>((f1 / Rl) * f2);
            }
        }
    }

    // VDWXC depends on unit cell, which might have changed.
    for (auto& xc : xc_func_) {
        if (xc->is_vdw()) {
            xc->vdw_update_unit_cell(ctx_.spfft(), ctx_.unit_cell().lattice_vectors());
        }
    }
}

bool Potential::is_gradient_correction() const
{
    bool is_gga{false};
    for (auto& ixc : xc_func_) {
        if (ixc->is_gga() || ixc->is_vdw()) {
            is_gga = true;
        }
    }
    return is_gga;
}

void Potential::insert_xc_functionals(const std::vector<std::string>& labels__)
{
    /* create list of XC functionals */
    for (auto& xc_label : labels__) {
        xc_func_.push_back(new XC_functional(ctx_.spfft(), ctx_.unit_cell().lattice_vectors(), xc_label,
                    ctx_.num_spins()));
    }
}

void Potential::generate(Density const& density__, bool use_symmetry__, bool transform_to_rg__)
{
    PROFILE("sirius::Potential::generate");

    if (!ctx_.full_potential()) {
        /* save current effective potential */
        for (size_t ig = 0; ig < effective_potential().f_pw_local().size(); ig++) {
            dveff_->f_pw_local(ig) = effective_potential().f_pw_local(ig);
        }
    }

    /* zero effective potential and magnetic field */
    zero();

    auto veff_callback = ctx_.veff_callback();
    if (veff_callback) {
        veff_callback();
        //if (!ctx_.full_potential()) {
        //    /* add local ionic potential to the effective potential */
        //    effective_potential().add(local_potential());
        //}
        /* transform to real space */
        //fft_transform(1);
    } else {
        /* solve Poisson equation */
        poisson(density__.rho());

        /* add Hartree potential to the total potential */
        effective_potential().add(hartree_potential());

        if (ctx_.cfg().control().print_hash()) {
            auto h = effective_potential().hash_f_rg();
            if (ctx_.comm().rank() == 0) {
                utils::print_hash("Vha", h);
            }
        }

        if (ctx_.full_potential()) {
            xc(density__);
        } else {
            /* add local ionic potential to the effective potential */
            effective_potential().add(local_potential());
            /* construct XC potentials from rho + rho_core */
            xc<true>(density__);
        }
        /* add XC potential to the effective potential */
        effective_potential().add(xc_potential());

        if (ctx_.cfg().control().print_hash()) {
            auto h = effective_potential().hash_f_rg();
            if (ctx_.comm().rank() == 0) {
                utils::print_hash("Vha+Vxc", h);
            }
        }

        if (ctx_.full_potential()) {
            effective_potential().sync_mt();
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field(j).sync_mt();
            }
        }

        /* get plane-wave coefficients of effective potential;
         * they will be used in three places:
         *  1) compute D-matrix
         *  2) establish a mapping between fine and coarse FFT grid for the Hloc operator
         *  3) symmetrize effective potential */
        fft_transform(-1);
    }

    if (use_symmetry__) {
        /* symmetrize potential and effective magnetic field */
        this->symmetrize();
        if (transform_to_rg__) {
            /* transform potential to real space after symmetrization */
            this->fft_transform(1);
        }
    }

    if (!ctx_.full_potential()) {
        /* this is needed later to compute scf correction to forces */
        for (size_t ig = 0; ig < effective_potential().f_pw_local().size(); ig++) {
            dveff_->f_pw_local(ig) = effective_potential().f_pw_local(ig) - dveff_->f_pw_local(ig);
        }
    }

    if (ctx_.cfg().control().print_hash()) {
        auto h = effective_potential().hash_f_pw();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("V(G)", h);
        }
    }

    if (!ctx_.full_potential()) {
        generate_D_operator_matrix();
        generate_PAW_effective_potential(density__);
    }

    if (ctx_.hubbard_correction()) {
        this->U().generate_potential(density__.occupation_matrix());
    }

    if (ctx_.cfg().parameters().reduce_aux_bf() > 0 && ctx_.cfg().parameters().reduce_aux_bf() < 1) {
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            for (int x : {0, 1, 2}) {
                aux_bf_(x, ia) *= ctx_.cfg().parameters().reduce_aux_bf();
            }
        }
    }
}

}


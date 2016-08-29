// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains implementation oof constructor and destructor of sirius::Potential class.
 */

#include "potential.h"

namespace sirius {

Potential::Potential(Simulation_context& ctx__)
    : ctx_(ctx__),
      unit_cell_(ctx__.unit_cell()),
      comm_(ctx__.comm()),
      pseudo_density_order(9),
      mixer_(nullptr)
{
    runtime::Timer t("sirius::Potential::Potential");

    if (ctx_.full_potential() || ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
        lmax_ = std::max(ctx_.lmax_rho(), ctx_.lmax_pot());
        sht_ = std::unique_ptr<SHT>(new SHT(lmax_));
    }

    if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo) {
        l_by_lm_ = Utils::l_by_lm(lmax_);

        /* precompute i^l */
        zil_.resize(lmax_ + 1);
        for (int l = 0; l <= lmax_; l++) {
            zil_[l] = std::pow(double_complex(0, 1), l);
        }
        
        zilm_.resize(Utils::lmmax(lmax_));
        for (int l = 0, lm = 0; l <= lmax_; l++) {
            for (int m = -l; m <= l; m++, lm++) {
                zilm_[lm] = zil_[l];
            }
        }
    }

    effective_potential_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot(), 1);
    
    //int need_gvec = (ctx_.full_potential()) ? 0 : 1;
    int need_gvec{1};
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        effective_magnetic_field_[j] = new Periodic_function<double>(ctx_, ctx_.lmmax_pot(), need_gvec);
    }
    
    hartree_potential_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot(), 1);
    hartree_potential_->allocate_mt(false);
    
    xc_potential_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot(), 0);
    xc_potential_->allocate_mt(false);
    
    xc_energy_density_ = new Periodic_function<double>(ctx_, ctx_.lmmax_pot(), 0);
    xc_energy_density_->allocate_mt(false);

    if (!ctx_.full_potential())
    {
        local_potential_ = new Periodic_function<double>(ctx_, 0, 0);
        local_potential_->zero();

        generate_local_potential();
    }

    vh_el_ = mdarray<double, 1>(unit_cell_.num_atoms());

    init();

    spl_num_gvec_ = splindex<block>(ctx_.gvec().num_gvec(), comm_.size(), comm_.rank());
    
    if (ctx_.full_potential())
    {
        gvec_ylm_ = mdarray<double_complex, 2>(ctx_.lmmax_pot(), spl_num_gvec_.local_size());
        for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
        {
            int ig = spl_num_gvec_[igloc];
            auto rtp = SHT::spherical_coordinates(ctx_.gvec().gvec_cart(ig));
            SHT::spherical_harmonics(ctx_.lmax_pot(), rtp[1], rtp[2], &gvec_ylm_(0, igloc));
        }
    }

    // create list of XC functionals
    for (auto& xc_label: ctx_.xc_functionals())
    {
        xc_func_.push_back(new XC_functional(xc_label, ctx_.num_spins()));
    }

    // if PAW calc
    if(ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential)
    {
        init_PAW();
    }
}

Potential::~Potential()
{
    delete effective_potential_; 
    for (int j = 0; j < ctx_.num_mag_dims(); j++) delete effective_magnetic_field_[j];
    delete hartree_potential_;
    delete xc_potential_;
    delete xc_energy_density_;
    if (!ctx_.full_potential()) delete local_potential_;
    if (mixer_ != nullptr) delete mixer_;
    for (auto& ixc: xc_func_) delete ixc;
}

}

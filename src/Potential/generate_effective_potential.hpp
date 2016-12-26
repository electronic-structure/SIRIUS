// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file generate_effective_potential.hpp
 *   
 *  \brief Contains implementation of sirius::Potential::generate_effective_potential() methods.
 */

inline void Potential::generate_effective_potential(Periodic_function<double>* rho, 
                                                    Periodic_function<double>* magnetization[3])
{
    PROFILE("sirius::Potential::generate_effective_potential");

    ctx_.fft().prepare(ctx_.gvec().partition());

    /* zero effective potential and magnetic field */
    zero();

    /* solve Poisson equation */
    poisson(rho, hartree_potential_);

    /* add Hartree potential to the total potential */
    effective_potential_->add(hartree_potential_);

    xc(rho, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
   
    effective_potential_->add(xc_potential_);
    
    if (ctx_.full_potential()) {
        effective_potential_->sync_mt();
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            effective_magnetic_field_[j]->sync_mt();
        }
    }

    //if (debug_level > 1) check_potential_continuity_at_mt();
    
    /* needed to symmetrize potential and magentic field */
    effective_potential_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        effective_magnetic_field_[j]->fft_transform(-1);
    }
    ctx_.fft().dismiss();
}

inline void Potential::generate_effective_potential(Periodic_function<double>* rho, 
                                                    Periodic_function<double>* rho_core, 
                                                    Periodic_function<double>* magnetization[3])
{
    PROFILE("sirius::Potential::generate_effective_potential");

    ctx_.fft().prepare(ctx_.gvec().partition());

    /* zero effective potential and magnetic field */
    zero();

    /* solve Poisson equation with valence density */
    poisson(rho, hartree_potential_);

    /* add Hartree potential to the effective potential */
    effective_potential_->add(hartree_potential_);

    /* create temporary function for rho + rho_core */
    Periodic_function<double> rhovc(ctx_, 0, 0);
    rhovc.zero();
    rhovc.add(rho);
    rhovc.add(rho_core);

    /* construct XC potentials from rho + rho_core */
    xc(&rhovc, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
    
    /* add XC potential to the effective potential */
    effective_potential_->add(xc_potential_);
    
    /* add local ionic potential to the effective potential */
    effective_potential_->add(local_potential_);

    /* get plane-wave coefficients of effective potential;
     * they will be used in two places:
     *  1) compute D-matrix
     *  2) establish a mapping between fine and coarse FFT grid for the Hloc operator */
    effective_potential_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        effective_magnetic_field_[j]->fft_transform(-1);
    }

    ctx_.fft().dismiss();

    generate_D_operator_matrix();
}

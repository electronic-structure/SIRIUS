// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

/** \file step_function.hpp

    \brief Contains remaining implementation of sirius::Step_function class. 
*/

Step_function::Step_function(Unit_cell* unit_cell__, Reciprocal_lattice* reciprocal_lattice__)
    : unit_cell_(unit_cell__), 
      reciprocal_lattice_(reciprocal_lattice__)
{
    Timer t("sirius::Step_function::Step_function");

    if (unit_cell__->num_atoms() == 0) return;

    auto fft = reciprocal_lattice_->fft();
    
    mdarray<double, 2> ffac(unit_cell_->num_atom_types(), (int)reciprocal_lattice_->num_gvec_shells_total());
    get_step_function_form_factors(ffac);

    step_function_pw_.resize(fft->size());
    step_function_.resize(fft->size());
    
    std::vector<complex16> f_pw = reciprocal_lattice_->make_periodic_function(ffac, fft->size());
    for (int ig = 0; ig < fft->size(); ig++) step_function_pw_[ig] = -f_pw[ig];
    step_function_pw_[0] += 1.0;

    fft->input(fft->size(), reciprocal_lattice_->fft_index(), &step_function_pw_[0]);
    fft->transform(1);
    fft->output(&step_function_[0]);
    
    double vit = 0.0;
    for (int i = 0; i < fft->size(); i++) vit += step_function_[i] * unit_cell_->omega() / fft->size();
    
    if (fabs(vit - unit_cell_->volume_it()) > 1e-10)
    {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << fabs(vit - unit_cell_->volume_it());
        warning_global(__FILE__, __LINE__, s);
    }
}

void Step_function::get_step_function_form_factors(mdarray<double, 2>& ffac)
{
    assert(ffac.size(0) == unit_cell_->num_atom_types());
    ffac.zero();
    
    splindex<block> spl_num_gvec_shells(ffac.size(1), Platform::num_mpi_ranks(), Platform::mpi_rank());

    #pragma omp parallel for default(shared)
    for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
    {
        int igs = spl_num_gvec_shells[igsloc];
        double g = reciprocal_lattice_->gvec_shell_len(igs);
        double g3inv = (igs) ? 1.0 / pow(g, 3) : 0.0;

        for (int iat = 0; iat < unit_cell_->num_atom_types(); iat++)
        {            
            double R = unit_cell_->atom_type(iat)->mt_radius();
            double gR = g * R;

            ffac(iat, igs) = (igs) ? (sin(gR) - gR * cos(gR)) * g3inv : pow(R, 3) / 3.0;
        }
    }

    Platform::allgather(&ffac(0, 0), unit_cell_->num_atom_types() * spl_num_gvec_shells.global_offset(), 
                        unit_cell_->num_atom_types() * spl_num_gvec_shells.local_size());
}

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

/** \file step_function.cpp
 *
 *  \brief Contains remaining implementation of sirius::Step_function class. 
 */

#include "step_function.h"

namespace sirius {

Step_function::Step_function(Reciprocal_lattice* reciprocal_lattice__,
                             FFT3D<CPU>* fft__,
                             Communicator const& comm__)
    : reciprocal_lattice_(reciprocal_lattice__),
      fft_(fft__),
      comm_(comm__)
{
    update();
}

mdarray<double, 2> Step_function::get_step_function_form_factors(int num_gsh)
{
    auto uc = reciprocal_lattice_->unit_cell();
    mdarray<double, 2> ffac(uc->num_atom_types(), num_gsh);
    
    splindex<block> spl_num_gvec_shells(num_gsh, comm_.size(), comm_.rank());

    #pragma omp parallel for default(shared)
    for (int igsloc = 0; igsloc < (int)spl_num_gvec_shells.local_size(); igsloc++)
    {
        int igs = (int)spl_num_gvec_shells[igsloc];
        double G = reciprocal_lattice_->gvec_shell_len(igs);
        double g3inv = (igs) ? 1.0 / pow(G, 3) : 0.0;

        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {            
            double R = uc->atom_type(iat)->mt_radius();
            double GR = G * R;

            ffac(iat, igs) = (igs) ? (sin(GR) - GR * cos(GR)) * g3inv : pow(R, 3) / 3.0;
        }
    }
    
    int ld = uc->num_atom_types(); 
    comm_.allgather(ffac.ptr(), static_cast<int>(ld * spl_num_gvec_shells.global_offset()), 
                    static_cast<int>(ld * spl_num_gvec_shells.local_size()));
    return ffac;
}

void Step_function::update()
{
    Timer t("sirius::Step_function::Step_function::update");

    auto uc = reciprocal_lattice_->unit_cell();

    if (uc->num_atoms() == 0) return;
    
    auto ffac = get_step_function_form_factors(reciprocal_lattice_->num_gvec_shells_total());

    step_function_pw_.resize(fft_->size());
    step_function_.resize(fft_->size());
    
    std::vector<double_complex> f_pw = reciprocal_lattice_->make_periodic_function(ffac, fft_->size());
    for (int ig = 0; ig < fft_->size(); ig++) step_function_pw_[ig] = -f_pw[ig];
    step_function_pw_[0] += 1.0;

    fft_->input(fft_->size(), fft_->index_map(), &step_function_pw_[0]);
    fft_->transform(1);
    fft_->output(&step_function_[0]);
    
    double vit = 0.0;
    for (int i = 0; i < fft_->size(); i++) vit += step_function_[i] * uc->omega() / fft_->size();
    
    if (fabs(vit - uc->volume_it()) > 1e-10)
    {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << fabs(vit - uc->volume_it());
        warning_global(__FILE__, __LINE__, s);
    }
}

}

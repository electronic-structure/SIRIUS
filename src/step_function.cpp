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

/** \file step_function.cpp
 *
 *  \brief Contains remaining implementation of sirius::Step_function class. 
 */

#include "step_function.h"

namespace sirius {

Step_function::Step_function(Unit_cell             const& unit_cell__,
                             FFT3D*                       fft__,
                             Gvec_FFT_distribution const& gvec_fft_distr__,
                             Communicator          const& comm__)
{
    PROFILE();

    if (unit_cell__.num_atoms() == 0) return;

    auto& gvec = gvec_fft_distr__.gvec();
    
    auto ffac = get_step_function_form_factors(gvec.num_shells(), unit_cell__, gvec, comm__);

    step_function_pw_.resize(gvec.num_gvec());
    step_function_.resize(fft__->local_size());
    
    std::vector<double_complex> f_pw = unit_cell__.make_periodic_function(ffac, gvec);
    for (int ig = 0; ig < gvec.num_gvec(); ig++) step_function_pw_[ig] = -f_pw[ig];
    step_function_pw_[0] += 1.0;
    
    fft__->prepare();
    fft__->transform<1>(gvec_fft_distr__, &step_function_pw_[gvec_fft_distr__.offset_gvec_fft()]);
    fft__->output(&step_function_[0]);
    fft__->dismiss();
    
    double vit = 0.0;
    for (int i = 0; i < fft__->local_size(); i++) vit += step_function_[i];
    vit *= (unit_cell__.omega() / fft__->size());
    fft__->comm().allreduce(&vit, 1);
    
    if (std::abs(vit - unit_cell__.volume_it()) > 1e-10)
    {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << std::abs(vit - unit_cell__.volume_it());
        WARNING(s);
    }
    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z1 = mdarray<double_complex, 1>(&step_function_pw_[0], fft_->size()).checksum();
    double d1 = mdarray<double, 1>(&step_function_[0], fft_->size()).checksum();
    DUMP("checksum(step_function): %18.10f", d1);
    DUMP("checksum(step_function_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
    #endif
}

mdarray<double, 2> Step_function::get_step_function_form_factors(int                 num_gsh,
                                                                 Unit_cell    const& unit_cell__,
                                                                 Gvec         const& gvec__,
                                                                 Communicator const& comm__) const
{
    mdarray<double, 2> ffac(unit_cell__.num_atom_types(), num_gsh);

    splindex<block> spl_num_gvec_shells(num_gsh, comm__.size(), comm__.rank());

    #pragma omp parallel for
    for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
    {
        int igs = spl_num_gvec_shells[igsloc];
        double G = gvec__.shell_len(igs);
        double g3inv = (igs) ? 1.0 / std::pow(G, 3) : 0.0;

        for (int iat = 0; iat < unit_cell__.num_atom_types(); iat++)
        {
            double R = unit_cell__.atom_type(iat).mt_radius();
            double GR = G * R;

            ffac(iat, igs) = (igs) ? (std::sin(GR) - GR * std::cos(GR)) * g3inv : std::pow(R, 3) / 3.0;
        }
    }

    int ld = unit_cell__.num_atom_types();
    comm__.allgather(ffac.at<CPU>(), ld * spl_num_gvec_shells.global_offset(), ld * spl_num_gvec_shells.local_size());
    return ffac;
}

}

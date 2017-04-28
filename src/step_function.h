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

/** \file step_function.h
 *
 *  \brief Contains definition and partial implementation of sirius::Step_function class. 
 */

#ifndef __STEP_FUNCTION_H__
#define __STEP_FUNCTION_H__

#include "simulation_context_base.h"

namespace sirius {

/// Unit step function is defined to be 1 in the interstitial and 0 inside muffin-tins.
/** Unit step function is constructed from it's plane-wave expansion coefficients which are computed
 *  analytically:
 *  \f[
 *      \Theta({\bf r}) = \sum_{\bf G} \Theta({\bf G}) e^{i{\bf Gr}},
 *  \f]
 *  where
 *  \f[
 *      \Theta({\bf G}) = \frac{1}{\Omega} \int \Theta({\bf r}) e^{-i{\bf Gr}} d{\bf r} = 
 *          \frac{1}{\Omega} \int_{\Omega} e^{-i{\bf Gr}} d{\bf r} - \frac{1}{\Omega} \int_{MT} e^{-i{\bf Gr}} d{\bf r} = 
 *          \delta_{\bf G, 0} - \sum_{\alpha} \frac{1}{\Omega} \int_{MT_{\alpha}} e^{-i{\bf Gr}} d{\bf r}  
 *  \f]
 *  Integral of a plane-wave over the muffin-tin volume is taken using the spherical expansion of the plane-wave 
 *  around central point \f$ \tau_{\alpha} \f$:
 *  \f[
 *      \int_{MT_{\alpha}} e^{-i{\bf Gr}} d{\bf r} = 
 *          e^{-i{\bf G\tau_{\alpha}}} \int_{MT_{\alpha}} 4\pi \sum_{\ell m} (-i)^{\ell} j_{\ell}(Gr) 
 *          Y_{\ell m}(\hat {\bf G}) Y_{\ell m}^{*}(\hat {\bf r}) r^2 \sin \theta dr d\phi d\theta
 *  \f]
 *  In the above integral only \f$ \ell=m=0 \f$ term survives. So we have:
 *  \f[
 *      \int_{MT_{\alpha}} e^{-i{\bf Gr}} d{\bf r} = 4\pi e^{-i{\bf G\tau_{\alpha}}} \Theta(\alpha, G)
 *  \f]
 *  where
 *  \f[
 *      \Theta(\alpha, G) = \int_{0}^{R_{\alpha}} \frac{\sin(Gr)}{Gr} r^2 dr = 
 *          \left\{ \begin{array}{ll} \displaystyle R_{\alpha}^3 / 3 & G=0 \\
 *          \Big( \sin(GR_{\alpha}) - GR_{\alpha}\cos(GR_{\alpha}) \Big) / G^3 & G \ne 0 \end{array} \right.
 *  \f]
 *  are the so-called step function form factors. With this we have a final expression for the plane-wave coefficients 
 *  of the unit step function:
 *  \f[
 *      \Theta({\bf G}) = \delta_{\bf G, 0} - \sum_{\alpha} \frac{4\pi}{\Omega} e^{-i{\bf G\tau_{\alpha}}} 
 *          \Theta(\alpha, G)
 *  \f]
 */
class Step_function
{
    private:

        /// Plane wave expansion coefficients of the step function.
        std::vector<double_complex> step_function_pw_;
        
        /// Step function on the real-space grid.
        std::vector<double> step_function_;

    public:
        
        /// Constructor
        Step_function(Simulation_context_base& ctx__)
        {
            PROFILE("sirius::Step_function::Step_function");

            if (ctx__.unit_cell().num_atoms() == 0) {
                return;
            }

            step_function_pw_.resize(ctx__.gvec().num_gvec());
            step_function_.resize(ctx__.fft().local_size());

            Radial_integrals_theta ri(ctx__.unit_cell(), ctx__.pw_cutoff(), 100);

            auto f_pw = ctx__.make_periodic_function<index_domain_t::global>([&ri](int iat, double g)
                                                                             {
                                                                                 return ri.value(iat, g);
                                                                             });

            for (int ig = 0; ig < ctx__.gvec().num_gvec(); ig++) {
                step_function_pw_[ig] = -f_pw[ig];
            }
            step_function_pw_[0] += 1.0;
            
            ctx__.fft().transform<1>(ctx__.gvec().partition(), &step_function_pw_[ctx__.gvec().partition().gvec_offset_fft()]);
            ctx__.fft().output(&step_function_[0]);
            
            double vit{0};
            for (int i = 0; i < ctx__.fft().local_size(); i++) {
                vit += step_function_[i];
            }
            vit *= (ctx__.unit_cell().omega() / ctx__.fft().size());
            ctx__.fft().comm().allreduce(&vit, 1);
            
            if (std::abs(vit - ctx__.unit_cell().volume_it()) > 1e-10) {
                std::stringstream s;
                s << "step function gives a wrong volume for IT region" << std::endl
                  << "  difference with exact value : " << std::abs(vit - ctx__.unit_cell().volume_it());
                WARNING(s);
            }
            if (ctx__.control().print_checksum_) {
                double_complex z1 = mdarray<double_complex, 1>(&step_function_pw_[0], ctx__.gvec().num_gvec()).checksum();
                double d1 = mdarray<double, 1>(&step_function_[0], ctx__.fft().local_size()).checksum();
                ctx__.fft().comm().allreduce(&d1, 1);
                if (ctx__.comm().rank() == 0) {
                    DUMP("checksum(step_function): %18.10f", d1);
                    DUMP("checksum(step_function_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
                }
            }
        }
       
        /// Return plane-wave coefficient of the step function.
        inline double_complex const& theta_pw(int ig__) const
        {
            assert(ig__ >= 0 && ig__ < (int)step_function_pw_.size());
            return step_function_pw_[ig__];
        }

        /// Return the value of the step function for the grid point ir.
        inline double theta_r(int ir__) const
        {
            assert(ir__ >= 0 && ir__ < (int)step_function_.size());
            return step_function_[ir__];
        }
};

};

#endif //  __STEP_FUNCTION_H__

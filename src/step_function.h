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

/** \file step_function.h
 *
 *  \brief Contains definition and partial implementation of sirius::Step_function class. 
 */

#ifndef __STEP_FUNCTION_H__
#define __STEP_FUNCTION_H__

#include "unit_cell.h"
#include "fft3d.hpp"

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
        Step_function(Unit_cell const& unit_cell__, 
                      FFT3D* fft__,
                      Gvec const& gvec__,
                      Communicator const& comm__)
        {
            PROFILE("sirius::Step_function::Step_function");

            if (unit_cell__.num_atoms() == 0) {
                return;
            }

            auto ffac = get_step_function_form_factors(gvec__.num_shells(), unit_cell__, gvec__, comm__);

            step_function_pw_.resize(gvec__.num_gvec());
            step_function_.resize(fft__->local_size());
            
            std::vector<double_complex> f_pw = unit_cell__.make_periodic_function(ffac, gvec__);
            for (int ig = 0; ig < gvec__.num_gvec(); ig++) {
                step_function_pw_[ig] = -f_pw[ig];
            }
            step_function_pw_[0] += 1.0;
            
            fft__->transform<1>(gvec__.partition(), &step_function_pw_[gvec__.partition().gvec_offset_fft()]);
            fft__->output(&step_function_[0]);
            
            double vit = 0.0;
            for (int i = 0; i < fft__->local_size(); i++) {
                vit += step_function_[i];
            }
            vit *= (unit_cell__.omega() / fft__->size());
            fft__->comm().allreduce(&vit, 1);
            
            if (std::abs(vit - unit_cell__.volume_it()) > 1e-10) {
                std::stringstream s;
                s << "step function gives a wrong volume for IT region" << std::endl
                  << "  difference with exact value : " << std::abs(vit - unit_cell__.volume_it());
                WARNING(s);
            }
            #ifdef __PRINT_OBJECT_CHECKSUM
            //double_complex z1 = mdarray<double_complex, 1>(&step_function_pw_[0], fft__->local_size()).checksum();
            double d1 = mdarray<double, 1>(&step_function_[0], fft__->local_size()).checksum();
            comm__.allreduce(&d1, 1);
            DUMP("checksum(step_function): %18.10f", d1);
            //DUMP("checksum(step_function_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
            #endif
        }

        /// Get \f$ \Theta(\alpha, G) \f$ form factors of the step function.
        /**
         *  \f[
         *      \Theta(\alpha, G) = \int_{0}^{R_{\alpha}} \frac{\sin(Gr)}{Gr} r^2 dr = 
         *          \left\{ \begin{array}{ll} \displaystyle R_{\alpha}^3 / 3 & G=0 \\
         *          \Big( \sin(GR_{\alpha}) - GR_{\alpha}\cos(GR_{\alpha}) \Big) / G^3 & G \ne 0 \end{array} \right.
         *  \f]
         */
        mdarray<double, 2> get_step_function_form_factors(int num_gsh,
                                                          Unit_cell const& unit_cell__,
                                                          Gvec const& gvec__,
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

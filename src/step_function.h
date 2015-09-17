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

/** \file step_function.h
 *
 *  \brief Contains definition and partial implementation of sirius::Step_function class. 
 */

#ifndef __STEP_FUNCTION_H__
#define __STEP_FUNCTION_H__

#include "unit_cell.h"
#include "reciprocal_lattice.h"

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

        Unit_cell const& unit_cell_;
        
        /// Reciprocal lattice for the unit cell.
        Reciprocal_lattice const* reciprocal_lattice_;
        
        /// Plane wave expansion coefficients of the step function.
        std::vector<double_complex> step_function_pw_;
        
        /// Step function on the real-space grid.
        std::vector<double> step_function_;

        Communicator const& comm_;

        FFT3D* fft_;

        Gvec const& gvec_;
       
        void init();

    public:
        
        /// Constructor
        Step_function(Unit_cell const& unit_cell_, 
                      Reciprocal_lattice const* reciprocal_lattice__,
                      FFT3D* fft__,
                      Gvec const& gvec__,
                      Communicator const& comm__);

        /// Get \f$ \Theta(\alpha, G) \f$ form factors of the step function.
        /**
         *  \f[
         *      \Theta(\alpha, G) = \int_{0}^{R_{\alpha}} \frac{\sin(Gr)}{Gr} r^2 dr = 
         *          \left\{ \begin{array}{ll} \displaystyle R_{\alpha}^3 / 3 & G=0 \\
         *          \Big( \sin(GR_{\alpha}) - GR_{\alpha}\cos(GR_{\alpha}) \Big) / G^3 & G \ne 0 \end{array} \right.
         *  \f]
         */
        mdarray<double, 2> get_step_function_form_factors(int num_gsh) const;
       
        /// Return plane-wave coefficient of the step function.
        inline double_complex theta_pw(int ig__) const
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

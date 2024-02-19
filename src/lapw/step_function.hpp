// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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
 *
 *  \brief Generate unit step function for LAPW method.
 */

#ifndef __STEP_FUNCTION_HPP__
#define __STEP_FUNCTION_HPP__

#include "function3d/make_periodic_function.hpp"

namespace sirius {

/// Utility function to generate LAPW unit step function.
inline double
unit_step_function_form_factors(double R__, double g__)
{
    if (g__ < 1e-12) {
        return std::pow(R__, 3) / 3.0;
    } else {
        return (std::sin(g__ * R__) - g__ * R__ * std::cos(g__ * R__)) / std::pow(g__, 3);
    }
}

/// Representation of the unit step function.
struct step_function_t
{
    /// Step function on the real-space grid.
    mdarray<double, 1> rg;
    /// Plane wave expansion coefficients of the step function (global array).
    mdarray<std::complex<double>, 1> pw;
};

/// Unit step function is defined to be 1 in the interstitial and 0 inside muffin-tins.
/** Unit step function is constructed from it's plane-wave expansion coefficients which are computed
 *  analytically:
 *  \f[
 *      \Theta({\bf r}) = \sum_{\bf G} \Theta({\bf G}) e^{i{\bf Gr}},
 *  \f]
 *  where
 *  \f[
 *      \Theta({\bf G}) = \frac{1}{\Omega} \int \Theta({\bf r}) e^{-i{\bf Gr}} d{\bf r} =
 *          \frac{1}{\Omega} \int_{\Omega} e^{-i{\bf Gr}} d{\bf r} - \frac{1}{\Omega} \int_{MT} e^{-i{\bf Gr}}
 *           d{\bf r} = \delta_{\bf G, 0} - \sum_{\alpha} \frac{1}{\Omega} \int_{MT_{\alpha}} e^{-i{\bf Gr}}
 *           d{\bf r}
 *  \f]
 *  Integralof a plane-wave over the muffin-tin volume is taken using the spherical expansion of the
 *  plane-wave around central point \f$ \tau_{\alpha} \f$:
 *  \f[ \int_{MT_{\alpha}} e^{-i{\bf Gr}} d{\bf r} = e^{-i{\bf G\tau_{\alpha}}}
 *   \int_{MT_{\alpha}} 4\pi \sum_{\ell m} (-i)^{\ell} j_{\ell}(Gr) Y_{\ell m}(\hat {\bf G}) Y_{\ell m}^{*}(\hat
 *   {\bf r}) r^2 \sin \theta dr d\phi d\theta
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
 *  are the so-called step function form factors. With this we have a final expression for the plane-wave
 *  coefficients of the unit step function:
 *  \f[ \Theta({\bf G}) = \delta_{\bf G, 0} - \sum_{\alpha}
 *   \frac{4\pi}{\Omega} e^{-i{\bf G\tau_{\alpha}}} \Theta(\alpha, G)
 *  \f]
 */
inline auto
init_step_function(Unit_cell const& uc__, fft::Gvec const& gv__, fft::Gvec_fft const& gvec_fft__,
                   mdarray<std::complex<double>, 2> const& phase_factors_t__, fft::spfft_transform_type<double> spfft__)
{
    auto v = make_periodic_function<false>(uc__, gv__, phase_factors_t__, [&](int iat, double g) {
        auto R = uc__.atom_type(iat).mt_radius();
        return unit_step_function_form_factors(R, g);
    });

    step_function_t theta;
    theta.rg = mdarray<double, 1>({spfft__.local_slice_size()});
    theta.pw = mdarray<std::complex<double>, 1>({gv__.num_gvec()});

    try {
        for (int ig = 0; ig < gv__.num_gvec(); ig++) {
            theta.pw[ig] = -v[ig];
        }
        theta.pw[0] += 1.0;

        std::vector<std::complex<double>> ftmp(gvec_fft__.count());
        gvec_fft__.scatter_pw_global(&theta.pw[0], &ftmp[0]);
        spfft__.backward(reinterpret_cast<double const*>(ftmp.data()), SPFFT_PU_HOST);
        double* theta_ptr = spfft__.local_slice_size() == 0 ? nullptr : &theta.rg[0];
        fft::spfft_output(spfft__, theta_ptr);
    } catch (...) {
        std::stringstream s;
        s << "Error creating step function" << std::endl
          << "  local_slice_size() = " << spfft__.local_slice_size() << std::endl
          << "  gvec_fft__.count() = " << gvec_fft__.count();
        RTE_THROW(s);
    }

    double vit{0};
    for (int i = 0; i < spfft__.local_slice_size(); i++) {
        vit += theta.rg[i];
    }
    vit *= (uc__.omega() / fft::spfft_grid_size(spfft__));
    mpi::Communicator(spfft__.communicator()).allreduce(&vit, 1);

    if (std::abs(vit - uc__.volume_it()) > 1e-10) {
        std::stringstream s;
        s << "step function gives a wrong volume for IT region" << std::endl
          << "  difference with exact value : " << std::abs(vit - uc__.volume_it());
        if (gv__.comm().rank() == 0) {
            RTE_WARNING(s);
        }
    }

    return theta;
}

} // namespace sirius

#endif

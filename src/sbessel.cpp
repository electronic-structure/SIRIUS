#include <gsl/gsl_sf_bessel.h>

#include "sbessel.hpp"

namespace sirius {

Spherical_Bessel_functions::Spherical_Bessel_functions(int lmax__,
                                                       Radial_grid<double> const& rgrid__,
                                                       double q__)
    : lmax_(lmax__)
    , q_(q__)
    , rgrid_(&rgrid__)
{
    assert(q_ >= 0);

    sbessel_ = std::vector<Spline<double>>(lmax__ + 2);
    for (int l = 0; l <= lmax__ + 1; l++) {
        sbessel_[l] = Spline<double>(rgrid__);
    }

    std::vector<double> jl(lmax__ + 2);
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        double t = rgrid__[ir] * q__;
        gsl_sf_bessel_jl_array(lmax__ + 1, t, &jl[0]);
        for (int l = 0; l <= lmax__ + 1; l++) {
            sbessel_[l](ir) = jl[l];
        }
    }

    for (int l = 0; l <= lmax__ + 1; l++) {
        sbessel_[l].interpolate();
    }
}

void
Spherical_Bessel_functions::sbessel(int lmax__, double t__, double* jl__)
{
    gsl_sf_bessel_jl_array(lmax__, t__, jl__);
}

void
Spherical_Bessel_functions::sbessel_deriv_q(int lmax__, double q__, double x__, double* jl_dq__)
{
    std::vector<double> jl(lmax__ + 2);
    sbessel(lmax__ + 1, x__ * q__, &jl[0]);

    for (int l = 0; l <= lmax__; l++) {
        if (q__ != 0) {
            jl_dq__[l] = (l / q__) * jl[l] - x__ * jl[l + 1];
        } else {
            if (l == 1) {
                jl_dq__[l] = x__ / 3;
            } else {
                jl_dq__[l] = 0;
            }
        }
    }
}

Spline<double> const&
Spherical_Bessel_functions::operator[](int l__) const
{
    assert(l__ <= lmax_);
    return sbessel_[l__];
}

/// Derivative of Bessel function with respect to q.
/** \f[
 *    \frac{\partial j_{\ell}(q x)}{\partial q} = \frac{\ell}{q} j_{\ell}(q x) - x j_{\ell+1}(q x)
 *  \f]
 */
Spline<double>
Spherical_Bessel_functions::deriv_q(int l__)
{
    assert(l__ <= lmax_);
    assert(q_ >= 0);
    Spline<double> s(*rgrid_);
    if (q_ != 0) {
        for (int ir = 0; ir < rgrid_->num_points(); ir++) {
            s(ir) = (l__ / q_) * sbessel_[l__](ir) - (*rgrid_)[ir] * sbessel_[l__ + 1](ir);
        }
    } else {
        if (l__ == 1) {
            for (int ir = 0; ir < rgrid_->num_points(); ir++) {
                s(ir) = (*rgrid_)[ir] / 3;
            }
        }
    }
    s.interpolate();
    return s;
}


}  // sirius

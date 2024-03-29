/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file matching_coefficients.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::Matching_coefficients class.
 */

#ifndef __MATCHING_COEFFICIENTS_HPP__
#define __MATCHING_COEFFICIENTS_HPP__

#include <gsl/gsl_sf_bessel.h>
#include "unit_cell/unit_cell.hpp"
#include "core/fft/gvec.hpp"

namespace sirius {

/** The following matching conditions must be fulfilled:
 *  \f[
 *   \frac{\partial^j}{\partial r^j} \sum_{L \nu} A_{L \nu}^{\bf k}({\bf G})u_{\ell \nu}(r)
 *   Y_{L}(\hat {\bf r}) \bigg|_{R^{MT}}  = \frac{\partial^j}{\partial r^j} \frac{4 \pi}{\sqrt \Omega}
 *   e^{i{\bf (G+k)\tau}} \sum_{L}i^{\ell} j_{\ell}(|{\bf G+k}|r) Y_{L}^{*}(\widehat {\bf G+k}) Y_{L}(\hat {\bf r})
 *   \bigg|_{R^{MT}}
 *  \f]
 *  where \f$ L = \{ \ell, m \} \f$. Dropping sum over L we arrive to the following system of linear equations:
 *  \f[ \sum_{\nu} \frac{\partial^j u_{\ell \nu}(r)}{\partial r^j} \bigg|_{R^{MT}} A_{L \nu}^{\bf k}({\bf G})
 *   = \frac{4 \pi}{\sqrt \Omega} e^{i{\bf (G+k)\tau}} i^{\ell} \frac{\partial^j j_{\ell}(|{\bf G+k}|r)}{\partial r^j}
 *      \bigg|_{R^{MT}} Y_{L}^{*}(\widehat {\bf G+k})
 *  \f]
 *  The matching coefficients are then equal to:
 *  \f[
 *   A_{L \nu}^{\bf k}({\bf G}) = \sum_{j} \bigg[ \frac{\partial^j u_{\ell \nu}(r)}{\partial r^j} \bigg|_{R^{MT}}
 *   \bigg]_{\nu j}^{-1} \frac{\partial^j j_{\ell}(|{\bf G+k}|r)}{\partial r^j} \bigg|_{R^{MT}}
 *   \frac{4 \pi}{\sqrt \Omega} i^{\ell} e^{i{\bf (G+k)\tau}} Y_{L}^{*}(\widehat {\bf G+k})
 *  \f]
 */
class Matching_coefficients // TODO: compute on GPU
{
  private:
    /// Description of the unit cell.
    Unit_cell const& unit_cell_;

    /// Description of the G+k vectors.
    fft::Gvec const& gkvec_;

    std::vector<double> gkvec_len_;

    /// Spherical harmonics Ylm(theta, phi) of the G+k vectors.
    mdarray<std::complex<double>, 2> gkvec_ylm_;

    /// Precomputed values for the linear equations for matching coefficients.
    mdarray<std::complex<double>, 4> alm_b_;

    /// Generate matching coefficients for a specific \f$ \ell \f$ and order.
    /** \param [in] ngk           Number of G+k vectors.
     *  \param [in] phase_factors Phase factors of G+k vectors.
     *  \param [in] iat           Index of atom type.
     *  \param [in] l             Orbital quantum nuber.
     *  \param [in] lm            Composite l,m index.
     *  \param [in] nu            Order of radial function \f$ u_{\ell \nu}(r) \f$ for which coefficients are generated.
     *  \param [in] A             Inverse matrix of radial derivatives.
     *  \param [out] alm          Pointer to alm coefficients.
     */
    template <int N, bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    inline void
    generate(int ngk, std::vector<std::complex<double>> const& phase_factors__, int iat, int l, int lm, int nu,
             r3::matrix<double> const& A, T* alm) const
    {
        std::complex<double> zt;

        for (int igk = 0; igk < ngk; igk++) {
            switch (N) {
                case 1: {
                    zt = alm_b_(0, igk, l, iat) * A(0, 0);
                    break;
                }
                case 2: {
                    zt = alm_b_(0, igk, l, iat) * A(nu, 0) + alm_b_(1, igk, l, iat) * A(nu, 1);
                    break;
                }
                case 3: {
                    zt = alm_b_(0, igk, l, iat) * A(nu, 0) + alm_b_(1, igk, l, iat) * A(nu, 1) +
                         alm_b_(2, igk, l, iat) * A(nu, 2);
                    break;
                }
            }
            if (conjugate) {
                alm[igk] = std::conj(phase_factors__[igk] * zt) * gkvec_ylm_(igk, lm);
            } else {
                alm[igk] = phase_factors__[igk] * zt * std::conj(gkvec_ylm_(igk, lm));
            }
        }
    }

  public:
    /// Constructor
    Matching_coefficients(Unit_cell const& unit_cell__, fft::Gvec const& gkvec__)
        : unit_cell_(unit_cell__)
        , gkvec_(gkvec__)
    {
        int lmax_apw  = unit_cell__.lmax_apw();
        int lmmax_apw = sf::lmmax(lmax_apw);

        gkvec_ylm_ = mdarray<std::complex<double>, 2>({gkvec_.count(), lmmax_apw});
        gkvec_len_.resize(gkvec_.count());

        /* get length and Ylm harmonics of G+k vectors */
        #pragma omp parallel
        {
            std::vector<std::complex<double>> ylm(lmmax_apw);

            #pragma omp for
            for (auto it : gkvec_) {
                auto gkvec_cart = gkvec_.gkvec_cart(it.igloc);
                /* get r, theta, phi */
                auto vs = r3::spherical_coordinates(gkvec_cart);

                gkvec_len_[it.igloc] = vs[0];
                /* get spherical harmonics */
                sf::spherical_harmonics(lmax_apw, vs[1], vs[2], &ylm[0]);

                for (int lm = 0; lm < lmmax_apw; lm++) {
                    gkvec_ylm_(it.igloc, lm) = ylm[lm];
                }
            }
        }

        alm_b_ = mdarray<std::complex<double>, 4>({3, gkvec_.count(), lmax_apw + 1, unit_cell_.num_atom_types()});
        alm_b_.zero();

        #pragma omp parallel
        {
            /* value and first two derivatives of spherical Bessel functions */
            mdarray<double, 2> sbessel_mt({lmax_apw + 2, 3});

            #pragma omp for
            for (int igk = 0; igk < gkvec_.count(); igk++) {
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                    double R = unit_cell_.atom_type(iat).mt_radius();

                    double RGk = R * gkvec_len_[igk];

                    /* compute values and first and second derivatives of the spherical Bessel functions
                       at the MT boundary */
                    gsl_sf_bessel_jl_array(lmax_apw + 1, RGk, &sbessel_mt(0, 0));

                    /* Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                     *
                     * In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,1}]]
                     * Out[]= (n SphericalBesselJ[n,a x])/x-a SphericalBesselJ[1+n,a x]
                     *
                     * In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,2}]]
                     * Out[]= (((-1+n) n-a^2 x^2) SphericalBesselJ[n,a x]+2 a x SphericalBesselJ[1+n,a x])/x^2
                     */
                    for (int l = 0; l <= lmax_apw; l++) { // TODO: move to sbessel class
                        sbessel_mt(l, 1) = -sbessel_mt(l + 1, 0) * gkvec_len_[igk] + (l / R) * sbessel_mt(l, 0);
                        sbessel_mt(l, 2) = 2 * gkvec_len_[igk] * sbessel_mt(l + 1, 0) / R +
                                           ((l - 1) * l - std::pow(RGk, 2)) * sbessel_mt(l, 0) / std::pow(R, 2);
                    }

                    for (int l = 0; l <= lmax_apw; l++) {
                        std::complex<double> z = std::pow(std::complex<double>(0, 1), l);
                        double f               = fourpi / std::sqrt(unit_cell_.omega());
                        alm_b_(0, igk, l, iat) = z * f * sbessel_mt(l, 0);
                        alm_b_(1, igk, l, iat) = z * f * sbessel_mt(l, 1);
                        alm_b_(2, igk, l, iat) = z * f * sbessel_mt(l, 2);
                    }
                }
            }
        }
    }

    /// Generate plane-wave matching coefficients for the radial solutions of a given atom.
    /** \param [in]  atom      Atom, for which matching coefficients are generated.
        \param [out] alm       Array of matching coefficients with dimension indices \f$ ({\bf G+k}, \xi) \f$.
     */
    template <bool conjugate, typename T, typename = std::enable_if_t<!std::is_scalar<T>::value>>
    void
    generate(Atom const& atom__, mdarray<T, 2>& alm__) const
    {
        auto& type = atom__.type();

        RTE_ASSERT(type.max_aw_order() <= 3);

        int iat = type.id();

        std::vector<std::complex<double>> phase_factors(gkvec_.count());
        for (auto it : gkvec_) {
            double phase            = twopi * dot(gkvec_.gkvec(it.igloc), atom__.position());
            phase_factors[it.igloc] = std::exp(std::complex<double>(0, phase));
        }

        const double eps{0.1};
        std::vector<r3::matrix<double>> Al(atom__.type().lmax_apw() + 1);
        /* create inverse matrix of radial derivatives for all values of \ell */
        for (int l = 0; l <= atom__.type().lmax_apw(); l++) {
            /* order of augmentation for a given orbital quantum number */
            int num_aw = static_cast<int>(type.aw_descriptor(l).size());
            r3::matrix<double> A;
            /* create matrix of radial derivatives */
            for (int order = 0; order < num_aw; order++) {
                for (int dm = 0; dm < num_aw; dm++) {
                    A(dm, order) = atom__.symmetry_class().aw_surface_deriv(l, order, dm);
                }
            }

            /* invert matrix of radial derivatives */
            switch (num_aw) {
                case 1: {
                    if (unit_cell_.parameters().cfg().control().verification() >= 1) {
                        if (std::abs(A(0, 0)) < eps * (1.0 / std::sqrt(unit_cell_.omega()))) {
                            std::stringstream s;
                            s << "Ill defined plane wave matching problem for atom type " << iat << ", l = " << l
                              << std::endl
                              << "  radial function value at the MT boundary : " << A(0, 0);
                            RTE_WARNING(s.str());
                        }
                    }

                    A(0, 0) = 1.0 / A(0, 0);
                    break;
                }
                case 2: {
                    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

                    if (unit_cell_.parameters().cfg().control().verification() >= 1) {
                        if (std::abs(det) < eps * (1.0 / std::sqrt(unit_cell_.omega()))) {
                            std::stringstream s;
                            s << "Ill defined plane wave matching problem for atom type " << iat << ", l = " << l
                              << std::endl
                              << "  radial function value at the MT boundary : " << A(0, 0);
                            RTE_WARNING(s.str());
                        }
                    }

                    std::swap(A(0, 0), A(1, 1));
                    A(0, 0) /= det;
                    A(1, 1) /= det;
                    A(0, 1) = -A(0, 1) / det;
                    A(1, 0) = -A(1, 0) / det;
                    break;
                }
                case 3: {
                    A = inverse(A);
                    break;
                }
            }
            Al[l] = A;
        }

        for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
            int l  = type.indexb(xi).am.l();
            int lm = type.indexb(xi).lm;
            int nu = type.indexb(xi).order;

            /* order of augmentation for a given orbital quantum number */
            int num_aw = static_cast<int>(type.aw_descriptor(l).size());

            switch (num_aw) {
                /* APW */
                case 1: {
                    generate<1, conjugate>(gkvec_.count(), phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi));
                    break;
                }
                /* LAPW */
                case 2: {
                    generate<2, conjugate>(gkvec_.count(), phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi));
                    break;
                }
                /* Super LAPW */
                case 3: {
                    generate<3, conjugate>(gkvec_.count(), phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi));
                    break;
                }
                default: {
                    RTE_THROW("wrong order of augmented wave");
                }
            }
        }
    }

    auto const&
    gkvec() const
    {
        return gkvec_;
    }
};

} // namespace sirius

#endif // __MATCHING_COEFFICIENTS_HPP__

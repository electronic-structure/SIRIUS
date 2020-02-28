// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file matching_coefficients.hpp
 *
 *  \brief Contains definition and partial implementation of sirius::Matching_coefficients class.
 */

#ifndef __MATCHING_COEFFICIENTS_HPP__
#define __MATCHING_COEFFICIENTS_HPP__

#include <gsl/gsl_sf_bessel.h>
#include "Unit_cell/unit_cell.hpp"
#include "SDDK/gvec.hpp"

namespace sirius {

/** The following matching conditions must be fulfilled:
 *  \f[
 *      \frac{\partial^j}{\partial r^j} \sum_{L \nu} A_{L \nu}^{\bf k}({\bf G})u_{\ell \nu}(r)
 *       Y_{L}(\hat {\bf r}) \bigg|_{R^{MT}}  = \frac{\partial^j}{\partial r^j} \frac{4 \pi}{\sqrt \Omega}
 *       e^{i{\bf (G+k)\tau}} \sum_{L}i^{\ell} j_{\ell}(|{\bf G+k}|r) Y_{L}^{*}(\widehat {\bf G+k}) Y_{L}(\hat {\bf r})
 * \bigg|_{R^{MT}} \f] where \f$ L = \{ \ell, m \} \f$. Dropping sum over L we arrive to the following system of linear
 * equations: \f[ \sum_{\nu} \frac{\partial^j u_{\ell \nu}(r)}{\partial r^j} \bigg|_{R^{MT}} A_{L \nu}^{\bf k}({\bf G})
 * = \frac{4 \pi}{\sqrt \Omega} e^{i{\bf (G+k)\tau}} i^{\ell} \frac{\partial^j j_{\ell}(|{\bf G+k}|r)}{\partial r^j}
 *      \bigg|_{R^{MT}} Y_{L}^{*}(\widehat {\bf G+k})
 *  \f]
 *  The matching coefficients are then equal to:
 *  \f[
 *      A_{L \nu}^{\bf k}({\bf G}) = \sum_{j} \bigg[ \frac{\partial^j u_{\ell \nu}(r)}{\partial r^j} \bigg|_{R^{MT}}
 * \bigg]_{\nu j}^{-1} \frac{\partial^j j_{\ell}(|{\bf G+k}|r)}{\partial r^j} \bigg|_{R^{MT}} \frac{4 \pi}{\sqrt \Omega}
 * i^{\ell} e^{i{\bf (G+k)\tau}} Y_{L}^{*}(\widehat {\bf G+k}) \f]
 */
class Matching_coefficients // TODO: compute on GPU
{
  private:
    /// Description of the unit cell.
    Unit_cell const& unit_cell_;

    /// Number of the G+k vectors for which matching coefficients are constructed.
    int num_gkvec_;

    /// Index of the G+k vectors inside gkvec class, for which the matching coefficients are constructed.
    std::vector<int>& igk_;

    /// Description of the G+k vectors.
    Gvec const& gkvec_;

    /// Spherical harmonics Ylm(theta, phi) of the G+k vectors.
    sddk::mdarray<double_complex, 2> gkvec_ylm_;

    std::vector<double> gkvec_len_;

    /// Precomputed values for the linear equations for matching coefficients.
    sddk::mdarray<double_complex, 4> alm_b_;

    /// Maximum l for APW functions.
    int lmax_apw_{-1};

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
    template <int N, bool conjugate>
    inline void generate(int ngk, std::vector<double_complex> const& phase_factors__, int iat, int l, int lm, int nu,
                         matrix3d<double> const& A, double_complex* alm) const
    {
        double_complex zt;

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
    Matching_coefficients(Unit_cell const& unit_cell__, int lmax_apw__, int num_gkvec__, std::vector<int>& igk__,
                          Gvec const& gkvec__)
        : unit_cell_(unit_cell__)
        , num_gkvec_(num_gkvec__)
        , igk_(igk__)
        , gkvec_(gkvec__)
        , lmax_apw_(lmax_apw__)
    {
        int lmmax_apw = utils::lmmax(lmax_apw__);

        gkvec_ylm_ = mdarray<double_complex, 2>(num_gkvec_, lmmax_apw);

        gkvec_len_.resize(num_gkvec_);

        /* get length and Ylm harmonics of G+k vectors */
        #pragma omp parallel
        {
            std::vector<double_complex> ylm(lmmax_apw);

            #pragma omp for
            for (int i = 0; i < num_gkvec_; i++) {
                auto gkvec_cart = gkvec_.gkvec_cart<index_domain_t::global>(igk_[i]);
                /* get r, theta, phi */
                auto vs = SHT::spherical_coordinates(gkvec_cart);

                /* get spherical harmonics */
                sf::spherical_harmonics(lmax_apw__, vs[1], vs[2], &ylm[0]);
                gkvec_len_[i] = vs[0];

                for (int lm = 0; lm < lmmax_apw; lm++) {
                    gkvec_ylm_(i, lm) = ylm[lm];
                }
            }
        }

        alm_b_ = mdarray<double_complex, 4>(3, num_gkvec_, lmax_apw__ + 1, unit_cell_.num_atom_types());
        alm_b_.zero();

        #pragma omp parallel
        {
            /* value and first two derivatives of spherical Bessel functions */
            mdarray<double, 2> sbessel_mt(lmax_apw__ + 2, 3);

            #pragma omp for
            for (int igk = 0; igk < num_gkvec_; igk++) {
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                    double R = unit_cell_.atom_type(iat).mt_radius();

                    double RGk = R * gkvec_len_[igk];

                    /* compute values and first and second derivatives of the spherical Bessel functions
                       at the MT boundary */
                    gsl_sf_bessel_jl_array(lmax_apw__ + 1, RGk, &sbessel_mt(0, 0));

                    /* Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                     *
                     * In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,1}]]
                     * Out[]= (n SphericalBesselJ[n,a x])/x-a SphericalBesselJ[1+n,a x]
                     *
                     * In[]:= FullSimplify[D[SphericalBesselJ[n,a*x],{x,2}]]
                     * Out[]= (((-1+n) n-a^2 x^2) SphericalBesselJ[n,a x]+2 a x SphericalBesselJ[1+n,a x])/x^2
                     */
                    for (int l = 0; l <= lmax_apw__; l++) { // TODO: move to sbessel class
                        sbessel_mt(l, 1) = -sbessel_mt(l + 1, 0) * gkvec_len_[igk] + (l / R) * sbessel_mt(l, 0);
                        sbessel_mt(l, 2) = 2 * gkvec_len_[igk] * sbessel_mt(l + 1, 0) / R +
                                           ((l - 1) * l - std::pow(RGk, 2)) * sbessel_mt(l, 0) / std::pow(R, 2);
                    }

                    for (int l = 0; l <= lmax_apw__; l++) {
                        double_complex z       = std::pow(double_complex(0, 1), l);
                        double f               = fourpi / std::sqrt(unit_cell_.omega());
                        alm_b_(0, igk, l, iat) = z * f * sbessel_mt(l, 0);
                        alm_b_(1, igk, l, iat) = z * f * sbessel_mt(l, 1);
                        alm_b_(2, igk, l, iat) = z * f * sbessel_mt(l, 2);
                    }
                }
            }
        }
    }

    /// Generate plane-wave matching coefficents for the radial solutions of a given atom.
    /** \param [in]  atom      Atom, for which matching coefficents are generated.
        \param [out] alm       Array of matching coefficients with dimension indices \f$ ({\bf G+k}, \xi) \f$.
     */
    template <bool conjugate>
    void generate(Atom const& atom__, sddk::mdarray<double_complex, 2>& alm__) const
    {
        auto& type = atom__.type();

        assert(type.max_aw_order() <= 3);

        int iat = type.id();

        std::vector<double_complex> phase_factors(num_gkvec_);
        for (int i = 0; i < num_gkvec_; i++) {
            double phase     = twopi * dot(gkvec_.gkvec(igk_[i]), atom__.position());
            phase_factors[i] = std::exp(double_complex(0, phase));
        }

        const double eps{0.1};
        std::vector<matrix3d<double>> Al(lmax_apw_ + 1);
        /* create inverse matrix of radial derivatives for all values of \ell */
        for (int l = 0; l <= lmax_apw_; l++) {
            /* order of augmentation for a given orbital quantum number */
            int num_aw = static_cast<int>(type.aw_descriptor(l).size());
            matrix3d<double> A;
            /* create matrix of radial derivatives */
            for (int order = 0; order < num_aw; order++) {
                for (int dm = 0; dm < num_aw; dm++) {
                    A(dm, order) = atom__.symmetry_class().aw_surface_deriv(l, order, dm);
                }
            }

            /* invert matrix of radial derivatives */
            switch (num_aw) {
                case 1: {
                    if (unit_cell_.parameters().control().verification_ >= 1) {
                        if (std::abs(A(0, 0)) < eps * (1.0 / std::sqrt(unit_cell_.omega()))) {
                            std::stringstream s;
                            s << "Ill defined plane wave matching problem for atom type " << iat << ", l = " << l
                              << std::endl
                              << "  radial function value at the MT boundary : " << A(0, 0);
                            WARNING(s.str());
                        }
                    }

                    A(0, 0) = 1.0 / A(0, 0);
                    break;
                }
                case 2: {
                    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

                    if (unit_cell_.parameters().control().verification_ >= 1) {
                        if (std::abs(det) < eps * (1.0 / std::sqrt(unit_cell_.omega()))) {
                            std::stringstream s;
                            s << "Ill defined plane wave matching problem for atom type " << iat << ", l = " << l
                              << std::endl
                              << "  radial function value at the MT boundary : " << A(0, 0);
                            WARNING(s.str());
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
            int l  = type.indexb(xi).l;
            int lm = type.indexb(xi).lm;
            int nu = type.indexb(xi).order;

            /* order of augmentation for a given orbital quantum number */
            int num_aw = static_cast<int>(type.aw_descriptor(l).size());

            switch (num_aw) {
                /* APW */
                case 1: {
                    generate<1, conjugate>(num_gkvec_, phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi));
                    break;
                }
                /* LAPW */
                case 2: {
                    generate<2, conjugate>(num_gkvec_, phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi));
                    break;
                }
                /* Super LAPW */
                case 3: {
                    generate<3, conjugate>(num_gkvec_, phase_factors, iat, l, lm, nu, Al[l], &alm__(0, xi));
                    break;
                }
                default: {
                    TERMINATE("wrong order of augmented wave");
                }
            }
        }
    }
};

} // namespace sirius

#endif // __MATCHING_COEFFICIENTS_HPP__

/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __LAPW_XC_ATOM_HPP__
#define __LAPW_XC_ATOM_HPP__

#include "function3d/kinetic_density.hpp"
#include "unit_cell/atom.hpp"

namespace sirius {

/// Muffin-tin valence fields and the adjoint in the complex R_l Y_lm basis.
/** The radial functions are snapshotted. The atom's radial grid and basis index
 *  must outlive this object. There is no smooth-field subtraction or PAW Q term.
 *  Core charge/tau and their responses are separate from this valence operator.
 */
class LAPW_xc_atom
{
  private:
    Atom_type const& type_;
    int lmmax_;
    mdarray<double, 2> radial_;
    Gaunt_coefficients<std::complex<double>> gaunt_;
    Spheric_kinetic_density_operator<std::complex<double>> kinetic_;

    static mdarray<double, 2>
    radial_functions(Atom const& atom__, bool derivative__ = false)
    {
        auto const& type = atom__.type();
        mdarray<double, 2> result({type.num_mt_points(), type.indexr().size()});
        for (auto const& rf : type.indexr()) {
            for (int ir = 0; ir < type.num_mt_points(); ir++) {
                result(ir, rf.idxrf) = derivative__ ? atom__.symmetry_class().radial_function_derivative(ir, rf.idxrf) /
                                                              type.radial_grid(ir)
                                                    : atom__.symmetry_class().radial_function(ir, rf.idxrf);
                if (!std::isfinite(result(ir, rf.idxrf))) {
                    RTE_THROW("LAPW XC atom: nonfinite radial function");
                }
            }
        }
        return result;
    }

  public:
    LAPW_xc_atom(Atom const& atom__, int lmax__)
        : type_(atom__.type())
        , lmmax_(sf::lmmax(lmax__))
        , radial_(radial_functions(atom__))
        , gaunt_(type_.indexr().lmax(), lmax__, type_.indexr().lmax(), SHT::gaunt_yry)
        , kinetic_(type_.radial_grid(), type_.indexb(), radial_, lmax__, radial_wave_function_t::R, {},
                   radial_functions(atom__, true))
    {
    }

    /// Return {rho, tau} per spin, with D_ij = sum_n f_n conj(c_ni) c_nj.
    std::array<std::vector<Flm>, 2>
    fields(mdarray<std::complex<double>, 3> const& dm__) const
    {
        int n  = type_.indexb().size();
        int ns = static_cast<int>(dm__.size(2));
        if (dm__.size(0) != n || dm__.size(1) != n || (ns != 1 && ns != 2)) {
            RTE_THROW("LAPW XC atom: incompatible density matrix");
        }
        for (int s = 0; s < ns; s++) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    auto value = dm__(i, j, s);
                    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
                        std::abs(value - std::conj(dm__(j, i, s))) > 1e-12 * (1 + std::abs(value))) {
                        RTE_THROW("LAPW XC atom: nonfinite or non-Hermitian density matrix");
                    }
                }
            }
        }
        std::array<std::vector<Flm>, 2> result;
        mdarray<std::complex<double>, 2> dm({n, n});
        for (int s = 0; s < ns; s++) {
            result[0].emplace_back(lmmax_, type_.radial_grid());
            result[1].emplace_back(lmmax_, type_.radial_grid());
            auto& rho = result[0].back();
            rho.zero();
            result[1].back().zero();
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    dm(i, j) = dm__(i, j, s);
                }
                for (int i = 0; i <= j; i++) {
                    for (auto const& g : gaunt_.gaunt_vector(type_.indexb(i).lm, type_.indexb(j).lm)) {
                        double weight = (i == j ? 1 : 2) * std::real(dm__(i, j, s) * g.coef);
                        for (int ir = 0; ir < type_.num_mt_points(); ir++) {
                            rho(g.lm3, ir) +=
                                    weight * radial_(ir, type_.indexb(i).idxrf) * radial_(ir, type_.indexb(j).idxrf);
                        }
                    }
                }
            }
            kinetic_.add_density(dm, result[1].back());
        }
        return result;
    }

    /// Complex Hermitian matrix from already-weighted radial-sample covectors.
    /** No radial quadrature, r^2 factor or occupation is applied again. */
    mdarray<std::complex<double>, 2>
    matrix_elements(Flm const& drho__, Flm const& dtau__) const
    {
        if (drho__.angular_domain_size() != lmmax_ || drho__.radial_grid().hash() != type_.radial_grid().hash()) {
            RTE_THROW("LAPW XC atom: incompatible charge covector");
        }
        auto result = kinetic_.matrix_elements_from_derivatives(dtau__);
        int n       = type_.indexb().size();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : gaunt_.gaunt_vector(type_.indexb(i).lm, type_.indexb(j).lm)) {
                    for (int ir = 0; ir < type_.num_mt_points(); ir++) {
                        result(i, j) += drho__(g.lm3, ir) * g.coef * radial_(ir, type_.indexb(i).idxrf) *
                                        radial_(ir, type_.indexb(j).idxrf);
                    }
                }
                result(j, i) = std::conj(result(i, j));
            }
        }
        return result;
    }
};

} // namespace sirius

#endif

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SPHERIC_KINETIC_DENSITY_HPP__
#define __SPHERIC_KINETIC_DENSITY_HPP__

#include <type_traits>
#include "core/sht/gaunt.hpp"
#include "core/sht/sht.hpp"
#include "function3d/spheric_function.hpp"
#include "unit_cell/radial_functions_index.hpp"
#include "unit_cell/basis_functions_index.hpp"

namespace sirius {

enum class radial_wave_function_t
{
    R,
    r_times_R
};

/// Atom-local valence tau and its variational matrix in an R_l(r) Y_lm basis.
/** T=double selects real harmonics (PAW), T=complex<double> complex harmonics
 *  (LAPW). The density and v_tau are always expanded in real harmonics.
 *  The Hermitian density matrix uses D_ij = sum_n f_n conj(c_ni) c_nj.
 *  Core-state tau must be supplied separately. PAW charge-compensation functions
 *  do not specify orbital gradients and are not included. The grid and basis
 *  index must outlive this class.
 */
template <typename T>
class Spheric_kinetic_density_operator
{
  private:
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>);

    Radial_grid<double> const& grid_;
    basis_functions_index const& indexb_;
    int lmmax_;
    Gaunt_coefficients<T> gaunt_;
    mdarray<double, 2> derivative_;
    mdarray<double, 2> angular_radial_;

    static T
    gaunt(int l1, int l3, int l2, int m1, int m3, int m2)
    {
        if constexpr (std::is_same_v<T, double>) {
            return SHT::gaunt_rrr(l1, l3, l2, m1, m3, m2);
        } else {
            return SHT::gaunt_yry(l1, l3, l2, m1, m3, m2);
        }
    }

    double
    radial_product(int i__, int j__, int L__, int ir__) const
    {
        auto const& a = indexb_[i__];
        auto const& b = indexb_[j__];
        int l1 = a.am.l(), l2 = b.am.l();
        double angular = 0.5 * (l1 * (l1 + 1) + l2 * (l2 + 1) - L__ * (L__ + 1));
        return 0.5 * (derivative_(ir__, a.idxrf) * derivative_(ir__, b.idxrf) +
                      angular * angular_radial_(ir__, a.idxrf) * angular_radial_(ir__, b.idxrf));
    }

    void
    check_field(Spheric_function<function_domain_t::spectral, double> const& f__) const
    {
        if (f__.angular_domain_size() != lmmax_ || f__.radial_grid().hash() != grid_.hash()) {
            RTE_THROW("atom-local kinetic density: incompatible spherical field");
        }
    }

  public:
    Spheric_kinetic_density_operator(Radial_grid<double> const& grid__, basis_functions_index const& indexb__,
                                     mdarray<double, 2> const& radial__, int lmax__,
                                     radial_wave_function_t storage__        = radial_wave_function_t::R,
                                     std::vector<int> const& radial_sizes__  = {},
                                     mdarray<double, 2> const& derivatives__ = {})
        : grid_(grid__)
        , indexb_(indexb__)
        , lmmax_(sf::lmmax(lmax__))
        , gaunt_(indexb__.indexr().lmax(), lmax__, indexb__.indexr().lmax(), gaunt)
        , derivative_({grid__.num_points(), indexb__.indexr().size()})
        , angular_radial_({grid__.num_points(), indexb__.indexr().size()})
    {
        if (grid_.num_points() < 4 || grid_.first() < 0 || radial__.size(0) != grid_.num_points() ||
            radial__.size(1) != indexb_.indexr().size() ||
            (!radial_sizes__.empty() && radial_sizes__.size() != radial__.size(1))) {
            RTE_THROW("atom-local kinetic density: invalid radial functions or grid");
        }
        if (derivatives__.size() &&
            (storage__ != radial_wave_function_t::R || derivatives__.size(0) != radial__.size(0) ||
             derivatives__.size(1) != radial__.size(1))) {
            RTE_THROW("atom-local kinetic density: incompatible supplied radial derivatives");
        }
        derivative_.zero();
        angular_radial_.zero();
        for (auto const& rf : indexb_.indexr()) {
            if (rf.am.s() != 0) {
                RTE_THROW("spin-orbit partial waves require a spinor kinetic-density operator");
            }
            int nr = radial_sizes__.empty() ? grid_.num_points() : radial_sizes__[rf.idxrf];
            if (nr < 4 || nr > grid_.num_points()) {
                RTE_THROW("atom-local kinetic density: invalid radial support");
            }
            // LAPW already stores the derivatives used by the kinetic Hamiltonian.
            if (derivatives__.size()) {
                for (int ir = 0; ir < nr; ir++) {
                    double value = radial__(ir, rf.idxrf), derivative = derivatives__(ir, rf.idxrf);
                    if (!std::isfinite(value) || !std::isfinite(derivative) ||
                        (grid_[ir] == 0 && rf.am.l() > 0 && value != 0)) {
                        RTE_THROW("atom-local kinetic density: invalid supplied radial values or derivatives");
                    }
                    derivative_(ir, rf.idxrf)     = derivative;
                    angular_radial_(ir, rf.idxrf) = rf.am.l() == 0   ? 0
                                                    : grid_[ir] != 0 ? value / grid_[ir]
                                                    : rf.am.l() == 1 ? derivative
                                                                     : 0;
                }
                continue;
            }
            // Differentiate only the supplied partial wave, not its artificial zero-padded tail.
            Radial_grid_ext<double> radial_grid(nr, grid_.x().at(memory_t::host));
            Spline<double> f(radial_grid);
            for (int ir = 0; ir < nr; ir++) {
                f(ir) = radial__(ir, rf.idxrf);
            }
            if (storage__ == radial_wave_function_t::r_times_R) {
                if (grid_.first() == 0 && f(0) != 0) {
                    RTE_THROW("r R(r) must vanish at the origin");
                }
                double origin{0};
                if (grid_.first() == 0) {
                    origin = f.interpolate().deriv(1, 0);
                }
                for (int ir = 0; ir < nr; ir++) {
                    f(ir) = grid_[ir] == 0 ? (rf.am.l() == 0 ? origin : 0) : f(ir) / grid_[ir];
                }
            }
            if (grid_.first() == 0 && rf.am.l() > 0 && f(0) != 0) {
                RTE_THROW("regular radial functions with l > 0 must vanish at the origin");
            }
            f.interpolate();
            for (int ir = 0; ir < nr; ir++) {
                derivative_(ir, rf.idxrf) = f.deriv(1, ir);
                if (grid_[ir] == 0 && rf.am.l() > 1) {
                    derivative_(ir, rf.idxrf) = 0;
                }
                // The angular gradient vanishes for s waves. For regular p waves R/r -> R'(0).
                angular_radial_(ir, rf.idxrf) = rf.am.l() == 0   ? 0
                                                : grid_[ir] != 0 ? f(ir) / grid_[ir]
                                                : rf.am.l() == 1 ? derivative_(ir, rf.idxrf)
                                                                 : 0;
            }
        }
    }

    /// Accumulate one spin component, with occupations already in the density matrix.
    void
    add_density(mdarray<std::complex<double>, 2> const& dm__,
                Spheric_function<function_domain_t::spectral, double>& tau__) const
    {
        check_field(tau__);
        if (dm__.size(0) != indexb_.size() || dm__.size(1) != indexb_.size()) {
            RTE_THROW("atom-local kinetic density: incompatible density matrix");
        }
        for (int j = 0; j < indexb_.size(); j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : gaunt_.gaunt_vector(indexb_[i].lm, indexb_[j].lm)) {
                    double w = (i == j ? 1 : 2) * std::real(dm__(i, j) * g.coef);
                    for (int ir = 0; ir < grid_.num_points(); ir++) {
                        tau__(g.lm3, ir) += w * radial_product(i, j, g.l3, ir);
                    }
                }
            }
        }
    }

    /// Matrix of 1/2 integral v_tau grad(phi_i)^* . grad(phi_j), without integration by parts.
    /** Uses the same radial quadrature as inner() for spherical fields, including
     *  its r^2 weight before spline interpolation. No boundary condition is imposed
     *  on the partial waves at the sphere surface.
     */
    mdarray<T, 2>
    matrix_elements(Spheric_function<function_domain_t::spectral, double> const& vtau__) const
    {
        check_field(vtau__);
        mdarray<T, 2> result({indexb_.size(), indexb_.size()});
        Spline<T> integrand(grid_);
        for (int j = 0; j < indexb_.size(); j++) {
            for (int i = 0; i <= j; i++) {
                for (int ir = 0; ir < grid_.num_points(); ir++) {
                    integrand(ir) = 0;
                    for (auto const& g : gaunt_.gaunt_vector(indexb_[i].lm, indexb_[j].lm)) {
                        integrand(ir) += vtau__(g.lm3, ir) * g.coef * radial_product(i, j, g.l3, ir);
                    }
                    integrand(ir) *= grid_[ir] * grid_[ir];
                }
                result(i, j) = integrand.interpolate().integrate(0);
                result(j, i) = sirius::conj(result(i, j));
            }
        }
        return result;
    }

    /// Matrix derivative from discrete spherical tau covectors, without quadrature.
    /** Use for derivatives returned by an atom-grid interpolation transpose.
     *  Unlike matrix_elements(), d_tau__ already contains the complete discrete
     *  energy weights. No radial integration, r^2 or occupation factor is added.
     */
    mdarray<T, 2>
    matrix_elements_from_derivatives(Spheric_function<function_domain_t::spectral, double> const& d_tau__) const
    {
        check_field(d_tau__);
        mdarray<T, 2> result({indexb_.size(), indexb_.size()});
        result.zero();
        for (int j = 0; j < indexb_.size(); j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : gaunt_.gaunt_vector(indexb_[i].lm, indexb_[j].lm)) {
                    for (int ir = 0; ir < grid_.num_points(); ir++) {
                        result(i, j) += d_tau__(g.lm3, ir) * g.coef * radial_product(i, j, g.l3, ir);
                    }
                }
                result(j, i) = sirius::conj(result(i, j));
            }
        }
        return result;
    }
};

} // namespace sirius

#endif

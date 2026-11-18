/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __ULTRASOFT_XC_CORRECTION_HPP__
#define __ULTRASOFT_XC_CORRECTION_HPP__

#include "unit_cell/atom_type.hpp"
#include "function3d/spheric_function.hpp"

namespace sirius {

/// Ultrasoft Q and kinetic augmentation with their common discrete projector adjoint.
/** The PP generator supplies independent r^2 Q_ij^L and r^2 T_ij^L tables.
 *  T includes the positive kinetic-density convention's factor 1/2. Neither
 *  a Laplacian of Q nor a charge-overlap integral determines T. Frozen cores
 *  are added separately by the reconstruction sampler.
 */
class Ultrasoft_xc_correction
{
    Atom_type const& type_;
    Gaunt_coefficients<double> gaunt_;

    static Atom_type const&
    checked_type(Atom_type const& type__)
    {
        type__.check_kinetic_augmentation();
        if (type__.radial_grid().first() <= 0 || type__.indexb().size() == 0) {
            RTE_THROW("ultrasoft XC requires projectors on a positive radial grid");
        }
        return type__;
    }

    double
    product(int i, int j, int l, int ir, int field) const
    {
        int a = type_.indexb(i).idxrf, b = type_.indexb(j).idxrf;
        double value = field == 0 ? type_.q_radial_function(a, b, l)(ir) : type_.kinetic_augmentation(a, b, l)[ir];
        return value / std::pow(type_.radial_grid(ir), 2);
    }

  public:
    explicit Ultrasoft_xc_correction(Atom_type const& type__)
        : type_(checked_type(type__))
        , gaunt_(type_.indexr().lmax(), 2 * type_.indexr().lmax(), type_.indexr().lmax(), SHT::gaunt_rrr)
    {
    }

    double
    support_radius() const
    {
        return type_.radial_grid().last();
    }

    std::array<std::vector<Flm>, 2>
    fields(mdarray<std::complex<double>, 3> const& dm__) const
    {
        int n = type_.indexb().size(), blocks = static_cast<int>(dm__.size(2));
        if (dm__.size(0) != n || dm__.size(1) != n || blocks < 1 || blocks > 3 ||
            (type_.spin_orbit_coupling() && blocks != 3)) {
            RTE_THROW("ultrasoft XC: incompatible projector density matrix");
        }
        int ns = blocks == 3 ? 4 : blocks;
        std::array<std::vector<Flm>, 2> result;
        for (int s = 0; s < ns; s++) {
            for (int f = 0; f < 2; f++) {
                result[f].emplace_back(sf::lmmax(2 * type_.indexr().lmax()), type_.radial_grid());
                result[f].back().zero();
            }
            for (int j = 0; j < n; j++) {
                for (int i = 0; i <= j; i++) {
                    std::complex<double> d;
                    if (blocks != 3) {
                        d = dm__(i, j, s);
                    } else if (s < 2) {
                        d = dm__(i, j, 0) + (1.0 - 2 * s) * dm__(i, j, 1);
                    } else {
                        auto ud = dm__(i, j, 2), du = std::conj(dm__(j, i, 2));
                        d = s == 2 ? ud + du : std::complex<double>(0, 1) * (ud - du);
                    }
                    for (auto const& g : gaunt_.gaunt_vector(type_.indexb(i).lm, type_.indexb(j).lm)) {
                        double weight = (i == j ? 1 : 2) * d.real() * g.coef;
                        for (int ir = 0; ir < type_.num_mt_points(); ir++) {
                            for (int f = 0; f < 2; f++) {
                                result[f][s](g.lm3, ir) += weight * product(i, j, g.l3, ir, f);
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    mdarray<double, 2>
    matrix_elements(Flm const& drho__, Flm const& dtau__) const
    {
        int n = type_.indexb().size();
        for (auto const* f : {&drho__, &dtau__}) {
            if (f->angular_domain_size() != sf::lmmax(2 * type_.indexr().lmax()) ||
                f->radial_grid().hash() != type_.radial_grid().hash()) {
                RTE_THROW("ultrasoft XC: incompatible radial covectors");
            }
        }
        mdarray<double, 2> result({n, n});
        result.zero();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : gaunt_.gaunt_vector(type_.indexb(i).lm, type_.indexb(j).lm)) {
                    for (int ir = 0; ir < type_.num_mt_points(); ir++) {
                        result(i, j) += g.coef * (drho__(g.lm3, ir) * product(i, j, g.l3, ir, 0) +
                                                  dtau__(g.lm3, ir) * product(i, j, g.l3, ir, 1));
                    }
                }
                result(j, i) = result(i, j);
            }
        }
        return result;
    }
};

} // namespace sirius
#endif

/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_CALLBACK_HPP__
#define __XC_CALLBACK_HPP__

#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace sirius {

class Simulation_context;

/// A fixed quadrature with each complete atom block owned by one context rank.
/** Coordinates and block centres are Cartesian bohr. Energy and descriptor
 *  weights are independent inputs and do not scale the physical fields.
 *  Construct after defining the cell and atoms. Changing the geometry requires
 *  a new point set. Registration copies the arrays into the context.
 */
struct xc_point_set_t
{
    std::vector<std::array<double, 3>> coordinates;
    std::vector<double> energy_weights, descriptor_weights;
    std::vector<int> block_sizes;
    std::vector<std::array<double, 3>> block_centres;

    explicit xc_point_set_t(Simulation_context const& ctx__);
    void
    validate(Simulation_context const& ctx__) const;

  private:
    std::array<double, 9> lattice_;
    std::vector<std::array<double, 3>> atoms_;
};

/// Physical orbital fields on the local FFT slab or supplied point rows.
/** Atomic units, Fortran shapes (num_points,num_spins) for rho and tau and
 *  (num_points,3,num_spins) for gradient. One spin means total fields, two
 *  mean up/down fields. No clipping, renormalization or quadrature weighting
 *  is applied. The neutralizing charged-cell background is excluded.
 *  All context ranks participate, including ranks with no local points.
 */
struct xc_callback_input_t
{
    int num_points, num_spins;
    double point_weight;
    double const* rho;
    double const* gradient;
    double const* tau;
    /// Non-null for supplied points. point_weight is then unused.
    xc_point_set_t const* points{nullptr};
};

/// Globally reduced XC energy and its local primitive-field derivatives.
/** For a possibly nonlocal discrete energy E=F({rho_p,g_p,tau_p},weights),
 *  the point callback returns raw dE/drho_p, dE/dg_p and dE/dtau_p.
 *  The FFT callback returns these derivatives divided by the uniform grid
 *  weight. Return the same total energy on every context rank. Derivatives
 *  have the input shapes and are initially zero. They must include all
 *  cross-point and cross-rank dependencies of F, including contributions at
 *  zero energy-weight points. SIRIUS applies the sampling/gradient adjoint
 *  and -1/2 div(v_tau grad(psi)); the evaluator must not apply them itself.
 */
struct xc_callback_output_t
{
    double energy{0};
    double* vrho;
    double* vgradient;
    double* vtau;
};

/// Register before constructing Density/Potential and retain throughout SCF.
using xc_callback_t = std::function<void(Simulation_context const&, xc_callback_input_t const&, xc_callback_output_t&)>;

void
validate_external_xc_context(Simulation_context const& ctx__);

} // namespace sirius
#endif

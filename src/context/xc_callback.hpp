/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __XC_CALLBACK_HPP__
#define __XC_CALLBACK_HPP__

#include <functional>
#include <array>
#include <vector>

namespace sirius {

class Simulation_context;

/// A fixed distributed quadrature, grouped into complete local atom blocks.
/** Coordinates and block centres are Cartesian bohr. Blocks and their rows
 *  belong to exactly one context rank, independent of FFT replicas. Energy and
 *  descriptor weights are distinct and never multiply the physical fields.
 *  Construct after defining the cell/atoms, populate the arrays, then register
 *  as immutable callback state. Geometry changes require a new point set.
 */
struct xc_point_set_t
{
    std::vector<std::array<double, 3>> coordinates;
    std::vector<double> energy_weights;
    std::vector<double> descriptor_weights;
    std::vector<int> block_sizes;
    std::vector<std::array<double, 3>> block_centres;

    explicit xc_point_set_t(Simulation_context const& ctx__);
    void
    validate(Simulation_context const& ctx__) const;

  private:
    std::array<double, 9> lattice_;
    std::vector<std::array<double, 3>> atoms_;
};

/// Primitive fields on the local FFT slab or explicit point rows, in atomic units.
/** Scalar arrays have Fortran shape (num_points, num_spins), gradients
 *  (num_points, 3, num_spins). One spin component denotes total density/tau,
 *  two components denote up/down. No density clipping or renormalization is
 *  applied. The context supplies the cell, atoms, FFT layout and communicators.
 */
struct xc_callback_input_t
{
    int num_points;
    int num_spins;
    double point_weight;
    double const* rho;
    double const* gradient;
    double const* tau;
    /// Non-null for arbitrary points. point_weight is then unused.
    xc_point_set_t const* points{nullptr};
};

/// Total XC energy and primitive-field derivatives.
/** energy is the same globally reduced energy on every context rank.
 *  For the FFT callback, derivatives are divided by the grid quadrature weight.
 *  With input.points != nullptr, return raw discrete covectors, NOT derivatives
 *  divided by a quadrature weight. Zero energy-weight rows may have nonzero
 *  derivatives through the nonlocal descriptors.
 *  vrho/vtau and vgradient have the corresponding input shapes.
 *  The caller initializes all outputs to zero.
 *  A nonlocal evaluator must return derivatives of the entire discrete energy,
 *  not derivatives of independent local energy terms. SIRIUS applies the
 *  gradient adjoint and the tau operator; the callback must not apply them.
 */
struct xc_callback_output_t
{
    double energy{0};
    double* vrho;
    double* vgradient;
    double* vtau;
};

/// Called collectively on the context communicator, including empty local work.
/** Register before constructing Density/Potential and keep it fixed during SCF. */
using xc_callback_t = std::function<void(Simulation_context const&, xc_callback_input_t const&, xc_callback_output_t&)>;

void
validate_external_xc_context(Simulation_context const& ctx__);

} // namespace sirius

#endif

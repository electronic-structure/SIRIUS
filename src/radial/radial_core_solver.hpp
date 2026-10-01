/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __RADIAL_CORE_SOLVER_HPP__
#define __RADIAL_CORE_SOLVER_HPP__

#include "bspline.hpp"
#include "spline.hpp"
#include "radial_solver.hpp"
#include "core/constants.hpp"
#include "core/la/eigensolver.hpp"
#include "core/la/linalg.hpp"

namespace sirius {

struct radial_core_state_t
{
    double energy;
    double kinetic_energy;
    std::vector<double> coefficients;
    /// Signed radial orbital R_l and its radial derivative.
    std::vector<double> value;
    std::vector<double> derivative;
    /// Spherical averages for one electron, without occupation or radial Jacobians.
    std::vector<double> rho;
    std::vector<double> tau;
};

/// Nonrelativistic radial generalized-Kohn-Sham solver with discrete density/tau derivatives.
/** The regular radial orbital is expanded as R_l(r) = r^l sum_i c_i B_i(r).
 *  Dropping the last B-spline imposes a vanishing orbital at the outer boundary.
 *  Retaining it permits a regular boundary solution for LAPW radial functions.
 *  The overlap and bare kinetic/local-potential operators use Gauss-Legendre
 *  integration on the knot intervals. XC inputs are raw derivatives with respect
 *  to the sampled spherical-average rho and positive tau. Their interpolation,
 *  angular integration and radial weights must already have been differentiated
 *  by the functional. No additional quadrature weights are applied here.
 *
 *  The radial potential includes -Z/r. Only its regular remainder is splined.
 *  Knots are logarithmic in 1 + Z*r to resolve the nuclear length scale while
 *  retaining the extended tail. The basis resolution is independent of the field grid and must be converged
 *  for the core states of interest. Relativistic core equations are not covered.
 */
class Radial_core_solver
{
  private:
    static constexpr int order{6};
    bspline_basis<order> basis_;
    int l_;
    Radial_grid<double> const& grid_;
    la::dmatrix<double> overlap_;
    la::dmatrix<double> kinetic_;
    la::dmatrix<double> h0_;

    struct sample_t
    {
        int index;
        double value;
        double derivative;
        double angular;
    };
    std::vector<std::vector<sample_t>> samples_;

    static std::vector<double>
    knots(Radial_grid<double> const& grid__, int num_basis__, double mt_radius__, double zn__,
          std::vector<double> const& extra__)
    {
        if (!std::isfinite(zn__) || zn__ < 0 || num_basis__ <= order || grid__.num_points() < 4 ||
            grid__.first() <= 0 || !std::isfinite(grid__.last()) || grid__.last() <= grid__.first()) {
            RTE_THROW("radial core solver: invalid basis or field grid");
        }
        std::vector<double> result(num_basis__ + order, 0);
        double scale  = 1 / std::max(1.0, zn__);
        double extent = std::log1p(grid__.last() / scale);
        for (int i = 1; i <= num_basis__ - order; i++) {
            result[order + i - 1] = scale * std::expm1(extent * i / (num_basis__ - order + 1));
        }
        std::fill(result.begin() + num_basis__, result.end(), grid__.last());
        for (double r : extra__) {
            if (!std::isfinite(r) || r <= 0 || r >= grid__.last()) {
                RTE_THROW("radial core solver: invalid refinement knot");
            }
            auto position = std::lower_bound(result.begin(), result.end(), r);
            if (position == result.end() || *position != r) {
                result.insert(position, r);
            }
        }
        if (mt_radius__ != 0) {
            if (!std::isfinite(mt_radius__) || mt_radius__ <= grid__.first() || mt_radius__ >= grid__.last()) {
                RTE_THROW("radial core solver: invalid response boundary");
            }
            // The orbital is continuous at the MT boundary, but its derivative
            // need not be when the tau-dependent operator stops there.
            auto first = std::lower_bound(result.begin(), result.end(), mt_radius__);
            auto last  = std::upper_bound(first, result.end(), mt_radius__);
            first      = result.erase(first, last);
            result.insert(first, order - 1, mt_radius__);
        }
        return result;
    }

    void
    check_covectors(std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        for (auto const* field : {&drho__, &dtau__}) {
            if (field->size() != samples_.size() ||
                !std::all_of(field->begin(), field->end(), [](double v) { return std::isfinite(v); })) {
                RTE_THROW("radial core solver: invalid density/tau covector");
            }
        }
    }

    la::dmatrix<double>
    hamiltonian(std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        check_covectors(drho__, dtau__);
        la::dmatrix<double> h(size(), size());
        for (int j = 0; j < size(); j++) {
            for (int i = 0; i < size(); i++) {
                h(i, j) = h0_(i, j);
            }
        }
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            for (auto const& a : samples_[ir]) {
                for (auto const& b : samples_[ir]) {
                    h(a.index, b.index) += (drho__[ir] * a.value * b.value +
                                            0.5 * dtau__[ir] * (a.derivative * b.derivative + a.angular * b.angular)) /
                                           fourpi;
                }
            }
        }
        return h;
    }

    double
    quadratic(la::dmatrix<double> const& matrix__, std::vector<double> const& c__) const
    {
        double result{0};
        for (int j = 0; j < size(); j++) {
            for (int i = 0; i < size(); i++) {
                result += c__[i] * matrix__(i, j) * c__[j];
            }
        }
        return result;
    }

  public:
    Radial_core_solver(Radial_grid<double> const& grid__, int l__, double zn__, std::vector<double> const& potential__,
                       int num_basis__, double mt_radius__ = 0, bool include_boundary__ = false,
                       std::vector<double> const& extra_knots__ = {})
        : basis_(knots(grid__, num_basis__, mt_radius__, zn__, extra_knots__), 16, include_boundary__)
        , l_(l__)
        , grid_(grid__)
        , overlap_(basis_.size() - !include_boundary__, basis_.size() - !include_boundary__)
        , kinetic_(basis_.size() - !include_boundary__, basis_.size() - !include_boundary__)
        , h0_(basis_.size() - !include_boundary__, basis_.size() - !include_boundary__)
        , samples_(grid__.num_points())
    {
        if (l_ < 0 || !std::isfinite(zn__) || zn__ < 0 || potential__.size() != samples_.size()) {
            RTE_THROW("radial core solver: invalid angular momentum, charge or potential");
        }
        Spline<double> regular_potential(grid__);
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            if (!std::isfinite(potential__[ir]) || !std::isfinite(grid__[ir]) || (ir && grid__[ir] <= grid__[ir - 1])) {
                RTE_THROW("radial core solver: nonfinite potential or invalid radial grid");
            }
            regular_potential(ir) = potential__[ir] + zn__ / grid__[ir];
            if (!std::isfinite(regular_potential(ir))) {
                RTE_THROW("radial core solver: nonfinite regular potential");
            }
        }
        regular_potential.interpolate();
        overlap_.zero();
        kinetic_.zero();
        h0_.zero();
        for (auto const& p : basis_.basis_pairs()) {
            for (size_t q = 0; q < p.r.size(); q++) {
                double r = p.r[q], rl = std::pow(r, l_);
                double ri = rl * p.Bi[q], rj = rl * p.Bj[q];
                double di        = rl * (p.dBi[q] + l_ * p.Bi[q] / r);
                double dj        = rl * (p.dBj[q] + l_ * p.Bj[q] / r);
                double potential = regular_potential.at_point(std::max(r, grid__.first())) - zn__ / r;
                double weight    = p.w[q] * r * r;
                double t         = 0.5 * weight * (di * dj + l_ * (l_ + 1.0) * ri * rj / (r * r));
                overlap_(p.i, p.j) += weight * ri * rj;
                kinetic_(p.i, p.j) += t;
                h0_(p.i, p.j) += t + weight * potential * ri * rj;
            }
        }
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            double r = grid__[ir], rl = std::pow(r, l_);
            double sample_r = r == mt_radius__ ? std::nextafter(r, 0.0) : r;
            for (int i = 0; i < size(); i++) {
                if (sample_r >= basis_.knot(i) && sample_r <= basis_.knot(i + order)) {
                    double b = basis_(i, sample_r);
                    samples_[ir].push_back({i, rl * b, rl * (basis_.deriv(i, sample_r) + l_ * b / r),
                                            std::sqrt(l_ * (l_ + 1.0)) * rl * b / r});
                }
            }
        }
    }

    int
    size() const
    {
        return overlap_.num_rows();
    }

    std::vector<double>
    breakpoints() const
    {
        std::vector<double> result;
        for (int i = 0; i < basis_.num_knots(); i++) {
            double r = basis_.knot(i);
            if (result.empty() || result.back() != r) {
                result.push_back(r);
            }
        }
        return result;
    }

    /// Regular radial solution with u(R)=1 and its analytic energy derivatives.
    radial_solver_result_t
    solve_at_energy(int dme__, double energy__, std::vector<double> const& drho__,
                    std::vector<double> const& dtau__) const
    {
        if (size() != basis_.size() || dme__ < 0 || !std::isfinite(energy__)) {
            RTE_THROW("radial boundary solve requires an open boundary and a finite energy");
        }
        auto h = hamiltonian(drho__, dtau__);
        int n = size(), interior = n - 1;
        std::vector<double> scale(n), c(n), rhs(interior);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(overlap_(i, i)) || overlap_(i, i) <= 0) {
                RTE_THROW("radial boundary solve: singular overlap");
            }
            scale[i] = 1 / std::sqrt(overlap_(i, i));
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                h(i, j) = (h(i, j) - energy__ * overlap_(i, j)) * scale[i] * scale[j];
            }
        }
        double boundary = std::pow(grid_.last(), -l_) / scale.back();
        for (int order = 0; order <= dme__; order++) {
            la::dmatrix<double> a(interior, interior);
            for (int i = 0; i < interior; i++) {
                rhs[i] = order ? 0 : -h(i, interior) * boundary;
                if (order) {
                    for (int j = 0; j < n; j++) {
                        rhs[i] += order * overlap_(i, j) * scale[i] * scale[j] * c[j];
                    }
                }
                for (int j = 0; j < interior; j++) {
                    a(i, j) = h(i, j);
                }
            }
            int info =
                    la::wrap(la::lib_t::lapack).gesv(interior, 1, a.at(memory_t::host), interior, rhs.data(), interior);
            if (info) {
                RTE_THROW("radial boundary solve failed: " + std::to_string(info));
            }
            if (!std::all_of(rhs.begin(), rhs.end(), [](double x) { return std::isfinite(x); })) {
                RTE_THROW("radial boundary solve returned nonfinite coefficients");
            }
            std::copy(rhs.begin(), rhs.end(), c.begin());
            c.back() = order ? 0 : boundary;
        }
        radial_solver_result_t result{
                0, std::vector<double>(samples_.size()), std::vector<double>(samples_.size()), {}};
        Spline<double> derivative(grid_);
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            double u{0}, du{0};
            for (auto const& s : samples_[ir]) {
                u += scale[s.index] * c[s.index] * s.value;
                du += scale[s.index] * c[s.index] * s.derivative;
            }
            result.p[ir]     = grid_[ir] * u;
            result.rdudr[ir] = grid_[ir] * du;
            derivative(ir)   = du;
            if (ir && result.p[ir] * result.p[ir - 1] < 0) {
                result.num_nodes++;
            }
        }
        result.uderiv = {result.p.back() / grid_.last(), derivative(grid_.num_points() - 1),
                         derivative.interpolate().deriv(1, grid_.num_points() - 1)};
        return result;
    }

    /// Evaluate the field map without renormalization; energy here is the bare T+V contraction.
    radial_core_state_t
    state(std::vector<double> const& coefficients__) const
    {
        if (coefficients__.size() != size() ||
            !std::all_of(coefficients__.begin(), coefficients__.end(), [](double v) { return std::isfinite(v); })) {
            RTE_THROW("radial core solver: invalid orbital coefficients");
        }
        radial_core_state_t result;
        result.coefficients   = coefficients__;
        result.energy         = quadratic(h0_, coefficients__);
        result.kinetic_energy = quadratic(kinetic_, coefficients__);
        result.rho.resize(samples_.size());
        result.tau.resize(samples_.size());
        result.value.resize(samples_.size());
        result.derivative.resize(samples_.size());
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            double value{0}, derivative{0}, angular{0};
            for (auto const& s : samples_[ir]) {
                value += coefficients__[s.index] * s.value;
                derivative += coefficients__[s.index] * s.derivative;
                angular += coefficients__[s.index] * s.angular;
            }
            result.rho[ir]        = value * value / fourpi;
            result.tau[ir]        = 0.5 * (derivative * derivative + angular * angular) / fourpi;
            result.value[ir]      = value;
            result.derivative[ir] = derivative;
        }
        return result;
    }

    /// Return the lowest radial eigenstates for this l, in ascending energy order.
    std::vector<radial_core_state_t>
    solve(int num_states__, std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        if (size() == basis_.size() || num_states__ <= 0 || num_states__ > size()) {
            RTE_THROW("radial core solver: invalid number of eigenstates");
        }
        int n  = size();
        auto h = hamiltonian(drho__, dtau__);
        la::dmatrix<double> s(n, n), eigenvectors(n, n);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                s(i, j) = overlap_(i, j);
            }
        }
        // Diagonal overlap scaling avoids the large dynamic range of r^l B_i near the origin.
        std::vector<double> scale(n), energies(n);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(s(i, i)) || s(i, i) <= 0) {
                RTE_THROW("radial core solver: singular overlap");
            }
            scale[i] = 1 / std::sqrt(s(i, i));
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                h(i, j) *= scale[i] * scale[j];
                s(i, j) *= scale[i] * scale[j];
                if (!std::isfinite(h(i, j)) || !std::isfinite(s(i, j))) {
                    RTE_THROW("radial core solver: nonfinite radial operator");
                }
            }
        }
        auto solver = la::Eigensolver_factory("lapack");
        int info    = solver->solve(n, num_states__, h, s, energies.data(), eigenvectors);
        if (info) {
            RTE_THROW("radial core eigensolver failed: " + std::to_string(info));
        }
        std::vector<radial_core_state_t> result;
        for (int j = 0; j < num_states__; j++) {
            std::vector<double> c(n);
            for (int i = 0; i < n; i++) {
                c[i] = scale[i] * eigenvectors(i, j);
            }
            result.push_back(state(c));
            result.back().energy = energies[j];
        }
        return result;
    }
};

/// Converge the occupied core eigenstates used by both density and LAPW basis construction.
inline std::vector<radial_core_state_t>
solve_radial_core_states(Radial_grid<double> const& grid__, int l__, int zn__, std::vector<double> const& potential__,
                         int count__, std::array<std::vector<double>, 2> fields__, int nmtp__)
{
    if (nmtp__ < 2 || nmtp__ >= grid__.num_points()) {
        RTE_THROW("radial core solve requires an extended muffin-tin domain");
    }
    for (auto& field : fields__) {
        if (field.size() != nmtp__) {
            RTE_THROW("radial core solve: inconsistent muffin-tin XC derivatives");
        }
        field.resize(grid__.num_points(), 0);
    }
    std::vector<radial_core_state_t> previous;
    std::ostringstream convergence;
    for (int nb : {64, 128, 192, 256, 384, 512, 768}) {
        Radial_core_solver solver(grid__, l__, zn__, potential__, nb, grid__[nmtp__ - 1]);
        auto states = solver.solve(count__, fields__[0], fields__[1]);
        if (!previous.empty()) {
            double energy_error{0}, field_error{0};
            int worst_state{0}, worst_ir{0}, worst_part{0};
            for (int j = 0; j < count__; j++) {
                energy_error = std::max(energy_error, std::abs(states[j].energy - previous[j].energy) /
                                                              (1e-10 * std::max(1.0, std::abs(states[j].energy))));
                int part{0};
                for (auto pair : {std::make_pair(&states[j].rho, &previous[j].rho),
                                  std::make_pair(&states[j].tau, &previous[j].tau)}) {
                    double peak = *std::max_element(pair.first->begin(), pair.first->end());
                    for (int ir = 0; ir < nmtp__; ir++) {
                        double error =
                                std::abs((*pair.first)[ir] - (*pair.second)[ir]) / (1e-6 * std::max(peak, 1e-30));
                        if (error > field_error) {
                            field_error = error;
                            worst_state = j;
                            worst_ir    = ir;
                            worst_part  = part;
                        }
                    }
                    part++;
                }
            }
            convergence << " basis=" << nb << " energy=" << energy_error << " fields=" << field_error
                        << " state=" << worst_state << " part=" << worst_part << " r=" << grid__[worst_ir];
            if (std::max(energy_error, field_error) <= 1) {
                return states;
            }
        }
        previous = std::move(states);
    }
    RTE_THROW("tau-dependent radial core solve did not converge with basis refinement; scaled errors:" +
              convergence.str());
    return {};
}

/// Remove core overlaps without changing the LAPW boundary value or its first two derivatives.
/** The compact projectors w(r) R_core(r), with w=(1-(r/R_MT)^2)^3, span
 *  the occupied core constraints and vanish through second order at R_MT.
 *  Projection is independent of the auxiliary radial energy, so it also applies
 *  to energy derivatives. The caller normalizes the resulting LAPW functions.
 */
inline void
orthogonalize_radial_to_core(Radial_grid<double> const& grid__, std::vector<radial_core_state_t> const& core__,
                             radial_solver_result_t& radial__)
{
    if (core__.empty()) {
        return;
    }
    int n = static_cast<int>(core__.size()), nr = grid__.num_points();
    if (radial__.p.size() != nr || radial__.rdudr.size() != nr) {
        RTE_THROW("core projection: inconsistent radial field size");
    }
    for (auto const& state : core__) {
        if (state.value.size() < nr || state.derivative.size() < nr) {
            RTE_THROW("core projection: missing signed core orbital or derivative");
        }
    }
    la::dmatrix<double> overlap(n, n);
    std::vector<double> rhs(n), weight(nr), derivative(nr);
    double radius = grid__.last();
    for (int ir = 0; ir < nr; ir++) {
        double x = grid__[ir] / radius, t = 1 - x * x;
        weight[ir]     = t * t * t;
        derivative[ir] = -6 * x * t * t / radius;
    }
    Spline<double> product(grid__);
    for (int i = 0; i < n; i++) {
        for (int ir = 0; ir < nr; ir++) {
            product(ir) = core__[i].value[ir] * radial__.p[ir];
        }
        rhs[i] = product.interpolate().integrate(1);
        for (int j = 0; j <= i; j++) {
            for (int ir = 0; ir < nr; ir++) {
                product(ir) = core__[i].value[ir] * core__[j].value[ir] * weight[ir];
            }
            overlap(i, j) = overlap(j, i) = product.interpolate().integrate(2);
        }
    }
    int info = la::wrap(la::lib_t::lapack).gesv(n, 1, overlap.at(memory_t::host), n, rhs.data(), n);
    if (info || !std::all_of(rhs.begin(), rhs.end(), [](double x) { return std::isfinite(x); })) {
        RTE_THROW("core projection: singular or nonfinite core overlap");
    }
    radial__.num_nodes = 0;
    for (int ir = 0; ir < nr; ir++) {
        for (int j = 0; j < n; j++) {
            radial__.p[ir] -= rhs[j] * grid__[ir] * weight[ir] * core__[j].value[ir];
            radial__.rdudr[ir] -= rhs[j] * grid__[ir] *
                                  (derivative[ir] * core__[j].value[ir] + weight[ir] * core__[j].derivative[ir]);
        }
        if (!std::isfinite(radial__.p[ir]) || !std::isfinite(radial__.rdudr[ir])) {
            RTE_THROW("core projection: nonfinite projected radial function");
        }
        if (ir && radial__.p[ir] * radial__.p[ir - 1] < 0) {
            radial__.num_nodes++;
        }
    }
}

} // namespace sirius

#endif

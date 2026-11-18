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

extern "C" void FORTRAN(dgbsv)(ftn_int* n, ftn_int* kl, ftn_int* ku, ftn_int* nrhs, ftn_double* ab, ftn_int* ldab,
                               ftn_int* ipiv, ftn_double* b, ftn_int* ldb, ftn_int* info);

extern "C" void FORTRAN(dsbgvx)(ftn_char jobz, ftn_char range, ftn_char uplo, ftn_int* n, ftn_int* ka, ftn_int* kb,
                                ftn_double* ab, ftn_int* ldab, ftn_double* bb, ftn_int* ldbb, ftn_double* q,
                                ftn_int* ldq, ftn_double* vl, ftn_double* vu, ftn_int* il, ftn_int* iu,
                                ftn_double* abstol, ftn_int* m, ftn_double* w, ftn_double* z, ftn_int* ldz,
                                ftn_double* work, ftn_int* iwork, ftn_int* ifail, ftn_int* info, ftn_len jobz_len,
                                ftn_len range_len, ftn_len uplo_len);

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

struct radial_core_response_t
{
    radial_core_state_t value;
    radial_core_state_t derivative;
};

/// Nonrelativistic or scalar-ZORA radial generalized-Kohn-Sham solver with discrete density/tau derivatives.
/** The regular radial orbital is expanded as R_l(r) = r^s sum_i c_i B_i(r).
 *  For ZORA at a Coulomb nucleus, s=sqrt(1+l(l+1)-(Z/c)^2)-1; otherwise s=l.
 *  Dropping the last B-spline imposes a vanishing orbital at the outer boundary.
 *  The overlap and bare kinetic/local-potential operators use Gauss-Legendre
 *  integration on the knot intervals. XC inputs are raw derivatives with respect
 *  to the sampled spherical-average rho and positive tau. Their interpolation,
 *  angular integration and radial weights must already have been differentiated
 *  by the functional. No additional quadrature weights are applied here.
 *
 *  The radial potential includes -Z/r. Only its regular remainder is splined.
 *  A zero basis size selects the finite radial space defined by the stored field
 *  grid, grading the unsampled-origin transition and including its extended tail.
 *  Independent refinement at fixed nodal XC
 *  covectors can create spurious point-localized kinetic directions. Positive
 *  basis sizes retain logarithmic knots for auxiliary radial boundary problems
 *  and controlled analytic tests. ZORA uses the bare local potential in its
 *  inverse mass; the XC tau covector acts on the unweighted positive orbital tau.
 *  A finite nonzero XC kinetic coefficient can regularize the ZORA origin.
 *  Such problems use the optional r^l origin rather than the bare Coulomb power.
 *  Energy-dependent masses and Dirac spinors are not covered.
 */
class Radial_core_solver
{
  private:
    static constexpr int order{6};
    bspline_basis<order> basis_;
    int l_;
    Radial_grid<double> const& grid_;
    double zn_;
    relativity_t relativity_;
    double radial_power_;
    Spline<double> regular_potential_;
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

    double
    inverse_mass(double potential__) const
    {
        if (relativity_ == relativity_t::none) {
            return 1;
        }
        double mass = radial_solver_local::rel_mass<relativity_t::zora>(0, potential__);
        if (!std::isfinite(mass) || mass <= 0) {
            RTE_THROW("radial core solver: nonpositive or nonfinite ZORA mass");
        }
        return 1 / mass;
    }

    static std::vector<double>
    knots(Radial_grid<double> const& grid__, int num_basis__, double mt_radius__, double zn__,
          std::vector<double> const& extra__, relativity_t relativity__)
    {
        if (!std::isfinite(zn__) || zn__ < 0 || (num_basis__ != 0 && num_basis__ <= order) || grid__.num_points() < 4 ||
            grid__.first() <= 0 || !std::isfinite(grid__.last()) || grid__.last() <= grid__.first()) {
            RTE_THROW("radial core solver: invalid basis or field grid");
        }
        std::vector<double> result;
        if (num_basis__ == 0) {
            result.assign(order, 0);
            auto points = grid__.values();
            // The C2 inner field continuation spans [0, r_first]. Abruptly smaller
            // orbital intervals amplify its nodal derivatives into localized kinetic modes.
            double last = 0, interval = points.front();
            for (auto point = points.begin(); point != points.end() - 1; ++point) {
                if (*point - last >= 0.5 * interval) {
                    interval = *point - last;
                    last     = *point;
                    result.push_back(*point);
                }
            }
            result.insert(result.end(), order, grid__.last());
        } else {
            result.assign(num_basis__ + order, 0);
            double scale = 1 / std::max(1.0, zn__);
            if (relativity__ == relativity_t::zora && zn__ > 0) {
                scale = std::min(scale, sq_alpha_half * zn__);
            }
            double extent = std::log1p(grid__.last() / scale);
            for (int i = 1; i <= num_basis__ - order; i++) {
                result[order + i - 1] = scale * std::expm1(extent * i / (num_basis__ - order + 1));
            }
            std::fill(result.begin() + num_basis__, result.end(), grid__.last());
            if (relativity__ == relativity_t::zora && zn__ > 0) {
                // Resolve the first sampled derivative independently of the Coulomb mass scale.
                double first_interval = result[order];
                for (double r = grid__.first(); r < first_interval; r *= 2) {
                    result.insert(std::lower_bound(result.begin(), result.end(), r), r);
                }
            }
        }
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

    /// At fixed grid points, H_xc = sum_p [(dE/d rho_p) d rho_p/dD + (dE/d tau_p) d tau_p/dD],
    /// where D is the radial orbital density matrix. Both field derivatives already contain the quadrature weights.
    void
    add_xc(la::dmatrix<double>& h__, std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        check_covectors(drho__, dtau__);
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            for (auto const& a : samples_[ir]) {
                for (auto const& b : samples_[ir]) {
                    h__(a.index, b.index) +=
                            (drho__[ir] * a.value * b.value +
                             0.5 * dtau__[ir] * (a.derivative * b.derivative + a.angular * b.angular)) /
                            fourpi;
                }
            }
        }
    }

    la::dmatrix<double>
    hamiltonian(std::vector<double> const& drho__, std::vector<double> const& dtau__) const
    {
        la::dmatrix<double> h(size(), size());
        copy(h0_, h);
        add_xc(h, drho__, dtau__);
        return h;
    }

    double
    contract(la::dmatrix<double> const& matrix__, std::vector<double> const& a__, std::vector<double> const& b__) const
    {
        long double result{0};
        for (int j = 0; j < size(); j++) {
            for (int i = std::max(0, j - order + 1); i <= std::min(size() - 1, j + order - 1); i++) {
                result += static_cast<long double>(a__[i]) * matrix__(i, j) * b__[j];
            }
        }
        return result;
    }

    double
    quadratic(la::dmatrix<double> const& matrix__, std::vector<double> const& c__) const
    {
        return contract(matrix__, c__, c__);
    }

    void
    check_coefficients(std::vector<double> const& coefficients__) const
    {
        if (coefficients__.size() != static_cast<size_t>(size()) ||
            !std::all_of(coefficients__.begin(), coefficients__.end(), [](double v) { return std::isfinite(v); })) {
            RTE_THROW("radial core solver: invalid orbital coefficients");
        }
    }

    /// Solve the normalized Sternheimer system by a banded minor and a 2x2 border.
    std::vector<double>
    solve_constrained(la::dmatrix<double> const& h__, double energy__, std::vector<double> const& c__,
                      std::vector<double> rhs__) const
    {
        int n = size();
        std::vector<double> scale(n + 1), sc(n);
        double border_norm{0};
        for (int i = 0; i < n; i++) {
            scale[i] = 1 / std::sqrt(std::max({std::abs(h__(i, i)), std::abs(energy__ * overlap_(i, i)), 1e-30}));
            long double product{0};
            for (int j = std::max(0, i - order + 1); j <= std::min(n - 1, i + order - 1); j++) {
                product += static_cast<long double>(overlap_(i, j)) * c__[j];
            }
            sc[i] = product;
            border_norm += std::pow(scale[i] * sc[i], 2);
        }
        if (!std::isfinite(border_norm) || border_norm <= 0) {
            RTE_THROW("radial core response: invalid normalization constraint");
        }
        scale[n] = 1 / std::sqrt(border_norm);
        rhs__.resize(n + 1, 0);
        int pivot{0};
        for (int i = 0; i < n; i++) {
            rhs__[i] *= scale[i];
            sc[i] *= scale[i] * scale[n];
            if (std::abs(sc[i]) > std::abs(sc[pivot]))
                pivot = i;
        }
        int m = n - 1, bandwidth = order - 1, leading = 3 * bandwidth + 1, nrhs = 3, info;
        std::vector<double> band(leading * m, 0), vectors(3 * m);
        std::vector<ftn_int> ipiv(m);
        auto index = [pivot](int i) { return i < pivot ? i : i + 1; };
        auto a     = [&](int i, int j) { return scale[i] * (h__(i, j) - energy__ * overlap_(i, j)) * scale[j]; };
        for (int j = 0; j < m; j++) {
            for (int i = std::max(0, j - bandwidth); i <= std::min(m - 1, j + bandwidth); i++) {
                band[2 * bandwidth + i - j + leading * j] = a(index(i), index(j));
            }
            vectors[j]         = rhs__[index(j)];
            vectors[j + m]     = a(index(j), pivot);
            vectors[j + 2 * m] = sc[index(j)];
        }
        FORTRAN(dgbsv)
        (&m, &bandwidth, &bandwidth, &nrhs, band.data(), &leading, ipiv.data(), vectors.data(), &m, &info);
        if (info) {
            RTE_THROW("radial core response: singular banded principal minor");
        }
        long double s00 = a(pivot, pivot), s01 = sc[pivot], s11{0}, f0 = rhs__[pivot], f1{0};
        for (int i = 0; i < m; i++) {
            double u = a(index(i), pivot), v = sc[index(i)];
            s00 -= static_cast<long double>(u) * vectors[i + m];
            s01 -= static_cast<long double>(u) * vectors[i + 2 * m];
            s11 -= static_cast<long double>(v) * vectors[i + 2 * m];
            f0 -= static_cast<long double>(u) * vectors[i];
            f1 -= static_cast<long double>(v) * vectors[i];
        }
        long double determinant = s00 * s11 - s01 * s01;
        double xp = (f0 * s11 - f1 * s01) / determinant, multiplier = (f1 * s00 - f0 * s01) / determinant;
        for (int i = 0; i < m; i++) {
            rhs__[index(i)] = vectors[i] - vectors[i + m] * xp - vectors[i + 2 * m] * multiplier;
        }
        rhs__[pivot] = xp;
        rhs__[n]     = multiplier;
        if (info || !std::all_of(rhs__.begin(), rhs__.end(), [](double x) { return std::isfinite(x); })) {
            RTE_THROW("radial core response: singular or nonfinite eigenstate response");
        }
        for (int i = 0; i <= n; i++) {
            rhs__[i] *= scale[i];
        }
        return rhs__;
    }

  public:
    Radial_core_solver(Radial_grid<double> const& grid__, int l__, double zn__, std::vector<double> const& potential__,
                       int num_basis__, double mt_radius__ = 0, std::vector<double> const& extra_knots__ = {},
                       relativity_t relativity__ = relativity_t::none, bool regular_origin__ = false)
        // The fractional ZORA origin power requires finer quadrature than r^l.
        : basis_(knots(grid__, num_basis__, mt_radius__, zn__, extra_knots__,
                       regular_origin__ ? relativity_t::none : relativity__),
                 relativity__ == relativity_t::zora ? 32 : 16, false,
                 !regular_origin__ && relativity__ == relativity_t::zora && zn__ > 0
                         ? std::max(1.0, 4 / radial_solver_local::scalar_relativistic_origin_power(l__, zn__))
                         : 1)
        , l_(l__)
        , grid_(grid__)
        , zn_(zn__)
        , relativity_(relativity__)
        , radial_power_(!regular_origin__ && relativity__ == relativity_t::zora
                                ? radial_solver_local::scalar_relativistic_origin_power(l__, zn__) - 1
                                : l__)
        , regular_potential_(grid__)
        , overlap_(basis_.size() - 1, basis_.size() - 1)
        , kinetic_(basis_.size() - 1, basis_.size() - 1)
        , h0_(basis_.size() - 1, basis_.size() - 1)
        , samples_(grid__.num_points())
    {
        if (l_ < 0 || !std::isfinite(zn__) || zn__ < 0 || potential__.size() != samples_.size()) {
            RTE_THROW("radial core solver: invalid angular momentum, charge or potential");
        }
        if (relativity_ != relativity_t::none && relativity_ != relativity_t::zora) {
            RTE_THROW("radial core solver: unsupported relativistic equation");
        }
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            if (!std::isfinite(potential__[ir]) || !std::isfinite(grid__[ir]) || (ir && grid__[ir] <= grid__[ir - 1])) {
                RTE_THROW("radial core solver: nonfinite potential or invalid radial grid");
            }
            regular_potential_(ir) = potential__[ir] + zn__ / grid__[ir];
            if (!std::isfinite(regular_potential_(ir))) {
                RTE_THROW("radial core solver: nonfinite regular potential");
            }
            inverse_mass(potential__[ir]);
        }
        regular_potential_.interpolate();
        overlap_.zero();
        kinetic_.zero();
        h0_.zero();
        for (auto const& p : basis_.basis_pairs()) {
            for (size_t q = 0; q < p.r.size(); q++) {
                double r = p.r[q], rl = std::pow(r, radial_power_);
                double ri = rl * p.Bi[q], rj = rl * p.Bj[q];
                double di        = rl * (p.dBi[q] + radial_power_ * p.Bi[q] / r);
                double dj        = rl * (p.dBj[q] + radial_power_ * p.Bj[q] / r);
                double potential = regular_potential_.at_point(std::max(r, grid__.first())) - zn__ / r;
                double weight    = p.w[q] * r * r;
                double t = 0.5 * weight * inverse_mass(potential) * (di * dj + l_ * (l_ + 1.0) * ri * rj / (r * r));
                overlap_(p.i, p.j) += weight * ri * rj;
                kinetic_(p.i, p.j) += t;
                h0_(p.i, p.j) += t + weight * potential * ri * rj;
            }
        }
        for (int ir = 0; ir < grid__.num_points(); ir++) {
            double r = grid__[ir], rl = std::pow(r, radial_power_);
            double sample_r = r == mt_radius__ ? std::nextafter(r, 0.0) : r;
            for (int i = 0; i < size(); i++) {
                if (sample_r >= basis_.knot(i) && sample_r <= basis_.knot(i + order)) {
                    double b = basis_(i, sample_r);
                    samples_[ir].push_back({i, rl * b, rl * (basis_.deriv(i, sample_r) + radial_power_ * b / r),
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

    /// Evaluate the field map without renormalization; energy here is the bare T+V contraction.
    radial_core_state_t
    state(std::vector<double> const& coefficients__) const
    {
        check_coefficients(coefficients__);
        radial_core_state_t result;
        result.coefficients   = coefficients__;
        result.energy         = quadratic(h0_, coefficients__);
        result.kinetic_energy = quadratic(kinetic_, coefficients__);
        result.rho.resize(samples_.size());
        result.tau.resize(samples_.size());
        result.value.resize(samples_.size());
        result.derivative.resize(samples_.size());
        for (size_t ir = 0; ir < samples_.size(); ir++) {
            // Near the nucleus, large B-spline derivative contributions cancel.
            long double value{0}, derivative{0}, angular{0};
            for (auto const& s : samples_[ir]) {
                long double coefficient = coefficients__[s.index];
                value += coefficient * s.value;
                derivative += coefficient * s.derivative;
                angular += coefficient * s.angular;
            }
            result.rho[ir]        = value * value / fourpi;
            result.tau[ir]        = 0.5 * (derivative * derivative + angular * angular) / fourpi;
            result.value[ir]      = value;
            result.derivative[ir] = derivative;
            if (!std::isfinite(result.rho[ir]) || !std::isfinite(result.tau[ir]) || !std::isfinite(value) ||
                !std::isfinite(derivative)) {
                RTE_THROW("radial core solver: nonfinite orbital or density/tau field");
            }
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
        int n           = size();
        auto physical_h = hamiltonian(drho__, dtau__);
        la::dmatrix<double> eigenvectors(n, num_states__);
        // Diagonal overlap scaling avoids the large dynamic range of r^l B_i near the origin.
        std::vector<double> scale(n), energies(n);
        for (int i = 0; i < n; i++) {
            if (!std::isfinite(overlap_(i, i)) || overlap_(i, i) <= 0) {
                RTE_THROW("radial core solver: singular overlap");
            }
            scale[i] = 1 / std::sqrt(overlap_(i, i));
        }
        int bandwidth = order - 1, leading = bandwidth + 1, one = 1, count = 0, info;
        std::vector<double> ab(leading * n, 0), bb(leading * n, 0), q(size_t(n) * n), work(7 * n);
        std::vector<ftn_int> iwork(5 * n), ifail(n);
        double vl = 0, vu = 0, tolerance = 2 * la::linalg_base::dlamch('S');
        for (int j = 0; j < n; j++) {
            for (int i = std::max(0, j - bandwidth); i <= j; i++) {
                auto offset = bandwidth + i - j + leading * j;
                ab[offset]  = physical_h(i, j) * scale[i] * scale[j];
                bb[offset]  = overlap_(i, j) * scale[i] * scale[j];
                if (!std::isfinite(ab[offset]) || !std::isfinite(bb[offset])) {
                    RTE_THROW("radial core solver: nonfinite radial operator");
                }
            }
        }
        FORTRAN(dsbgvx)
        ("V", "I", "U", &n, &bandwidth, &bandwidth, ab.data(), &leading, bb.data(), &leading, q.data(), &n, &vl, &vu,
         &one, &num_states__, &tolerance, &count, energies.data(), eigenvectors.at(memory_t::host), &n, work.data(),
         iwork.data(), ifail.data(), &info, 1, 1, 1);
        if (info || count != num_states__) {
            RTE_THROW("radial core eigensolver failed: " + std::to_string(info));
        }
        std::vector<radial_core_state_t> result;
        for (int j = 0; j < num_states__; j++) {
            std::vector<double> c(n);
            for (int i = 0; i < n; i++) {
                c[i] = scale[i] * eigenvectors(i, j);
            }
            // Refine low core eigenstates against the large near-nucleus kinetic spectrum,
            // also present in nonrelativistic bases with tightly spaced radial knots.
            for (int iteration = 0; iteration < 6; iteration++) {
                double energy = quadratic(physical_h, c) / quadratic(overlap_, c);
                std::vector<double> residual(n);
                for (int i = 0; i < n; i++) {
                    long double r{0};
                    for (int k = std::max(0, i - order + 1); k <= std::min(n - 1, i + order - 1); k++) {
                        r -= (static_cast<long double>(physical_h(i, k)) -
                              static_cast<long double>(energy) * overlap_(i, k)) *
                             c[k];
                    }
                    residual[i] = r;
                }
                auto correction = solve_constrained(physical_h, energy, c, std::move(residual));
                for (int i = 0; i < n; i++) {
                    c[i] += correction[i];
                }
                for (auto const& previous : result) {
                    double overlap = contract(overlap_, previous.coefficients, c);
                    for (int i = 0; i < n; i++) {
                        c[i] -= overlap * previous.coefficients[i];
                    }
                }
                double norm = std::sqrt(quadratic(overlap_, c));
                for (auto& value : c) {
                    value /= norm;
                }
            }
            energies[j] = quadratic(physical_h, c);
            result.push_back(state(c));
            result.back().energy = energies[j];
        }
        return result;
    }

    /// Differentiate core eigenstates at fixed radial grid, basis, nuclear charge and boundary.
    /** The potential direction varies both the local operator and the ZORA inverse mass.
     *  The two XC directions are
     *  variations of the already-weighted sample covectors, without further quadrature.
     *  The tangent orbitals satisfy c^T S dc = 0. Energies are eigenvalue derivatives.
     *  Optional reference states must be eigenstates of the same potential and XC operator. */
    std::vector<radial_core_response_t>
    solve_response(int num_states__, std::vector<double> const& drho__, std::vector<double> const& dtau__,
                   std::vector<double> const& potential_direction__, std::vector<double> const& rho_direction__,
                   std::vector<double> const& tau_direction__,
                   std::vector<radial_core_state_t> const* reference__ = nullptr) const
    {
        if (potential_direction__.size() != samples_.size() ||
            !std::all_of(potential_direction__.begin(), potential_direction__.end(),
                         [](double x) { return std::isfinite(x); })) {
            RTE_THROW("radial core response: invalid potential direction");
        }
        int n = size();
        la::dmatrix<double> dh(n, n), dk(n, n);
        dh.zero();
        dk.zero();
        Spline<double> direction(grid_, potential_direction__);
        for (auto const& p : basis_.basis_pairs()) {
            for (size_t q = 0; q < p.r.size(); q++) {
                double r = p.r[q], rl = std::pow(r, radial_power_);
                double dv = direction.at_point(std::max(r, grid_.first()));
                double ri = rl * p.Bi[q], rj = rl * p.Bj[q];
                double weight = p.w[q] * r * r;
                double dt{0};
                if (relativity_ == relativity_t::zora) {
                    double minv = inverse_mass(regular_potential_.at_point(std::max(r, grid_.first())) - zn_ / r);
                    double di   = rl * (p.dBi[q] + radial_power_ * p.Bi[q] / r);
                    double dj   = rl * (p.dBj[q] + radial_power_ * p.Bj[q] / r);
                    dt          = 0.5 * weight * sq_alpha_half * minv * minv * dv *
                                  (di * dj + l_ * (l_ + 1.0) * ri * rj / (r * r));
                }
                dk(p.i, p.j) += dt;
                dh(p.i, p.j) += weight * ri * rj * dv + dt;
            }
        }
        add_xc(dh, rho_direction__, tau_direction__);
        auto h      = hamiltonian(drho__, dtau__);
        auto states = reference__ ? *reference__ : solve(num_states__, drho__, dtau__);
        if (num_states__ <= 0 || states.size() != static_cast<size_t>(num_states__)) {
            RTE_THROW("radial core response: inconsistent reference states");
        }
        std::vector<radial_core_response_t> result;
        for (auto& value : states) {
            auto const& c = value.coefficients;
            check_coefficients(c);
            if (!std::isfinite(value.energy)) {
                RTE_THROW("radial core response: nonfinite reference energy");
            }
            std::vector<double> rhs(n);
            for (int i = 0; i < n; i++) {
                long double r{0};
                for (int j = std::max(0, i - order + 1); j <= std::min(n - 1, i + order - 1); j++) {
                    r -= static_cast<long double>(dh(i, j)) * c[j];
                }
                rhs[i] = r;
            }
            auto solution = solve_constrained(h, value.energy, c, std::move(rhs));
            std::vector<double> dc(solution.begin(), solution.begin() + n);
            auto tangent           = state(dc);
            tangent.energy         = -solution[n];
            tangent.kinetic_energy = quadratic(dk, c) + 2 * contract(kinetic_, c, dc);
            for (int ir = 0; ir < grid_.num_points(); ir++) {
                double product  = value.value[ir] * tangent.value[ir];
                tangent.rho[ir] = 2 * product / fourpi;
                tangent.tau[ir] = (value.derivative[ir] * tangent.derivative[ir] +
                                   l_ * (l_ + 1.0) * product / (grid_[ir] * grid_[ir])) /
                                  fourpi;
            }
            result.push_back({std::move(value), std::move(tangent)});
        }
        return result;
    }
};

} // namespace sirius

#endif

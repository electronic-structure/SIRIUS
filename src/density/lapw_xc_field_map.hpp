/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __LAPW_XC_FIELD_MAP_HPP__
#define __LAPW_XC_FIELD_MAP_HPP__

#include "density/density.hpp"
#include "function3d/lapw_xc_fields.hpp"
#include "function3d/smooth_periodic_function_sampler.hpp"
#include "function3d/spheric_function_interpolator.hpp"
#include "function3d/xc_point_fields.hpp"

namespace sirius {

/// Piecewise LAPW fields on a distributed point set, for one joint XC call.
/** Points inside a muffin-tin (including its surface) use that sphere's AE
 *  expansion; other points use the interstitial Fourier fields. No PAW-style
 *  addition/subtraction or second interstitial characteristic function applies.
 *  Spheres must not overlap. Stored physical rho/tau already include any core
 *  contribution and SCF mixing. They are not rebuilt from density matrices,
 *  whose basis may have changed since the physical fields were generated.
 *
 *  Points belong once to the context communicator, independently of FFT layout.
 *  All operations are collective, including empty point ranks. The geometry and
 *  radial sampling grids must remain fixed through the forward/reverse pair.
 *  Potential uses the adjoint for Hamiltonian assembly and discrete energy
 *  contractions. Spherical core covectors drive the separate radial core solve.
 */
class LAPW_xc_field_map
{
  private:
    struct source_t
    {
        std::vector<Spheric_function_interpolator> rho, tau;
        std::vector<r3::vector<double>> displacements;
        std::vector<int> targets;

        source_t(Density const& density__, int ia__)
        {
            auto const& grid = density__.ctx().unit_cell().atom(ia__).type().radial_grid();
            int lmmax        = sf::lmmax(density__.ctx().lmax_rho());
            Flm r(lmmax, grid), t(lmmax, grid);
            for (int s = 0; s < density__.ctx().num_spins(); s++) {
                for (size_t i = 0; i < r.size(); i++) {
                    r[i] = density__.rho().mt()[ia__][i];
                    t[i] = density__.kinetic_density().scalar().mt()[ia__][i];
                    if (density__.ctx().num_spins() == 2) {
                        r[i] = 0.5 * (r[i] + (1 - 2 * s) * density__.mag(0).mt()[ia__][i]);
                        t[i] = 0.5 * (t[i] + (1 - 2 * s) * density__.kinetic_density().vector(0).mt()[ia__][i]);
                    }
                    if (!std::isfinite(r[i]) || !std::isfinite(t[i])) {
                        RTE_THROW("LAPW XC field map: nonfinite physical muffin-tin field");
                    }
                }
                rho.emplace_back(r, spheric_inner_extension_t::charge);
                tau.emplace_back(t, spheric_inner_extension_t::kinetic);
            }
        }
    };

    Simulation_context const& ctx_;
    int num_points_;
    std::vector<source_t> sources_;
    std::vector<int> interstitial_targets_;
    std::unique_ptr<Smooth_periodic_function_sampler> interstitial_;

    void
    check(bool failed__, char const* message__) const
    {
        int failed = failed__;
        ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW(message__);
        }
    }

  public:
    LAPW_xc_field_map(Density const& density__, std::vector<r3::vector<double>> const& points__)
        : ctx_(density__.ctx())
        , num_points_(static_cast<int>(points__.size()))
    {
        bool failed{false};
        std::string message;
        std::vector<r3::vector<double>> interstitial_points;
        try {
            if (!ctx_.full_potential() || !density__.has_kinetic_density() || ctx_.num_mag_dims() == 3 ||
                ctx_.so_correction() || points__.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                RTE_THROW("LAPW XC field map requires scalar nonmagnetic/collinear full-potential fields");
            }
            for (auto const& p : points__) {
                for (int x = 0; x < 3; x++) {
                    if (!std::isfinite(p[x])) {
                        RTE_THROW("LAPW XC field map: nonfinite coordinate");
                    }
                }
            }
            auto const& uc = ctx_.unit_cell();
            std::vector<int> owner(num_points_, -1);
            for (int ia = 0; ia < uc.num_atoms(); ia++) {
                auto const& atom = uc.atom(ia);
                sources_.emplace_back(density__, ia);
                auto& source  = sources_.back();
                double radius = atom.type().radial_grid().last();
                auto limits   = r3::find_translations(radius, uc.lattice_vectors());
                for (int p = 0; p < num_points_; p++) {
                    auto d0    = points__[p] - r3::dot(uc.lattice_vectors(), atom.position());
                    auto shift = r3::dot(uc.inverse_lattice_vectors(), d0);
                    for (int x = 0; x < 3; x++) {
                        shift[x] = std::nearbyint(shift[x]);
                    }
                    d0 -= r3::dot(uc.lattice_vectors(), shift);
                    for (int z = -limits[2]; z <= limits[2]; z++) {
                        for (int y = -limits[1]; y <= limits[1]; y++) {
                            for (int x = -limits[0]; x <= limits[0]; x++) {
                                auto d = d0 - r3::dot(uc.lattice_vectors(), r3::vector<double>(x, y, z));
                                if (d.length() > radius) {
                                    continue;
                                }
                                if (owner[p] != -1 || d.length() < 1e-12) {
                                    RTE_THROW("LAPW XC field map: overlapping spheres or exact-nucleus target");
                                }
                                owner[p] = ia;
                                source.displacements.push_back(d);
                                source.targets.push_back(p);
                            }
                        }
                    }
                }
            }
            for (int p = 0; p < num_points_; p++) {
                if (owner[p] == -1) {
                    interstitial_targets_.push_back(p);
                    interstitial_points.push_back(points__[p]);
                }
            }
        } catch (std::exception const& e) {
            failed  = true;
            message = e.what();
        }
        check(failed, ("LAPW XC field map failed on a context rank: " + message).c_str());
        interstitial_ = std::make_unique<Smooth_periodic_function_sampler>(ctx_.gvec(), interstitial_points);
    }

    xc_point_fields_t
    sample(std::vector<Smooth_periodic_function<double>> const& rho__,
           std::vector<Smooth_periodic_function<double>> const& tau__) const
    {
        int ns = ctx_.num_spins();
        check(rho__.size() != static_cast<size_t>(ns) || tau__.size() != static_cast<size_t>(ns),
              "LAPW XC field map: incompatible spin fields");
        xc_point_fields_t result(num_points_, ns);
        for (int s = 0; s < ns; s++) {
            auto r = interstitial_->sample(rho__[s]);
            auto t = interstitial_->sample(tau__[s]);
            for (size_t j = 0; j < interstitial_targets_.size(); j++) {
                int p            = interstitial_targets_[j];
                result.rho(p, s) = r[j][0];
                result.tau(p, s) = t[j][0];
                for (int x = 0; x < 3; x++) {
                    result.gradient(p, x, s) = r[j][x + 1];
                }
            }
        }
        bool failed{false};
        try {
            for (auto const& source : sources_) {
                for (size_t j = 0; j < source.targets.size(); j++) {
                    int p = source.targets[j];
                    for (int s = 0; s < ns; s++) {
                        auto r           = source.rho[s].value_gradient(source.displacements[j]);
                        auto t           = source.tau[s].value_gradient(source.displacements[j]);
                        result.rho(p, s) = r[0];
                        result.tau(p, s) = t[0];
                        for (int x = 0; x < 3; x++) {
                            result.gradient(p, x, s) = r[x + 1];
                        }
                    }
                }
            }
        } catch (std::exception const&) {
            failed = true;
        }
        check(failed, "LAPW XC field map: MT point evaluation failed on a context rank");
        for (auto const* f : {&result.rho, &result.tau}) {
            for (size_t i = 0; i < f->size(); i++) {
                failed |= !std::isfinite((*f)[i]);
            }
        }
        for (size_t i = 0; i < result.gradient.size(); i++) {
            failed |= !std::isfinite(result.gradient[i]);
        }
        check(failed, "LAPW XC field map: point evaluation failed on a context rank");
        return result;
    }

    /// Adjoint of the discrete, fixed-geometry field selection and sampling.
    /** MT gradients return through the spherical interpolation transpose, not
     *  through a separately discretized divergence. Interstitial potentials
     *  already exclude MT points and must not acquire a second sphere mask.
     *  Core covectors are retained for the separate radial core response. Local
     *  valence covectors are not contracted with this snapshot's radial basis,
     *  which may be updated before the next Hamiltonian is built.
     */
    lapw_xc_field_adjoint_t
    adjoint(xc_point_fields_t const& d__) const
    {
        int ns       = ctx_.num_spins();
        bool invalid = d__.rho.size(0) != num_points_ || d__.rho.size(1) != ns || d__.tau.size(0) != num_points_ ||
                       d__.tau.size(1) != ns || d__.gradient.size(0) != num_points_ || d__.gradient.size(1) != 3 ||
                       d__.gradient.size(2) != ns;
        for (auto const* field : {&d__.rho, &d__.tau}) {
            for (size_t i = 0; i < field->size(); i++) {
                invalid |= !std::isfinite((*field)[i]);
            }
        }
        for (size_t i = 0; i < d__.gradient.size(); i++) {
            invalid |= !std::isfinite(d__.gradient[i]);
        }
        check(invalid, "LAPW XC field map: invalid point covectors");
        lapw_xc_field_adjoint_t result;
        bool failed{false};
        try {
            for (size_t ia = 0; ia < sources_.size(); ia++) {
                auto const& source = sources_[ia];
                auto const& type   = ctx_.unit_cell().atom(static_cast<int>(ia)).type();
                result.local.emplace_back();
                result.core.push_back(
                        {std::vector<double>(type.num_mt_points()), std::vector<double>(type.num_mt_points())});
                for (int s = 0; s < ns; s++) {
                    std::vector<std::array<double, 4>> dr(source.targets.size()), dt(source.targets.size());
                    for (size_t j = 0; j < source.targets.size(); j++) {
                        int p = source.targets[j];
                        dr[j] = {d__.rho(p, s), d__.gradient(p, 0, s), d__.gradient(p, 1, s), d__.gradient(p, 2, s)};
                        dt[j] = {d__.tau(p, s), 0, 0, 0};
                    }
                    result.local.back()[0].push_back(source.rho[s].adjoint(source.displacements, dr));
                    result.local.back()[1].push_back(source.tau[s].adjoint(source.displacements, dt));
                    for (int ir = 0; ir < type.num_mt_points(); ir++) {
                        result.core.back()[0][ir] += result.local.back()[0][s](0, ir) / (ns * y00);
                        result.core.back()[1][ir] += result.local.back()[1][s](0, ir) / (ns * y00);
                    }
                }
            }
        } catch (std::exception const&) {
            failed = true;
        }
        check(failed, "LAPW XC field map: MT adjoint failed on a context rank");
        auto validate = [&] {
            bool invalid{false};
            for (size_t ia = 0; ia < result.local.size(); ia++) {
                for (auto const& part : result.local[ia]) {
                    for (auto const& field : part) {
                        for (size_t i = 0; i < field.size(); i++) {
                            invalid |= !std::isfinite(field[i]);
                        }
                    }
                }
                for (auto const& c : result.core[ia]) {
                    for (double value : c) {
                        invalid |= !std::isfinite(value);
                    }
                }
            }
            check(invalid, "LAPW XC field map: nonfinite MT adjoint");
        };
        validate();
        for (size_t ia = 0; ia < result.local.size(); ia++) {
            for (auto& part : result.local[ia]) {
                for (auto& field : part) {
                    ctx_.comm().allreduce(field.at(memory_t::host), field.size());
                }
            }
            for (auto& c : result.core[ia]) {
                ctx_.comm().allreduce(c.data(), c.size());
            }
        }
        validate();
        for (int s = 0; s < ns; s++) {
            result.rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            result.tau.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            std::vector<std::array<double, 4>> dr(interstitial_targets_.size()), dt(interstitial_targets_.size());
            for (size_t j = 0; j < interstitial_targets_.size(); j++) {
                int p = interstitial_targets_[j];
                dr[j] = {d__.rho(p, s), d__.gradient(p, 0, s), d__.gradient(p, 1, s), d__.gradient(p, 2, s)};
                dt[j] = {d__.tau(p, s), 0, 0, 0};
            }
            interstitial_->adjoint(dr, result.rho.back());
            interstitial_->adjoint(dt, result.tau.back());
        }
        invalid = false;
        for (auto const* fields : {&result.rho, &result.tau}) {
            for (auto const& f : *fields) {
                for (auto value : f.f_pw_local()) {
                    invalid |= !std::isfinite(value.real()) || !std::isfinite(value.imag());
                }
            }
        }
        check(invalid, "LAPW XC field map: nonfinite interstitial adjoint");
        return result;
    }
};

} // namespace sirius

#endif

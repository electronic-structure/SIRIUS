/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PAW_XC_RECONSTRUCTION_HPP__
#define __PAW_XC_RECONSTRUCTION_HPP__

#include "density/density.hpp"
#include "density/paw_xc_correction.hpp"
#include "density/ultrasoft_xc_correction.hpp"
#include "function3d/spheric_function_interpolator.hpp"
#include "function3d/xc_point_fields.hpp"
#include <optional>

namespace sirius {

/// Sum PAW/ultrasoft corrections and pseudo cores on an arbitrary XC point set.
/** Each point belongs to exactly one context rank. Coordinates are Cartesian
 *  bohr and may lie outside the primitive cell. The density matrix is already
 *  accumulated over bands/k points and replicated, as in Density. This object
 *  snapshots its atom-local fields and source/image map for a single energy
 *  evaluation and its reverse pass. Geometry and radial grids must outlive it.
 *  All public operations are collective, including ranks without points.
 */
class PAW_xc_reconstruction
{
  private:
    struct source_t
    {
        std::optional<PAW_xc_correction> op;
        std::optional<Ultrasoft_xc_correction> us;
        int atom;
        std::array<std::vector<Flm>, 2> fields;
        std::vector<Spheric_function_interpolator> rho;
        std::vector<Spheric_function_interpolator> tau;
        std::vector<Spheric_function_interpolator> core;
        std::vector<r3::vector<double>> displacements;
        std::vector<r3::vector<double>> images;
        std::vector<int> targets;

        source_t(int atom__, Atom_type const& type__, mdarray<std::complex<double>, 3> const& dm__, bool spinor__)
            : atom(atom__)
        {
            if (type__.is_paw()) {
                op.emplace(type__);
                fields = op->fields(dm__, false);
            } else if (type__.augment()) {
                us.emplace(type__);
                fields = us->fields(dm__);
            } else {
                type__.check_ps_core_kinetic_density();
            }
            for (size_t s = 0; s < fields[0].size(); s++) {
                rho.emplace_back(fields[0][s], spheric_inner_extension_t::charge);
                tau.emplace_back(fields[1][s], spheric_inner_extension_t::kinetic);
            }
            // Keep the frozen core out of state-dependent spline differences near the nucleus.
            Flm field(1, type__.radial_grid());
            auto charge = type__.is_paw() ? type__.paw_ae_core_charge_density() : type__.ps_core_charge_density();
            charge.resize(type__.num_mt_points(), 0);
            auto kinetic = type__.is_paw() ? type__.paw_ae_core_kinetic_density()
                                           : (type__.has_ps_core_kinetic_density()
                                                      ? type__.ps_core_kinetic_density()
                                                      : std::vector<double>(type__.num_mt_points(), 0));
            for (auto const* values : std::array<std::vector<double> const*, 2>{&charge, &kinetic}) {
                for (int ir = 0; ir < type__.num_mt_points(); ir++) {
                    field(0, ir) = (*values)[ir] / ((spinor__ ? 1 : dm__.size(2)) * y00);
                }
                core.emplace_back(field, spheric_inner_extension_t::charge);
            }
        }
    };

    Simulation_context const& ctx_;
    int num_points_;
    std::vector<source_t> sources_;

    void
    check_buffers(mdarray<double, 2> const& rho__, mdarray<double, 3> const& grad__,
                  mdarray<double, 2> const& tau__) const
    {
        int ns     = ctx_.num_mag_dims() + 1;
        int failed = rho__.size(0) != num_points_ || rho__.size(1) != ns || grad__.size(0) != num_points_ ||
                     grad__.size(1) != 3 || grad__.size(2) != ns || tau__.size(0) != num_points_ || tau__.size(1) != ns;
        ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("PAW XC reconstruction: incompatible point buffers on a context rank");
        }
    }

  public:
    PAW_xc_reconstruction(Density const& density__, std::vector<r3::vector<double>> const& points__)
        : ctx_(density__.ctx())
        , num_points_(static_cast<int>(points__.size()))
    {
        int failed{0};
        std::string message;
        try {
            if (ctx_.full_potential() || points__.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                RTE_THROW("PAW XC reconstruction requires a pseudopotential context and a bounded point set");
            }
            auto const& uc = ctx_.unit_cell();
            std::vector<int> atoms;
            for (int ia = 0; ia < uc.num_atoms(); ia++) {
                auto const& type = uc.atom(ia).type();
                if (type.is_paw() || type.augment() || type.has_ps_core_kinetic_density()) {
                    atoms.push_back(ia);
                }
            }
            for (int ia : atoms) {
                auto const& atom = uc.atom(ia);
                sources_.emplace_back(ia, atom.type(), density__.density_matrix(ia), ctx_.num_mag_dims() == 3);
                auto& source   = sources_.back();
                double support = source.op ? source.op->support_radius() : atom.type().radial_grid().last();
                auto limits    = r3::find_translations(support, uc.lattice_vectors());
                for (int p = 0; p < num_points_; p++) {
                    auto displacement0 = points__[p] - r3::dot(uc.lattice_vectors(), atom.position());
                    auto fractional    = r3::dot(uc.inverse_lattice_vectors(), displacement0);
                    for (int x = 0; x < 3; x++) {
                        if (!std::isfinite(fractional[x])) {
                            RTE_THROW("PAW XC reconstruction: nonfinite point coordinate");
                        }
                        fractional[x] = std::nearbyint(fractional[x]);
                    }
                    displacement0 -= r3::dot(uc.lattice_vectors(), fractional);
                    // Enumerate all images in the support, not only the nearest atom image.
                    for (int z = -limits[2]; z <= limits[2]; z++) {
                        for (int y = -limits[1]; y <= limits[1]; y++) {
                            for (int x = -limits[0]; x <= limits[0]; x++) {
                                auto displacement =
                                        displacement0 - r3::dot(uc.lattice_vectors(),
                                                                r3::vector<double>({double(x), double(y), double(z)}));
                                double r = displacement.length();
                                if (r > support) {
                                    continue;
                                }
                                if (r < 1e-12) {
                                    RTE_THROW("PAW XC reconstruction: exact nuclei require a cusp-gradient convention");
                                }
                                source.displacements.push_back(displacement);
                                source.images.push_back(fractional +
                                                        r3::vector<double>({double(x), double(y), double(z)}));
                                source.targets.push_back(p);
                            }
                        }
                    }
                }
            }
        } catch (std::exception const& e) {
            failed  = 1;
            message = e.what();
        }
        ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("PAW XC reconstruction failed on a context rank: " + message);
        }
    }

    /// Differentiate the local reconstruction, including frozen-core fields.
    /** Density matrices, radial functions and image support membership are held
     *  fixed. The projector-density response must be added by the host. The
     *  lattice derivative holds Cartesian source atoms and target points fixed.
     */
    xc_geometry_adjoint_t
    geometry_adjoint(xc_point_fields_t const& derivatives__) const
    {
        auto const& r = derivatives__.rho;
        auto const& g = derivatives__.gradient;
        auto const& t = derivatives__.tau;
        check_buffers(r, g, t);
        int failed{0};
        for (int s = 0; s < ctx_.num_mag_dims() + 1; s++) {
            for (int p = 0; p < num_points_; p++) {
                failed |= !std::isfinite(r(p, s)) || !std::isfinite(t(p, s));
                for (int x = 0; x < 3; x++) {
                    failed |= !std::isfinite(g(p, x, s));
                }
            }
        }
        ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("PAW XC geometry adjoint: nonfinite covectors on a context rank");
        }
        xc_geometry_adjoint_t result(num_points_, ctx_.unit_cell().num_atoms());
        for (size_t i = 0; i < sources_.size(); i++) {
            auto const& source = sources_[i];
            int ia             = source.atom;
            // Frozen cores contribute only to the total spinor component.
            for (int s = 0; s < ctx_.num_mag_dims() + 1; s++) {
                for (size_t j = 0; j < source.targets.size(); j++) {
                    int p = source.targets[j];
                    std::array<double, 4> dr{r(p, s), g(p, 0, s), g(p, 1, s), g(p, 2, s)};
                    std::array<double, 4> dt{t(p, s), 0, 0, 0};
                    auto const& d = source.displacements[j];
                    r3::vector<double> v{0, 0, 0};
                    if (ctx_.num_mag_dims() != 3 || s == 0) {
                        v += source.core[0].position_adjoint(d, dr) + source.core[1].position_adjoint(d, dt);
                    }
                    if (source.op || source.us) {
                        v += source.rho[s].position_adjoint(d, dr) + source.tau[s].position_adjoint(d, dt);
                    }
                    result.points[p] += v;
                    for (int x = 0; x < 3; x++) {
                        result.atoms(x, ia) -= v[x];
                        for (int y = 0; y < 3; y++) {
                            result.lattice(x, y) -= v[x] * source.images[j][y];
                        }
                    }
                }
            }
        }
        ctx_.comm().allreduce(result.atoms.at(memory_t::host), result.atoms.size());
        ctx_.comm().allreduce(&result.lattice(0, 0), 9);
        return result;
    }

    /// Add the complete source/image sum to smooth rho, grad-rho and tau.
    /** Smooth rho excludes Q and pseudo-core charge. Smooth tau contains
     *  only pseudo-orbital gradients. No model weights act on physical fields.
     */
    void
    add_to(mdarray<double, 2>& rho__, mdarray<double, 3>& grad__, mdarray<double, 2>& tau__) const
    {
        check_buffers(rho__, grad__, tau__);
        for (auto const& source : sources_) {
            for (int s = 0; s < ctx_.num_mag_dims() + 1; s++) {
                for (size_t j = 0; j < source.targets.size(); j++) {
                    int p = source.targets[j];
                    std::array<double, 4> core_rho{}, core_tau{};
                    if (ctx_.num_mag_dims() != 3 || s == 0) {
                        core_rho = source.core[0].value_gradient(source.displacements[j]);
                        core_tau = source.core[1].value_gradient(source.displacements[j]);
                    }
                    if (source.op || source.us) {
                        auto rho = source.rho[s].value_gradient(source.displacements[j]);
                        auto tau = source.tau[s].value_gradient(source.displacements[j]);
                        for (int x = 0; x < 4; x++) {
                            core_rho[x] += rho[x];
                        }
                        core_tau[0] += tau[0];
                    }
                    rho__(p, s) += core_rho[0];
                    tau__(p, s) += core_tau[0];
                    for (int x = 0; x < 3; x++) {
                        grad__(p, x, s) += core_rho[x + 1];
                    }
                }
            }
        }
    }

    /// Global projector matrices from raw, already-weighted energy covectors.
    /** Results follow unit_cell().atoms(), with separate up/down matrices
     *  for collinear spin or total/z/x/y matrices for spinors. The smooth-field reverse operation is unchanged and
     *  must additionally return through its Fourier/interpolation and Q maps.
     */
    std::vector<mdarray<double, 3>>
    projector_matrices(mdarray<double, 2> const& drho__, mdarray<double, 3> const& dgrad__,
                       mdarray<double, 2> const& dtau__) const
    {
        check_buffers(drho__, dgrad__, dtau__);
        std::vector<mdarray<double, 3>> result(ctx_.unit_cell().num_atoms());
        for (size_t i = 0; i < sources_.size(); i++) {
            auto const& source = sources_[i];
            if (!source.op && !source.us) {
                continue;
            }
            int ia     = source.atom;
            int n      = ctx_.unit_cell().atom(ia).type().indexb().size();
            int ns     = ctx_.num_mag_dims() + 1;
            result[ia] = mdarray<double, 3>(std::array<index_range, 3>{n, n, ns});
            for (int s = 0; s < ns; s++) {
                std::vector<std::array<double, 4>> drho(source.targets.size()), dtau(source.targets.size());
                for (size_t j = 0; j < source.targets.size(); j++) {
                    int p   = source.targets[j];
                    drho[j] = {drho__(p, s), dgrad__(p, 0, s), dgrad__(p, 1, s), dgrad__(p, 2, s)};
                    dtau[j] = {dtau__(p, s), 0, 0, 0};
                }
                auto r      = source.rho[s].adjoint(source.displacements, drho);
                auto t      = source.tau[s].adjoint(source.displacements, dtau);
                auto matrix = source.op ? source.op->matrix_elements(r, t) : source.us->matrix_elements(r, t);
                for (int b = 0; b < n; b++) {
                    for (int a = 0; a < n; a++) {
                        result[ia](a, b, s) = matrix(a, b);
                    }
                }
            }
            ctx_.comm().allreduce(result[ia].at(memory_t::host), result[ia].size());
        }
        return result;
    }
};

} // namespace sirius

#endif

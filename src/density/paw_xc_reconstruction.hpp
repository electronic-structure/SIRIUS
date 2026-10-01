/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PAW_XC_RECONSTRUCTION_HPP__
#define __PAW_XC_RECONSTRUCTION_HPP__

#include "density/density.hpp"
#include "density/paw_xc_correction.hpp"
#include "function3d/spheric_function_interpolator.hpp"

namespace sirius {

/// Sum PAW source/image corrections on a distributed, arbitrary XC point set.
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
        PAW_xc_correction op;
        std::array<std::vector<Flm>, 2> fields;
        std::vector<Spheric_function_interpolator> rho;
        std::vector<Spheric_function_interpolator> tau;
        std::vector<Spheric_function_interpolator> core;
        std::vector<r3::vector<double>> displacements;
        std::vector<int> targets;

        source_t(Atom_type const& type__, mdarray<std::complex<double>, 3> const& dm__)
            : op(type__)
            , fields(op.fields(dm__, false))
        {
            for (size_t s = 0; s < fields[0].size(); s++) {
                rho.emplace_back(fields[0][s], spheric_inner_extension_t::charge);
                tau.emplace_back(fields[1][s], spheric_inner_extension_t::kinetic);
            }
            // Keep the frozen core out of state-dependent spline differences near the nucleus.
            Flm field(1, type__.radial_grid());
            for (auto const* values : {&type__.paw_ae_core_charge_density(), &type__.paw_ae_core_kinetic_density()}) {
                for (int ir = 0; ir < type__.num_mt_points(); ir++) {
                    field(0, ir) = (*values)[ir] / (dm__.size(2) * y00);
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
        int ns     = ctx_.num_spins();
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
            if (ctx_.full_potential() || ctx_.num_mag_dims() == 3 ||
                points__.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                RTE_THROW("PAW XC reconstruction requires scalar nonmagnetic/collinear PAW fields");
            }
            auto const& uc = ctx_.unit_cell();
            for (int i = 0; i < uc.num_paw_atoms(); i++) {
                int ia           = uc.paw_atom_index(paw_atom_index_t::global(i));
                auto const& atom = uc.atom(ia);
                sources_.emplace_back(atom.type(), density__.density_matrix(ia));
                auto& source   = sources_.back();
                double support = source.op.support_radius();
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

    /// Add the complete source/image sum to smooth rho, grad-rho and tau.
    /** Smooth rho excludes Q and pseudo-core charge. Smooth tau contains
     *  only pseudo-orbital gradients. No model weights act on physical fields.
     */
    void
    add_to(mdarray<double, 2>& rho__, mdarray<double, 3>& grad__, mdarray<double, 2>& tau__) const
    {
        check_buffers(rho__, grad__, tau__);
        for (auto const& source : sources_) {
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (size_t j = 0; j < source.targets.size(); j++) {
                    int p         = source.targets[j];
                    auto rho      = source.rho[s].value_gradient(source.displacements[j]);
                    auto tau      = source.tau[s].value_gradient(source.displacements[j]);
                    auto core_rho = source.core[0].value_gradient(source.displacements[j]);
                    auto core_tau = source.core[1].value_gradient(source.displacements[j]);
                    rho__(p, s) += rho[0] + core_rho[0];
                    tau__(p, s) += tau[0] + core_tau[0];
                    for (int x = 0; x < 3; x++) {
                        grad__(p, x, s) += rho[x + 1] + core_rho[x + 1];
                    }
                }
            }
        }
    }

    /// Global projector matrices from raw, already-weighted energy covectors.
    /** Results follow unit_cell().paw_atoms(), with separate up/down matrices
     *  for collinear spin. The smooth-field reverse operation is unchanged and
     *  must additionally return through its Fourier/interpolation and Q maps.
     */
    std::vector<mdarray<double, 3>>
    projector_matrices(mdarray<double, 2> const& drho__, mdarray<double, 3> const& dgrad__,
                       mdarray<double, 2> const& dtau__) const
    {
        check_buffers(drho__, dgrad__, dtau__);
        std::vector<mdarray<double, 3>> result;
        for (size_t i = 0; i < sources_.size(); i++) {
            auto const& source = sources_[i];
            int ia             = ctx_.unit_cell().paw_atom_index(paw_atom_index_t::global(static_cast<int>(i)));
            int n              = ctx_.unit_cell().atom(ia).type().indexb().size();
            result.emplace_back(std::array<index_range, 3>{n, n, ctx_.num_spins()});
            for (int s = 0; s < ctx_.num_spins(); s++) {
                std::vector<std::array<double, 4>> drho(source.targets.size()), dtau(source.targets.size());
                for (size_t j = 0; j < source.targets.size(); j++) {
                    int p   = source.targets[j];
                    drho[j] = {drho__(p, s), dgrad__(p, 0, s), dgrad__(p, 1, s), dgrad__(p, 2, s)};
                    dtau[j] = {dtau__(p, s), 0, 0, 0};
                }
                auto r      = source.rho[s].adjoint(source.displacements, drho);
                auto t      = source.tau[s].adjoint(source.displacements, dtau);
                auto matrix = source.op.matrix_elements(r, t);
                for (int b = 0; b < n; b++) {
                    for (int a = 0; a < n; a++) {
                        result.back()(a, b, s) = matrix(a, b);
                    }
                }
            }
            ctx_.comm().allreduce(result.back().at(memory_t::host), result.back().size());
        }
        return result;
    }
};

} // namespace sirius

#endif

/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PAW_XC_FIELD_MAP_HPP__
#define __PAW_XC_FIELD_MAP_HPP__

#include "density/paw_xc_reconstruction.hpp"
#include "function3d/smooth_periodic_function_sampler.hpp"
#include "function3d/xc_point_fields.hpp"

namespace sirius {

/// Smooth potentials in reciprocal space and the additional local PAW matrices.
struct paw_xc_field_adjoint_t
{
    std::vector<Smooth_periodic_function<double>> rho;
    std::vector<Smooth_periodic_function<double>> tau;
    std::vector<mdarray<double, 3>> local_matrices;
};

/// Joint smooth/local PAW field map for one nonlocal XC evaluation.
/** Every target row is owned once across the context communicator. Scalar
 *  inputs denote total fields; collinear inputs are separate up/down fields,
 *  not charge/magnetization components. Smooth rho excludes augmentation Q
 *  and pseudo-core charge. Smooth tau is the positive pseudo-orbital
 *  kinetic density, without Q or core terms. Coefficients must be current.
 *
 *  The local fields are snapshotted from the supplied density matrix. Keep the
 *  geometry fixed between construction, sample() and adjoint(). Both operations
 *  are collective, including empty target ranks. No model weights are applied
 *  to physical fields, and the reverse does not multiply quadrature weights.
 */
class PAW_xc_field_map
{
  private:
    Simulation_context const& ctx_;
    int num_points_;
    PAW_xc_reconstruction local_;
    Smooth_periodic_function_sampler smooth_;

  public:
    PAW_xc_field_map(Density const& density__, std::vector<r3::vector<double>> const& points__)
        : ctx_(density__.ctx())
        , num_points_(static_cast<int>(points__.size()))
        , local_(density__, points__)
        , smooth_(ctx_.gvec(), points__)
    {
    }

    xc_point_fields_t
    sample(std::vector<Smooth_periodic_function<double>> const& rho__,
           std::vector<Smooth_periodic_function<double>> const& tau__) const
    {
        int ns     = ctx_.num_spins();
        int failed = rho__.size() != static_cast<size_t>(ns) || tau__.size() != static_cast<size_t>(ns);
        ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("PAW XC field map: incompatible spin fields on a context rank");
        }
        xc_point_fields_t result(num_points_, ns);
        for (int s = 0; s < ns; s++) {
            auto r = smooth_.sample(rho__[s]);
            auto t = smooth_.sample(tau__[s]);
            for (int p = 0; p < num_points_; p++) {
                result.rho(p, s) = r[p][0];
                result.tau(p, s) = t[p][0];
                for (int x = 0; x < 3; x++) {
                    result.gradient(p, x, s) = r[p][x + 1];
                }
            }
        }
        local_.add_to(result.rho, result.gradient, result.tau);
        return result;
    }

    /// Return the adjoint of the complete reconstruction, not separate XC calls.
    /** Fourier coefficients are normalized as ordinary SIRIUS potentials. The
     *  rho result includes the Cartesian-gradient adjoint. Apply it as a local
     *  potential and cancel its ordinary smooth-Q projector contraction.
     *  local_matrices contains the hard-minus-soft response, without Q or a
     *  frozen-core derivative, in unit_cell().paw_atoms() order.
     *  The tau result must pass through the fine/coarse Fourier adjoint before
     *  the positive-tau orbital operator is applied. No second radial integral
     *  or occupation factor belongs in either returned projector contribution.
     */
    paw_xc_field_adjoint_t
    adjoint(xc_point_fields_t const& derivatives__) const
    {
        int ns        = ctx_.num_spins();
        auto const& r = derivatives__.rho;
        auto const& g = derivatives__.gradient;
        auto const& t = derivatives__.tau;
        int failed    = r.size(0) != num_points_ || r.size(1) != ns || t.size(0) != num_points_ || t.size(1) != ns ||
                     g.size(0) != num_points_ || g.size(1) != 3 || g.size(2) != ns;
        auto finite = [](auto const& a) {
            for (size_t i = 0; i < a.size(); i++) {
                if (!std::isfinite(a[i])) {
                    return false;
                }
            }
            return true;
        };
        failed |= !finite(r) || !finite(g) || !finite(t);
        ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
        if (failed) {
            RTE_THROW("PAW XC field map: invalid point covectors on a context rank");
        }
        paw_xc_field_adjoint_t result;
        result.local_matrices = local_.projector_matrices(r, g, t);
        for (int s = 0; s < ns; s++) {
            result.rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            result.tau.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            std::vector<std::array<double, 4>> dr(num_points_), dt(num_points_);
            for (int p = 0; p < num_points_; p++) {
                dr[p] = {r(p, s), g(p, 0, s), g(p, 1, s), g(p, 2, s)};
                dt[p] = {t(p, s), 0, 0, 0};
            }
            smooth_.adjoint(dr, result.rho.back());
            smooth_.adjoint(dt, result.tau.back());
        }
        return result;
    }
};

} // namespace sirius

#endif

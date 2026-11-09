/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file testing_context.hpp
 *
 *  \brief Create a simulation context with a fake atom type for tests.
 *
 *  Kept separate from testing.hpp, which most tests include, as it requires the full Simulation_context.
 */

#ifndef __TESTING_CONTEXT_HPP__
#define __TESTING_CONTEXT_HPP__

#include <memory>
#include <vector>
#include "testing.hpp"
#include "context/simulation_context.hpp"

namespace sirius {

inline auto
create_simulation_context(nlohmann::json const& conf__, r3::matrix<double> L__, int num_atoms__,
                          std::vector<r3::vector<double>> coord__, bool add_vloc__, bool add_dion__)
{
    auto ctx = std::make_unique<Simulation_context>(conf__);

    ctx->unit_cell().set_lattice_vectors(L__);
    if (num_atoms__) {
        auto& atype = ctx->unit_cell().add_atom_type("Cu");

        if (ctx->cfg().parameters().electronic_structure_method() == "full_potential_lapwlo") {

        } else {
            /* set pseudo charge */
            atype.zn(11);
            /* set radial grid */
            atype.set_radial_grid(radial_grid_t::lin_exp, 1000, 0.0, 100.0, 6);
            /* cutoff at ~1 a.u. */
            int icut    = atype.radial_grid().index_of(1.0);
            double rcut = atype.radial_grid(icut);
            /* create beta radial function */
            std::vector<double> beta(icut + 1);
            std::vector<double> beta1(icut + 1);
            for (int l = 0; l <= 2; l++) {
                for (int i = 0; i <= icut; i++) {
                    double x = atype.radial_grid(i);
                    beta[i]  = confined_polynomial(x, rcut, l, l + 1, 0);
                    beta1[i] = confined_polynomial(x, rcut, l, l + 2, 0);
                }
                /* add radial function for l */
                atype.add_beta_radial_function(angular_momentum(l), beta);
                atype.add_beta_radial_function(angular_momentum(l), beta1);
            }

            std::vector<double> ps_wf(atype.radial_grid().num_points());
            for (int l = 0; l <= 2; l++) {
                for (int i = 0; i < atype.radial_grid().num_points(); i++) {
                    double x = atype.radial_grid(i);
                    ps_wf[i] = std::exp(-x) * std::pow(x, l);
                }
                /* add radial function for l */
                atype.add_ps_atomic_wf(3, angular_momentum(l), ps_wf);
            }

            /* set local part of potential */
            std::vector<double> vloc(atype.radial_grid().num_points(), 0);
            if (add_vloc__) {
                for (int i = 0; i < atype.radial_grid().num_points(); i++) {
                    double x = atype.radial_grid(i);
                    vloc[i]  = -atype.zn() / (std::exp(-x * (x + 1)) + x);
                }
            }
            atype.local_potential(vloc);
            /* set Dion matrix */
            int nbf = atype.num_beta_radial_functions();
            matrix<double> dion({nbf, nbf});
            dion.zero();
            if (add_dion__) {
                for (int i = 0; i < nbf; i++) {
                    dion(i, i) = -10.0;
                }
            }
            atype.d_mtrx_ion(dion);
            /* set atomic density */
            std::vector<double> arho(atype.radial_grid().num_points());
            for (int i = 0; i < atype.radial_grid().num_points(); i++) {
                double x = atype.radial_grid(i);
                arho[i]  = 2 * atype.zn() * std::exp(-x * x) * x;
            }
            atype.ps_total_charge_density(arho);
        }

        for (auto v : coord__) {
            ctx->unit_cell().add_atom("Cu", v, {0, 0, 1});
        }
    }
    ctx->initialize();
    return ctx;
}

} // namespace sirius

#endif

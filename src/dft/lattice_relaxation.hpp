/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file lattice_relaxation.hpp
 *
 *  \brief Lattice relaxation implementation.
 */

#ifndef __LATTICE_RELAXATION_HPP__
#define __LATTICE_RELAXATION_HPP__

#include "dft_ground_state.hpp"
#if defined(SIRIUS_VCSQNM)
#include "vcsqnm/periodic_optimizer.hpp"
#endif

namespace sirius {

class Lattice_relaxation
{
  private:
    DFT_ground_state& dft_;

  public:
    Lattice_relaxation(DFT_ground_state& dft__)
        : dft_{dft__}
    {
    }

    nlohmann::json
    find(int max_num_steps__, double forces_thr__ = 1e-4, double stress_thr__ = -1e-4)
    {
        nlohmann::json result;
#if defined(SIRIUS_VCSQNM)
        bool compute_stress = stress_thr__ > 0;
        bool compute_forces = forces_thr__ > 0;

        int na = dft_.ctx().unit_cell().num_atoms();

        std::unique_ptr<vcsqnm::PES_optimizer::periodic_optimizer> geom_opt;
        Eigen::MatrixXd r(3, na);
        Eigen::MatrixXd f(3, na);
        Eigen::Vector3d lat_a, lat_b, lat_c;
        for (int x = 0; x < 3; x++) {
            lat_a[x] = dft_.ctx().unit_cell().lattice_vector(0)[x];
            lat_b[x] = dft_.ctx().unit_cell().lattice_vector(1)[x];
            lat_c[x] = dft_.ctx().unit_cell().lattice_vector(2)[x];
        }

        Eigen::Matrix3d stress;

        for (int ia = 0; ia < na; ia++) {
            for (auto x : {0, 1, 2}) {
                r(x, ia) = dft_.ctx().unit_cell().atom(ia).position()[x];
            }
        }
        /*
         * @param initial_step_size initial step size. default is 1.0. For systems with hard bonds (e.g. C-C) use a
         * value between and 1.0 and 2.5. If a system only contains weaker bonds a value up to 5.0 may speed up the
         * convergence.
         * @param nhist_max Maximal number of steps that will be stored in the history list. Use a value between 3
         * and 20. Must be <= than 3*nat + 9.
         * @param lattice_weight weight / size of the supercell that is used to transform lattice derivatives. Use a
         * value between 1 and 2. Default is 2.
         * @param alpha0 Lower limit on the step size. 1.e-2 is the default.
         * @param eps_subsp Lower limit on linear dependencies of basis vectors in history list. Default 1.e-4.
         * */
        auto& inp                = dft_.ctx().cfg().vcsqnm();
        double initial_step_size = inp.initial_step_size();
        int nhist_max            = inp.nhist_max();
        double lattice_weight    = inp.lattice_weight();
        double alpha0            = inp.alpha0();
        double eps_subsp         = inp.eps_subsp();
        if (compute_forces && compute_stress) {
            geom_opt = std::make_unique<vcsqnm::PES_optimizer::periodic_optimizer>(
                    na, lat_a, lat_b, lat_c, initial_step_size, nhist_max, lattice_weight, alpha0, eps_subsp);
        } else if (compute_forces) {
            geom_opt = std::make_unique<vcsqnm::PES_optimizer::periodic_optimizer>(na, initial_step_size, nhist_max,
                                                                                   alpha0, eps_subsp);
        }

        bool stress_converged{true};
        bool forces_converged{true};

        for (int istep = 0; istep < max_num_steps__; istep++) {
            RTE_OUT(dft_.ctx().out()) << "optimisation step " << istep + 1 << " out of " << max_num_steps__
                                      << std::endl;

            auto& inp = dft_.ctx().cfg().parameters();
            bool write_state{false};

            /* launch the calculation */
            result = dft_.find(inp.density_tol(), inp.energy_tol(),
                               dft_.ctx().cfg().iterative_solver().energy_tolerance(), inp.num_dft_iter(), write_state);

            rte::ostream out(dft_.ctx().out(), __func__);

            if (compute_stress) {
                dft_.stress().calc_stress_total();
                dft_.stress().print_info(out, dft_.ctx().verbosity());

                auto st = dft_.stress().stress_total();
                double d{0};
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        stress(i, j) = -st(i, j);
                        d += std::abs(st(i, j));
                    }
                }
                if (d < stress_thr__) {
                    stress_converged = true;
                } else {
                    stress_converged = false;
                }

                out << "total stress value: " << d << ", stress threshold: " << stress_thr__
                    << ", converged: " << stress_converged << std::endl;
            }

            if (compute_forces) {
                dft_.forces().calc_forces_total();
                dft_.forces().print_info(out, dft_.ctx().verbosity());

                auto& ft = dft_.forces().forces_total();
                double d{0};
                for (int i = 0; i < dft_.ctx().unit_cell().num_atoms(); i++) {
                    for (int x : {0, 1, 2}) {
                        f(x, i) = ft(x, i);
                        d += std::abs(ft(x, i));
                    }
                }
                if (d < forces_thr__) {
                    forces_converged = true;
                } else {
                    forces_converged = false;
                }
                out << "total forces value: " << d << ", forces threshold: " << forces_thr__
                    << ", converged: " << forces_converged << std::endl;
            }
            auto etot = result["energy"]["total"].get<double>();

            if (forces_converged && stress_converged) {
                out << "lattice relaxation is converged in " << istep << " steps" << std::endl;
                break;
            }

            /*
             * compute new geometry
             */
            if (compute_forces && compute_stress) {
                geom_opt->step(r, etot, f, lat_a, lat_b, lat_c, stress);
            } else if (compute_forces) {
                geom_opt->step(r, etot, f);
            }
            /*
             * update geometry
             */
            auto& ctx = const_cast<Simulation_context&>(dft_.ctx());
            ctx.unit_cell().set_lattice_vectors({lat_a[0], lat_a[1], lat_a[2]}, {lat_b[0], lat_b[1], lat_b[2]},
                                                {lat_c[0], lat_c[1], lat_c[2]});
            for (int ia = 0; ia < na; ia++) {
                ctx.unit_cell().atom(ia).set_position({r(0, ia), r(1, ia), r(2, ia)});
            }
            dft_.update();
            ctx.unit_cell().print_geometry_info(out, ctx.verbosity());
        }
        if (!(forces_converged && stress_converged)) {
            RTE_OUT(dft_.ctx().out()) << "lattice relaxation not converged" << std::endl;
        }
#else
        RTE_THROW("not compiled with vcsqnm support");
#endif
        return result;
    }
};

} // namespace sirius

#endif

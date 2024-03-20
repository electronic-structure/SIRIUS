/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

void
test_lapw_xc(cmd_args const& args__)
{
    auto pw_cutoff = args__.value<double>("pw_cutoff", 12);
    auto N         = args__.value<int>("N", 1);

    /* create simulation context */
    auto json_conf = R"({
      "parameters" : {
        "electronic_structure_method" : "full_potential_lapwlo",
        "use_symmetry" : false
      }
    })"_json;

    Simulation_context ctx(json_conf);

    int lmax{8};
    ctx.lmax_apw(lmax);
    ctx.lmax_pot(lmax);
    ctx.lmax_rho(lmax);
    ctx.add_xc_functional("XC_GGA_X_PW91");
    ctx.add_xc_functional("XC_GGA_C_PW91");

    /* add a new atom type to the unit cell */
    auto& atype = ctx.unit_cell().add_atom_type("Cu");
    /* set pseudo charge */
    atype.zn(29);
    /* set radial grid */
    atype.set_radial_grid(radial_grid_t::lin_exp, 2000, 1e-6, 2.0, 6);
    /* set free atom grid */
    atype.set_free_atom_radial_grid(Radial_grid_lin_exp<double>(5000, 1e-6, 20.0));
    /* set atomic density */
    std::vector<double> atom_rho(atype.free_atom_radial_grid().num_points());
    for (int i = 0; i < atype.free_atom_radial_grid().num_points(); i++) {
        auto x      = atype.free_atom_radial_grid(i);
        atom_rho[i] = 2 * std::sqrt(atype.zn()) * std::exp(-x);
    }
    atype.free_atom_density(atom_rho);

    /* add LAPW descriptors */
    /*   they are not necessary for this test, but atom type would not initialize otherwise */
    for (int l = 0; l <= lmax; l++) {
        atype.add_aw_descriptor(-1, l, 0.15, 0, 0);
        atype.add_aw_descriptor(-1, l, 0.15, 1, 0);
    }

    /* lattice constant */
    double a{5};
    /* set lattice vectors */
    ctx.unit_cell().set_lattice_vectors({{a * N, 0, 0}, {0, a * N, 0}, {0, 0, a * N}});
    /* add atoms */
    double p = 1.0 / N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                ctx.unit_cell().add_atom("Cu", {i * p, j * p, k * p});
            }
        }
    }

    /* initialize the context */
    ctx.verbosity(1);
    ctx.pw_cutoff(pw_cutoff);
    ctx.processing_unit(args__.value<std::string>("device", "CPU"));

    ctx.initialize();

    Density rho(ctx);
    rho.initial_density();

    Potential pot(ctx);
    pot.xc(rho);
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"device=", "(string) CPU or GPU"},
                   {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                   {"N=", "(int) cell multiplicity"}});

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_lapw_xc(args);
    int rank = mpi::Communicator::world().rank();
    sirius::finalize();
    if (!rank) {
        const auto timing_result = global_rtgraph_timer.process();
        std::cout << timing_result.print();
    }
}

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

void
test_davidson(cmd_args const& args__)
{
    auto pw_cutoff    = args__.value<double>("pw_cutoff", 30);
    auto gk_cutoff    = args__.value<double>("gk_cutoff", 10);
    auto N            = args__.value<int>("N", 1);
    auto mpi_grid     = args__.value("mpi_grid", std::vector<int>({1, 1}));
    auto solver       = args__.value<std::string>("solver", "lapack");
    auto xc_name      = args__.value<std::string>("xc_name", "XC_LDA_X");
    auto num_mag_dims = args__.value<int>("num_mag_dims", 0);

    bool add_dion{false};
    bool add_vloc{false};

    PROFILE_START("test_davidson|setup")

    auto json_conf                            = R"({
      "parameters" : {
        "electronic_structure_method" : "pseudopotential",
        "use_symmetry" : true
      }
    })"_json;
    json_conf["parameters"]["xc_functionals"] = {xc_name};
    json_conf["parameters"]["pw_cutoff"]      = pw_cutoff;
    json_conf["parameters"]["gk_cutoff"]      = gk_cutoff;
    json_conf["parameters"]["num_mag_dims"]   = num_mag_dims;
    json_conf["control"]["mpi_grid_dims"]     = mpi_grid;

    double p = 1.0 / N;
    std::vector<r3::vector<double>> coord;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                coord.push_back(r3::vector<double>(i * p, j * p, k * p));
            }
        }
    }

    auto sctx_ptr = sirius::create_simulation_context(json_conf, {{5.0 * N, 0, 0}, {0, 5.0 * N, 0}, {0, 0, 5.0 * N}},
                                                      N * N * N, coord, add_vloc, add_dion);

    auto& ctx = *sctx_ptr;

    PROFILE_STOP("test_davidson|setup")

    Density rho(ctx);
    rho.initial_density();
    rho.print_info(ctx.out());

    check_xc_potential(rho);
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"device=", "(string) CPU or GPU"},
                   {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                   {"gk_cutoff=", "(double) plane-wave cutoff for wave-functions"},
                   {"N=", "(int) cell multiplicity"},
                   {"mpi_grid=", "(int[2]) dimensions of the MPI grid for band diagonalization"},
                   {"solver=", "eigen-value solver"},
                   {"xc_name=", "name of XC potential"},
                   {"num_mag_dims=", "number of magnetic dimensions"}});

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test_davidson(args);
    int rank = mpi::Communicator::world().rank();
    sirius::finalize();
    if (rank == 0) {
        const auto timing_result = global_rtgraph_timer.process();
        std::cout << timing_result.print();
    }
}

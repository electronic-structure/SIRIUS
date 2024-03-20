/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

void
plot(cmd_args& args)
{
    vector3d<double> origin = args.value<vector3d<double>>("origin");

    vector3d<double> b1 = args.value<vector3d<double>>("b1");
    int N1              = args.value<int>("N1");

    bool plot2d{false};
    vector3d<double> b2;
    int N2;
    if (args.exist("b2")) {
        b2     = args.value<vector3d<double>>("b2");
        N2     = args.value<int>("N2");
        plot2d = true;
    }

    std::string coords = args["coordinates"];
    bool cart{false};
    if (coords == "cart") {
        cart = true;
    } else {
        if (coords != "frac") {
            RTE_THROW("wrong type of coordinates");
        }
    }

    sirius::Simulation_context ctx("sirius.json", mpi_comm_world());

    JSON_tree parser("sirius.json");
    sirius::Parameters_input_section inp;
    inp.read(parser);

    ctx.set_esm_type(inp.esm_);
    ctx.set_pw_cutoff(inp.pw_cutoff_);
    ctx.set_aw_cutoff(inp.aw_cutoff_);
    ctx.set_gk_cutoff(inp.gk_cutoff_);
    ctx.lmax_apw(inp.lmax_apw_);
    ctx.lmax_pot(inp.lmax_pot_);
    ctx.lmax_rho(inp.lmax_rho_);
    ctx.set_num_mag_dims(inp.num_mag_dims_);
    ctx.set_auto_rmt(inp.auto_rmt_);

    ctx.initialize();

    sirius::Potential* potential = new sirius::Potential(ctx);
    potential->allocate();

    sirius::Density* density = new sirius::Density(ctx);
    density->allocate();

    density->load();
    // potential->load();

    // density->generate_pw_coefs();

    // density->initial_density();

    if (plot2d) {
        splindex<block> spl_N2(N2, ctx.comm().size(), ctx.comm().rank());

        mdarray<double, 2> rho(N1, N2);

        runtime::Timer t1("compute_density");
        for (int j2 = 0; j2 < spl_N2.local_size(); j2++) {
            int i2 = spl_N2[j2];

            std::cout << "column " << i2 << " out of " << N2 << std::endl;

            #pragma omp parallel for
            for (int i1 = 0; i1 < N1; i1++) {
                vector3d<double> v;
                for (int x : {0, 1, 2}) {
                    v[x] = origin[x] + double(i1) * b1[x] / (N1 - 1) + double(i2) * b2[x] / (N2 - 1);
                }

                if (!cart) {
                    v = ctx.unit_cell().get_cartesian_coordinates(v);
                }

                rho(i1, i2) = density->rho()->value(v);
            }
        }
        t1.stop();

        ctx.comm().allgather(&rho(0, 0), spl_N2.global_offset() * N1, spl_N2.local_size() * N1);

        if (ctx.comm().rank() == 0) {
            sirius::HDF5_tree h5out("rho.h5", true);
            h5out.write("rho", rho);
        }
    } else {
        std::vector<double> rho(N1);
        std::vector<double> r(N1);

        #pragma omp parallel for
        for (int i1 = 0; i1 < N1; i1++) {
            vector3d<double> v;
            for (int x : {0, 1, 2}) {
                v[x] = origin[x] + double(i1) * b1[x] / (N1 - 1);
            }

            if (!cart) {
                v = ctx.unit_cell().get_cartesian_coordinates(v);
            }

            rho[i1] = density->rho()->value(v);
            r[i1]   = v.length();
        }

        FILE* fout = fopen("rho.dat", "w");
        for (int i = 0; i < N1; i++) {
            fprintf(fout, "%18.10f %18.10f\n", r[i], rho[i]);
        }
        fclose(fout);
    }

    delete density;
    delete potential;
}

int
main(int argn, char** argv)
{
    using namespace sirius;

    cmd_args args;
    args.register_key("--origin=", "{vector3d} plot origin");
    args.register_key("--b1=", "{vector3d} 1st boundary vector");
    args.register_key("--b2=", "{vector3d} 2nd boundary vector");
    args.register_key("--N1=", "{int} number of 1st boundary vector divisions");
    args.register_key("--N2=", "{int} number of 2nd boundary vector divisions");
    args.register_key("--coordinates=", "{cart | frac} Cartesian or fractional coordinates");
    args.parse_args(argn, argv);

    if (args.exist("help")) {
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    plot(args);
    sirius::finalize();
}

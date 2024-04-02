/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
test1()
{
    Simulation_context ctx(mpi_comm_world(), "pseudopotential");
    ctx.set_processing_unit("cpu");

    int N = 3;
    double a{4};
    ctx.unit_cell().set_lattice_vectors({{N * a, 0, 0}, {0, N * a, 0}, {0, 0, N * a}});

    ctx.unit_cell().add_atom_type("A");

    auto& atype = ctx.unit_cell().atom_type(0);

    atype.zn(1);
    atype.set_radial_grid(radial_grid_t::lin_exp_grid, 1000, 0, 2);

    std::vector<double> beta(atype.num_mt_points());
    for (int i = 0; i < atype.num_mt_points(); i++) {
        double x = atype.radial_grid(i);
        beta[i]  = std::exp(-x) * (4 - x * x);
    }
    atype.add_beta_radial_function(0, beta);
    atype.add_beta_radial_function(1, beta);
    atype.add_beta_radial_function(2, beta);

    matrix<double> d_mtrx_ion(atype.num_beta_radial_functions(), atype.num_beta_radial_functions());
    d_mtrx_ion.zero();
    for (int i = 0; i < atype.num_beta_radial_functions(); i++) {
        d_mtrx_ion(i, i) = 1;
    }
    atype.d_mtrx_ion(d_mtrx_ion);

    Spline<double> ps_dens(atype.radial_grid());
    for (int i = 0; i < atype.num_mt_points(); i++) {
        double x   = atype.radial_grid(i);
        ps_dens(i) = std::exp(-x * x) * x * x;
    }
    double norm = ps_dens.interpolate().integrate(0);
    ps_dens.scale(1 / norm);
    atype.ps_total_charge_density(ps_dens.values());
    atype.ps_core_charge_density(std::vector<double>(atype.num_mt_points(), 0));

    for (int i1 = 0; i1 < N; i1++) {
        for (int i2 = 0; i2 < N; i2++) {
            for (int i3 = 0; i3 < N; i3++) {
                if (i1 + i2 + i3 == 0) {
                    ctx.unit_cell().add_atom("A", {0.0001, 0, 0});
                } else {
                    ctx.unit_cell().add_atom("A", {1.0 * i1 / N, 1.0 * i2 / N, 1.0 * i3 / N});
                }
            }
        }
    }

    ctx.set_verbosity(3);

    ctx.initialize();

    Density dens(ctx);
    dens.allocate();
    dens.initial_density();

    Potential pot(ctx);
    pot.allocate();
    pot.generate(dens);

    Hamiltonian hmlt(ctx, pot);
    hmlt.prepare<double_complex>();

    double vk[] = {0, 0, 0};
    K_point kp(ctx, vk, 1.0);

    kp.initialize();

    printf("num_gkvec: %i\n", kp.num_gkvec());

    Band band(ctx);
    band.initialize_subspace<double_complex>(&kp, hmlt, 0);

    auto& psi = kp.spinor_wave_functions();
    Wave_functions hpsi(kp.gkvec_partition(), ctx.num_bands());
    Wave_functions spsi(kp.gkvec_partition(), ctx.num_bands());

    ctx.fft_coarse().prepare(kp.gkvec_partition());

    hmlt.apply_h_s<double_complex>(&kp, 0, 0, ctx.num_bands(), psi, hpsi, spsi);

    ctx.fft_coarse().dismiss();

    // for (int k = 0; k < 10; k++) {
    //     kp.beta_projectors().prepare();
    //     for (int ichunk = 0; ichunk < kp.beta_projectors().num_chunks(); ichunk++) {
    //         kp.beta_projectors().generate(ichunk);
    //     }
    //     kp.beta_projectors().dismiss();
    // }
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize();

    test1();

    if (mpi_comm_world().rank() == 0) {
        sddk::timer::print();
    }

    sirius::finalize();
}

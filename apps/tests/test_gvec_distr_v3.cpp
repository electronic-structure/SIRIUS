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
test_gvec_distr(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    Gvec gvec(M, cutoff__, mpi_comm_world(), false);

    MPI_grid mpi_grid({2, 2, 2}, mpi_comm_world());

    Gvec_partition gvp(gvec, mpi_grid.communicator(1 << 1), mpi_grid.communicator(1 << 0 | 1 << 2));

    std::vector<double_complex> fpw(gvec.count());
    for (int igloc = 0; igloc < gvec.count(); igloc++) {
        int ig     = igloc + gvec.offset();
        fpw[igloc] = type_wrapper<double_complex>::random(); // ig;
    }

    std::vector<double_complex> fpw_fft(gvp.gvec_count_fft());

    gvp.gather_pw(fpw.data(), fpw_fft.data());

    double diff{0};
    for (int ig = 0; ig < gvec.count(); ig++) {
        diff += std::abs(fpw[ig] - fpw_fft[gvp.gvec_fft_slab().offsets[gvp.comm_ortho_fft().rank()] + ig]);
    }

    runtime::pstdout pout(mpi_comm_world());

    pout.printf("--- num_gvec: %i ---\n", gvec.num_gvec());
    pout.printf("--- fft rank: %i --- \n", gvp.fft_comm().rank());
    for (int ig = 0; ig < gvp.gvec_count_fft(); ig++) {
        pout.printf("%f\n", fpw_fft[ig].real());
    }
    pout.printf("diff: %f\n", diff);
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.register_key("--cutoff=", "{double} wave-functions cutoff");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto cutoff = args.value<double>("cutoff", 3.0);

    sirius::initialize(1);
    test_gvec_distr(cutoff);
    sirius::finalize();
}

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

    Gvec gvec_coarse(M, cutoff__, mpi_comm_world(), false);

    Gvec gvec1(M, cutoff__ + 2, mpi_comm_world(), false);
    Gvec gvec2(cutoff__ + 2, gvec_coarse);

    runtime::pstdout pout(mpi_comm_world());
    pout.printf("rank: %i\n", mpi_comm_world().rank());
    pout.printf("-------------------------\n");
    pout.printf("num_gvec_coarse     : %i\n", gvec_coarse.num_gvec());
    pout.printf("num_gvec_coarse_loc : %i\n", gvec_coarse.count());
    pout.printf("num_zcols_coarse    : %i\n", gvec_coarse.num_zcol());
    pout.printf(".........................\n");
    pout.printf("num_gvec            : %i\n", gvec1.num_gvec());
    pout.printf("num_gvec_loc        : %i\n", gvec1.count());
    pout.printf("num_zcols           : %i\n", gvec1.num_zcol());
    pout.printf(".........................\n");
    pout.printf("num_gvec            : %i\n", gvec2.num_gvec());
    pout.printf("num_gvec_loc        : %i\n", gvec2.count());
    pout.printf("num_zcols           : %i\n", gvec2.num_zcol());
    pout.flush();

    if (gvec1.num_gvec() != gvec2.num_gvec()) {
        RTE_THROW("wrong number of G-vectors in gvec2");
    }

    for (int igloc = 0; igloc < gvec_coarse.count(); igloc++) {
        int ig = gvec_coarse.offset() + igloc;
        auto G = gvec_coarse.gvec(ig);

        int igloc2 = gvec2.gvec_base_mapping(igloc);
        auto G2    = gvec2.gvec(gvec2.offset() + igloc2);
        if (!(G[0] == G2[0] && G[1] == G2[1] && G[2] == G2[2])) {
            printf("wrong order of G-vectors: %i %i %i vs. %i %i %i\n", G[0], G[1], G[2], G2[0], G2[1], G2[2]);
            printf("ig=%i, ig2=%i\n", ig, gvec2.offset() + igloc2);
        }
    }
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

    auto cutoff = args.value<double>("cutoff", 10.0);

    sirius::initialize(1);
    test_gvec_distr(cutoff);
    sirius::finalize();
}

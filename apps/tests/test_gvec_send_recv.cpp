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
test_gvec_send_recv(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    MPI_grid mpi_grid({2, 2}, mpi_comm_world());
    // MPI_grid mpi_grid({1, 1}, mpi_comm_world());

    Gvec gvec(mpi_grid.communicator(1 << 0));

    auto& comm_k = mpi_grid.communicator(1 << 1);

    if (comm_k.rank() == 0) {
        gvec = Gvec(M, cutoff__, mpi_grid.communicator(1 << 0), false);
    }

    gvec.send_recv(comm_k, 0, 1, gvec);

    std::cout << "num_gvec = " << gvec.num_gvec() << "\n";

    Gvec gvec1(mpi_grid.communicator(1 << 0));

    gvec.send_recv(comm_k, 0, 0, gvec1);
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
    test_gvec_send_recv(cutoff);
    sirius::finalize();
}

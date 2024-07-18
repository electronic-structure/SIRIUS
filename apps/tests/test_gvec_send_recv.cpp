/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include "testing.hpp"

using namespace sirius;

int
test_gvec_send_recv(cmd_args const& args)
{
    auto cutoff = args.value<double>("cutoff", 3.0);

    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    mpi::Grid mpi_grid({2, 2}, mpi::Communicator::world());
    // MPI_grid mpi_grid({1, 1}, mpi_comm_world());

    fft::Gvec gvec(mpi_grid.communicator(1 << 0));

    auto& comm_k = mpi_grid.communicator(1 << 1);

    if (comm_k.rank() == 0) {
        gvec = fft::Gvec(M, cutoff, mpi_grid.communicator(1 << 0), false);
    }

    fft::send_recv(comm_k, gvec, 0, 1);

    std::cout << "num_gvec = " << gvec.num_gvec() << "\n";

    fft::Gvec gvec1(mpi_grid.communicator(1 << 0));

    send_recv(comm_k, gvec1, 0, 0);

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"cutoff=", "{double} wave-functions cutoff"}});

    sirius::initialize(1);
    int result = call_test("test_gvec_send_recv", test_gvec_send_recv, args);
    sirius::finalize();
    return result;
}

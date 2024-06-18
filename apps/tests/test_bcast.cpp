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
test1(cmd_args const& args__)
{
    auto size = args__.value("size", (1 << 20));

    double const mb{1<<20};

    double t = -wtime();
    mdarray<char, 1> buf({size});
    buf.zero();

    /* N of MPI ranks broadcast one by one the buffer of `size` bytes. Each of N-1 ranks recieves a copy
     * of a buffer. Number of bytes moved across the network: N * (N-1) * size */

    int N =  mpi::Communicator::world().size();
    for (int r = 0; r < N; r++) {
        mpi::Communicator::world().bcast(buf.at(memory_t::host), size, r);
    }
    t += wtime();
    if (mpi::Communicator::world().rank() == 0) {
        printf("time  : %f sec.\n", t);
        printf("speed : %f Mb/s,\n", static_cast<uint64_t>(N) * (N - 1) * size / mb / t);
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {
        {"size=", "buffer size in bytes"}});

    sirius::initialize(1);
    call_test("test_bcast_v2", test1, args);
    sirius::finalize();
}

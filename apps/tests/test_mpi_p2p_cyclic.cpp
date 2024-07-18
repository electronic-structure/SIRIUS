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

int
test_p2p_cyclic(cmd_args const& args)
{
    auto size = args.value("size", (1 << 20));

    mdarray<char, 1> buf({size});
    mdarray<char, 1> buf1({size});
    buf.zero();

    int P = mpi::Communicator::world().size();

    /* P MPI ranks send a message of 'size' bytes in a cyclic fashion: 0 -> 1 -> 2 ... -> 0.
     * Total size of data transferred through the network: P * 'size' */

    auto t0 = time_now();
    for (int r = 0; r < P; r++) {
        int rank  = mpi::Communicator::world().rank();
        int rank1 = (rank + 1) % P;
        auto req  = mpi::Communicator::world().isend(buf.at(memory_t::host), size, rank1,
                                                     mpi::Communicator::get_tag(rank, rank1));
        int rank2 = rank - 1;
        if (rank2 < 0) {
            rank2 = P - 1;
        }
        mpi::Communicator::world().recv(buf1.at(memory_t::host), size, rank2, mpi::Communicator::get_tag(rank, rank2));
        req.wait();
    }
    mpi::Communicator::world().barrier();
    double t = time_interval(t0);
    if (mpi::Communicator::world().rank() == 0) {
        printf("time : %f sec.\n", t);
        printf("speed : %f Mb/s/rank\n", size * P / 1024.0 / 1024 / t);
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"size=", "buffer size in bytes"}});

    sirius::initialize(1);
    int result = call_test("test_p2p_cyclic", test_p2p_cyclic, args);
    sirius::finalize();
    return result;
}

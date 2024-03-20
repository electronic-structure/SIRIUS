/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

int
test1(int size)
{
    double t = -wtime();
    mdarray<char, 1> buf({size});
    mdarray<char, 1> buf1({size});
    buf.zero();

    for (int r = 0; r < mpi::Communicator::world().size(); r++) {
        int rank  = mpi::Communicator::world().rank();
        int rank1 = (rank + 1) % mpi::Communicator::world().size();
        auto req  = mpi::Communicator::world().isend(buf.at(memory_t::host), size, rank1,
                                                     mpi::Communicator::get_tag(rank, rank1));
        int rank2 = rank - 1;
        if (rank2 < 0) {
            rank2 = mpi::Communicator::world().size() - 1;
        }
        mpi::Communicator::world().recv(buf1.at(memory_t::host), size, rank2, mpi::Communicator::get_tag(rank, rank2));
        req.wait();
    }
    t += wtime();
    if (mpi::Communicator::world().rank() == 0) {
        printf("time : %f sec.\n", t);
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--size=", "buffer size in bytes");

    args.parse_args(argn, argv);
    auto size = args.value("size", (1 << 20));

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(1);
    test1(size);
    sirius::finalize();
}

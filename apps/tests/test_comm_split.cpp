/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

void
test_comm_split(int comm_size)
{
    if (mpi::Communicator::world().rank() == 0) {
        printf("sub comm size: %i\n", comm_size);
    }

    auto c1 = mpi::Communicator::world().split(mpi::Communicator::world().rank() / comm_size);
    auto c2 = mpi::Communicator::world().split(mpi::Communicator::world().rank() % comm_size);

    auto c3 = c1;
    mpi::Communicator c4(c2);

    mpi::pstdout pout(mpi::Communicator::world());

    pout << "global rank: " << mpi::Communicator::world().rank() << ", c1.rank: " << c1.rank()
         << ", c2.rank: " << c2.rank() << std::endl;
    std::cout << pout.flush(0);
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--comm_size=", "{int} size of sub-communicator");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    int comm_size = args.value<int>("comm_size", 1);

    sirius::initialize(1);
    test_comm_split(comm_size);
    sirius::finalize();
}

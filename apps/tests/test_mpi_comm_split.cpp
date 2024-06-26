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
test_mpi_comm_split(cmd_args const& args__)
{
    int comm_size = args__.value<int>("comm_size", 1);

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

    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"comm_size=", "{int} size of sub-communicator"}});

    sirius::initialize(1);
    call_test("test_mpi_comm_split", test_mpi_comm_split, args);
    sirius::finalize();
}

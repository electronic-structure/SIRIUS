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
test_mpi_grid(cmd_args const& args)
{
    std::vector<int> mpi_grid_dims = args.value("mpi_grid", std::vector<int>({1, 1, 1}));

    mpi::Grid mpi_grid(mpi_grid_dims, mpi::Communicator::world());

    mpi::pstdout pout(mpi::Communicator::world());

    if (mpi::Communicator::world().rank() == 0) {
        pout << "dimensions: " << mpi_grid.communicator(1 << 0).size() << " " << mpi_grid.communicator(1 << 1).size()
             << " " << mpi_grid.communicator(1 << 2).size() << std::endl;
    }

    pout << "rank(flat): " << mpi::Communicator::world().rank()
         << ", coordinate: " << mpi_grid.communicator(1 << 0).rank() << " " << mpi_grid.communicator(1 << 1).rank()
         << " " << mpi_grid.communicator(1 << 2).rank() << ", hostname: " << hostname() << std::endl;
    std::cout << pout.flush(0) << std::endl;
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"mpi_grid=", "{vector3d<int>} MPI grid"}});

    sirius::initialize(1);
    int result = call_test("test_mpi_grid", test_mpi_grid, args);
    sirius::finalize();
    return result;
}

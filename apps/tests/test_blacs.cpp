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
test_blacs()
{
#if defined(SIRIUS_SCALAPACK)
    std::cout << "self_comm_size : " << mpi::Communicator::self().size()
              << ", self_comm_rank : " << mpi::Communicator::self().rank() << std::endl;
    std::cout << "world_comm_size : " << mpi::Communicator::world().size() << ", world_comm_rank "
              << mpi::Communicator::world().rank() << std::endl;

    auto blacs_handler = la::linalg_base::create_blacs_handler(mpi::Communicator::self().native());
    blacs_handler      = la::linalg_base::create_blacs_handler(mpi::Communicator::world().native());
    std::cout << "blacs_handler : " << blacs_handler << std::endl;
#endif
    return 0;
}

int
main(int argn, char** argv)
{
    sirius::initialize(true);
    int result = call_test("test_blacs", test_blacs);
    sirius::finalize(true);
    return result;
}

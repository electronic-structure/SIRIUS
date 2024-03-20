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
main(int argn, char** argv)
{
    sirius::initialize(true);

#if defined(SIRIUS_SCALAPACK)
    std::cout << mpi::Communicator::self().size() << " " << mpi::Communicator::self().rank() << std::endl;
    std::cout << mpi::Communicator::world().size() << " " << mpi::Communicator::world().rank() << std::endl;

    auto blacs_handler = la::linalg_base::create_blacs_handler(mpi::Communicator::self().native());
    blacs_handler      = la::linalg_base::create_blacs_handler(mpi::Communicator::world().native());
    std::cout << blacs_handler << std::endl;

    sirius::finalize(true);
#endif
    return 0;
}

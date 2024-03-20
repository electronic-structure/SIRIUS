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
    sirius::initialize(1);
    mpi::pstdout pout(mpi::Communicator::world());
    pout << "Hello from rank : " << mpi::Communicator::world().rank() << std::endl;

    /* this is a collective operation */
    auto s = pout.get().str();

    if (mpi::Communicator::world().rank() == 0) {
        std::cout << s;
    }

    sirius::finalize();

    return 0;
}

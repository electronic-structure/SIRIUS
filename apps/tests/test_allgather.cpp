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
test_allgather()
{
    int N = 11;
    std::vector<double> vec(N, 0.0);

    splindex_block<> spl(N, n_blocks(mpi::Communicator::world().size()), block_id(mpi::Communicator::world().rank()));

    for (int i = 0; i < spl.local_size(); i++) {
        vec[spl.global_index(i)] = mpi::Communicator::world().rank() + 1.0;
    }

    {
        mpi::pstdout pout(mpi::Communicator::world());
        if (mpi::Communicator::world().rank() == 0) {
            pout << "before" << std::endl;
        }
        pout << "rank : " << mpi::Communicator::world().rank() << " array : ";
        for (int i = 0; i < N; i++) {
            pout << vec[i];
        }
        pout << std::endl;
        std::cout << pout.flush(0);

        mpi::Communicator::world().allgather(&vec[0], spl.local_size(), spl.global_offset());

        if (mpi::Communicator::world().rank() == 0) {
            pout << "after" << std::endl;
        }
        pout << "rank : " << mpi::Communicator::world().rank() << " array : ";
        for (int i = 0; i < N; i++) {
            pout << vec[i];
        }
        pout << std::endl;
        std::cout << pout.flush(0);
    }
    mpi::Communicator::world().barrier();
}

int
main(int argn, char** argv)
{
    sirius::initialize(true);

    test_allgather();

    sirius::finalize();
}

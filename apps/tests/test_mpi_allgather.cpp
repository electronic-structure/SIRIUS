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
test_allgather()
{
    int mb{1 << 20};
    int N = 1 * mb;
    std::vector<double> vec(N, 0.0);

    splindex_block<> spl(N, n_blocks(mpi::Communicator::world().size()), block_id(mpi::Communicator::world().rank()));

    for (int i = 0; i < spl.local_size(); i++) {
        vec[spl.global_index(i)] = mpi::Communicator::world().rank() + 1.0;
    }

    /* Gather scattered buffer. Ff P is the number of MPI ranks and N is the buffer size, then each MPI
     * ranks stores N / P elements of the buffer. Each rank sends N/P elements to all other ranks and
     * receives (P-1) * N/P elements. Total data exchange is P * N */

    auto t0 = time_now();
    mpi::Communicator::world().allgather(&vec[0], spl.local_size(), spl.global_offset());
    auto t = time_interval(t0);

    int err{0};
    for (int i = 0; i < N; i++) {
        auto loc = spl.location(i);
        if (vec[i] != loc.ib + 1.0) {
            err = 1;
        }
    }

    int P = mpi::Communicator::world().size();
    if (mpi::Communicator::world().rank() == 0) {
        printf("time  : %f sec.\n", t);
        printf("speed : %f Mb/s,\n", static_cast<uint64_t>(P) * N * sizeof(double) / t / mb);
    }

    return err;
}

int
main(int argn, char** argv)
{
    sirius::initialize(true);

    call_test("test_allgather", test_allgather);

    sirius::finalize();
}

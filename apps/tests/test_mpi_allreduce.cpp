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
test_allreduce()
{
    int mb{1 << 20};
    int N = 100 * mb;
    std::vector<int> vec(N, 0);

    for (int i = 0; i < N; i++) {
        vec[i] = mpi::Communicator::world().rank();
    }

    /* Execute MPI reductions; For P MPI ranks and N elements, the total network traffic is P * (P - 1) * N
     * elements. */
    auto t0 = time_now();
    mpi::Communicator::world().allreduce(&vec[0], N);
    auto t = time_interval(t0);

    int P = mpi::Communicator::world().size();

    int err{0};
    for (int i = 0; i < N; i++) {
        if (vec[i] != P * (P - 1) / 2) {
            err = 1;
        }
    }

    if (mpi::Communicator::world().rank() == 0) {
        printf("time  : %f sec.\n", t);
        printf("speed : %f Mb/s/rank\n", static_cast<uint64_t>(P) * N * sizeof(int) / t / mb);
    }

    return err;
}

int
main(int argn, char** argv)
{
    sirius::initialize(true);

    call_test("test_allreduce", test_allreduce);

    sirius::finalize();
}

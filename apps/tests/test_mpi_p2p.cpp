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
test_p2p()
{
    auto& comm = mpi::Communicator::world();

    int N = 2 * (1 << 20);
    std::vector<double> a(N, 1234);

    comm.barrier();
    auto t0 = time_now();
    if (comm.rank() == 0) {
        comm.isend(&a[0], N, 1, 13);
    } else {
        comm.recv(&a[0], N, 0, 13);
    }
    comm.barrier();
    double t = time_interval(t0);

    double sz = N * sizeof(double) / double(1 << 20);

    printf("size: %.4f MB, time: %.4f (us), transfer speed: %.4f MB/s\n", sz, t * 1e6, sz / t);
    return 0;
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    int result = call_test("test_p2p", test_p2p);
    sirius::finalize();
    return result;
}

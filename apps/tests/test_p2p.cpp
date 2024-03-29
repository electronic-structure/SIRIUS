/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
test_p2p()
{
    Communicator comm(MPI_COMM_WORLD);

    int N = 2 * (1 << 20);
    std::vector<double> a(N, 1234);

    comm.barrier();
    Timer t1("comm:p2p");
    if (comm.rank() == 0) {
        comm.isend(&a[0], N, 1, 13);
    } else {
        comm.recv(&a[0], N, 0, 13);
    }
    comm.barrier();
    double tval = t1.stop();

    double sz = N * sizeof(double) / double(1 << 20);

    printf("size: %.4f MB, time: %.4f (us), transfer speed: %.4f MB/sec\n", sz, tval * 1e6, sz / tval);
}

int
main(int argn, char** argv)
{
    Platform::initialize(1);
    test_p2p();
    Platform::finalize();
}

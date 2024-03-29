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
test()
{
    int size = (1 << 22);
    int nt   = 1;
    mdarray<int, 2> buff(size, nt);

    Communicator comm(MPI_COMM_WORLD);

    Timer t("bcast");
    #pragma omp parallel num_threads(nt)
    {
        int thread_id = Platform::thread_id();
        for (int i = 0; i < comm.size(); i++) {
            comm.bcast(&buff(0, thread_id), size, i);
        }
    }
    double tval = t.stop();

    if (comm.rank() == 0) {
        printf("bcast sepeed : %f [MB / sec]\n", double(size * sizeof(int) * comm.size()) / tval / (1 << 20));
    }
}

int
main(int argn, char** argv)
{
    Platform::initialize(1);
    test();
    Platform::finalize();
}

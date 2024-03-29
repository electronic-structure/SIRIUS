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
test1()
{
    int N         = 1234;
    int num_gkvec = 300000;

    splindex<block> spl_ngk(num_gkvec, Platform::comm_world().size(), Platform::comm_world().rank());

    matrix<double_complex> A(spl_ngk.local_size(), N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < (int)spl_ngk.local_size(); j++)
            A(j, i) = 1.0 / (i + j + 1);
    }
    matrix<double_complex> C(N, N);
    C.zero();

    Timer t("allreduce");
    for (int i = 0; i < 10; i++) {
        blas<cpu>::gemm(2, 0, N, N, (int)spl_ngk.local_size(), &A(0, 0), A.ld(), &A(0, 0), A.ld(), &C(0, 0), C.ld());
        Platform::comm_world().allreduce(&C(0, 0), N * N);
    }
    double tval = t.stop();

    if (Platform::comm_world().rank() == 0) {
        printf("average time: %12.6f\n", tval / 10);
        printf("performance (GFlops) : %12.6f\n", 10 * 8e-9 * N * N * (int)spl_ngk.local_size() / tval);
    }
}

int
main(int argn, char** argv)
{
    Platform::initialize(1);
    test1();
    Platform::finalize();
}

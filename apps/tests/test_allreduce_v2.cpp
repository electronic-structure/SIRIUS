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
allreduce_func(mdarray<double_complex, 2>& buf)
{
    mpi_comm_world().barrier();
    mpi_comm_world().allreduce(buf.at<CPU>(), static_cast<int>(buf.size()));
    mpi_comm_world().barrier();
}

void
test1()
{
    int N  = 2048;
    int BS = 256;
    int n  = static_cast<int>(1.0 * N / BS + 1e-10);

    if (mpi_comm_world().rank() == 0) {
        printf("BS=%i, N=%i, n=%i\n", BS, N, n);
    }

    mdarray<double_complex, 2> buf(BS, BS);
    buf.zero();
    allreduce_func(buf);

    for (int repeat = 0; repeat < 30; repeat++) {
        sddk::timer t1("allreduce_func");
        for (int i = 0; i < n * n; i++) {
            allreduce_func(buf);
        }
        double time = t1.stop();

        double perf = 8e-9 * N * N * (N * 1000) / time / mpi_comm_world().size();
        if (mpi_comm_world().rank() == 0) {
            printf("predicted performance : %12.6f (GFlops / rank)\n", perf);
        }
    }

    // int N = 1234;
    // int num_gkvec = 300000;

    // splindex<block> spl_ngk(num_gkvec, Platform::comm_world().size(), Platform::comm_world().rank());

    // matrix<double_complex> A(spl_ngk.local_size(), N);
    // for (int i = 0; i < N; i++)
    //{
    //     for (int j = 0; j < (int)spl_ngk.local_size(); j++) A(j, i) = 1.0 / (i + j + 1);
    // }
    // matrix<double_complex> C(N, N);
    // C.zero();

    // Timer t("allreduce");
    // for (int i = 0; i < 10; i++)
    //{
    //     blas<cpu>::gemm(2, 0, N, N, (int)spl_ngk.local_size(), &A(0, 0), A.ld(), &A(0, 0), A.ld(), &C(0, 0), C.ld());
    //     Platform::comm_world().allreduce(&C(0, 0), N * N);
    // }
    // double tval = t.stop();

    // if (Platform::comm_world().rank() == 0)
    //{
    //     printf("average time: %12.6f\n", tval / 10);
    //     printf("performance (GFlops) : %12.6f\n", 10 * 8e-9 * N * N * (int)spl_ngk.local_size() / tval);
    // }
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    test1();
    sddk::timer::print();
    sirius::finalize();
}

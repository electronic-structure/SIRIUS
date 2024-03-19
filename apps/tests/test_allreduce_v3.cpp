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
    mpi_comm_world().allreduce(buf.at<CPU>(), static_cast<int>(buf.size()));
}

void
test1(int BS, int n, int nt)
{
    int N = BS * n;

    if (mpi_comm_world().rank() == 0) {
        printf("BS=%i, N=%i, n=%i\n", BS, N, n);
    }

    mdarray<double_complex, 3> buf(BS, BS, nt);
    buf.zero();

    // omp_set_nested(1);

    mpi_comm_world().barrier();
    sddk::timer t1("allreduce");

    if (mpi_comm_world().rank() == 0) {
        printf("spawning %i thread\n", nt);
    }

    std::vector<Communicator> c;
    for (int i = 0; i < nt; i++) {
        c.push_back(std::move(mpi_comm_world().duplicate()));
    }

    #pragma omp parallel num_threads(nt)
    {
        int tid    = omp_get_thread_num();
        auto& comm = c[tid];
        int s{0};
        for (int ibc = 0; ibc < n; ibc++) {
            for (int ibr = 0; ibr < n; ibr++) {
                if (s % nt == tid) {
                    comm.allreduce(buf.at<CPU>(0, 0, tid), BS * BS);
                }
                s++;
            }
        }
    }
    mpi_comm_world().barrier();
    double tval = t1.stop();
    if (mpi_comm_world().rank() == 0) {
        printf("time: %12.6f sec.\n", tval);
    }

    // double time = t1.stop();

    // double perf = 8e-9 * N * N * (N * 1000) / time / mpi_comm_world().size();
    // if (mpi_comm_world().rank() == 0) {
    //     printf("predicted performance : %12.6f (GFlops / rank)\n", perf);
    // }

    // omp_set_nested(0);

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
    cmd_args args;
    args.register_key("--BS=", "{int} block size of the panel");
    args.register_key("--n=", "{int} matrix size multiplier");
    args.register_key("--nt=", "{int} number of threads for independent MPI_Allreduce");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    int BS = args.value<int>("BS", 256);
    int n  = args.value<int>("n", 4);
    int nt = args.value<int>("nt", 1);

    sirius::initialize(1);
    for (int repeat = 0; repeat < 10; repeat++) {
        test1(BS, n, nt);
    }
    sddk::timer::print();
    sirius::finalize();
}

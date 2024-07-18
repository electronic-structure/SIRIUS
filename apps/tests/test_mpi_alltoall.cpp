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

double
test_mpi_alltoall_impl(int M__, int N__)
{
    auto& comm = mpi::Communicator::world();

    splindex_block spl_M(M__, n_blocks(comm.size()), block_id(comm.rank()));
    splindex_block spl_N(N__, n_blocks(comm.size()), block_id(comm.rank()));

    matrix<std::complex<double>> a({spl_M.local_size(), N__});
    a = [](int64_t i, int64_t j) { return random<std::complex<double>>(); };

    matrix<std::complex<double>> b({M__, spl_N.local_size()});
    b.zero();

    auto h = a.hash();

    mpi::block_data_descriptor sd(comm.size());
    mpi::block_data_descriptor rd(comm.size());

    for (int rank = 0; rank < comm.size(); rank++) {
        sd.counts[rank] = spl_M.local_size() * spl_N.local_size(block_id(rank));
        rd.counts[rank] = spl_M.local_size(block_id(rank)) * spl_N.local_size();
    }
    sd.calc_offsets();
    rd.calc_offsets();

    if (comm.rank() == 0) {
        printf("number of ranks: %i\n", comm.size());
        printf("local buffer size: %f Mb\n", spl_M.local_size() * N__ * sizeof(std::complex<double>) / double(1 << 20));
    }

    /* P MPI ranks reshuffle data. Each rank stores N * M / P elements which it sends and receives.
     * Total ammount of data sent and recieved by each MPI ranks is 2 * M * N / P */

    comm.barrier();
    auto t0 = time_now();
    comm.alltoall(&a(0, 0), &sd.counts[0], &sd.offsets[0], &b(0, 0), &rd.counts[0], &rd.offsets[0]);
    double t = time_interval(t0);

    comm.alltoall(&b(0, 0), &rd.counts[0], &rd.offsets[0], &a(0, 0), &sd.counts[0], &sd.offsets[0]);

    if (a.hash() != h) {
        RTE_THROW("wrong hash\n");
    }

    return (spl_M.local_size() * N__ + M__ * spl_N.local_size()) * sizeof(std::complex<double>) / t / (1 << 30);
}

int
test_mpi_alltoall(cmd_args const& args__)
{
    int M      = args__.value<int>("M", 1000);
    int N      = args__.value<int>("N", 100);
    int repeat = args__.value<int>("repeat", 1);
    sirius::Measurement perf;
    for (int i = 0; i < repeat; i++) {
        perf.push_back(test_mpi_alltoall_impl(M, N));
    }
    if (mpi::Communicator::world().rank() == 0) {
        printf("average performance: %12.4f GB/s/rank\n", perf.average());
        printf("sigma: %12.4f GB/s/rank\n", perf.sigma());
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"N=", "{int} number of matrix rows"},
                   {"M=", "{int} number of matrix columns"},
                   {"repeat=", "{int} repeat test number of times"}});

    sirius::initialize(1);
    int result = call_test("test_mpi_alltoall", test_mpi_alltoall, args);
    sirius::finalize();
    return result;
}

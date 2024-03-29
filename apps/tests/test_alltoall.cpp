/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

double
test_alltoall(int num_gkvec__, int num_bands__, int num_groups__)
{
    int np = mpi_comm_world().size() / num_groups__;
    if (np * num_groups__ != mpi_comm_world().size()) {
        RTE_THROW("wrong number of MPI ranks");
    }

    int num_bands = num_bands__ / num_groups__;
    if (num_bands * num_groups__ != num_bands__) {
        RTE_THROW("wrong number of bands");
    }

    std::vector<int> mpi_grid_dims = {np, num_groups__};
    MPI_grid mpi_grid(mpi_grid_dims, mpi_comm_world());

    auto& comm = mpi_grid.communicator(1 << 0);

    splindex<block> spl_gkvec(num_gkvec__, comm.size(), comm.rank());
    splindex<block> spl_bands(num_bands, comm.size(), comm.rank());

    matrix<double_complex> a(spl_gkvec.local_size(), num_bands);
    matrix<double_complex> b(num_gkvec__, spl_bands.local_size());

    for (int i = 0; i < num_bands; i++) {
        for (int j = 0; j < spl_gkvec.local_size(); j++)
            a(j, i) = type_wrapper<double_complex>::random();
    }
    b.zero();

    auto h = a.hash();

    block_data_descriptor sd(comm.size());
    block_data_descriptor rd(comm.size());

    for (int rank = 0; rank < comm.size(); rank++) {
        sd.counts[rank] = spl_gkvec.local_size() * spl_bands.local_size(rank);
        rd.counts[rank] = spl_gkvec.local_size(rank) * spl_bands.local_size();
    }
    sd.calc_offsets();
    rd.calc_offsets();

    if (comm.rank() == 0) {
        printf("number of ranks: %i\n", comm.size());
        printf("local buffer size: %f Mb\n",
               spl_gkvec.local_size() * num_bands * sizeof(double_complex) / double(1 << 20));
    }

    comm.barrier();
    runtime::Timer t("alltoall");
    comm.alltoall(&a(0, 0), &sd.counts[0], &sd.offsets[0], &b(0, 0), &rd.counts[0], &rd.offsets[0]);
    double tval = t.stop();

    comm.alltoall(&b(0, 0), &rd.counts[0], &rd.offsets[0], &a(0, 0), &sd.counts[0], &sd.offsets[0]);

    if (a.hash() != h)
        printf("wrong hash\n");

    double perf = num_gkvec__ * num_bands__ * sizeof(double_complex) / tval / (1 << 30);

    return perf;
}

// void test_alltoall_v2()
//{
//     Communicator comm(MPI_COMM_WORLD);
//
//     std::vector<int> counts_in(comm.size(), 16);
//     std::vector<double> sbuf(16);
//
//     std::vector<int> counts_out(comm.size(), 0);
//     counts_out[0] = 16 * comm.size();
//
//     std::vector<double> rbuf(counts_out[comm.rank()]);
//
//     auto a2a_desc = comm.map_alltoall(counts_in, counts_out);
//     comm.alltoall(&sbuf[0], &a2a_desc.sendcounts[0], &a2a_desc.sdispls[0], &rbuf[0], &a2a_desc.recvcounts[0],
//     &a2a_desc.rdispls[0]);
//
//     PRINT("test_alltoall_v2 done");
//     comm.barrier();
// }

// void test_alltoall_v3()
//{
//     Communicator comm(MPI_COMM_WORLD);
//
//     std::vector<int> counts_in(comm.size(), 16);
//     std::vector<double_complex> sbuf(16);
//
//     std::vector<int> counts_out(comm.size(), 0);
//     if (comm.size() == 1)
//     {
//         counts_out[0] = 16 * comm.size();
//     }
//     else
//     {
//         counts_out[0] = 8 * comm.size();
//         counts_out[1] = 8 * comm.size();
//     }
//
//     std::vector<double_complex> rbuf(counts_out[comm.rank()]);
//
//     auto a2a_desc = comm.map_alltoall(counts_in, counts_out);
//     comm.alltoall(&sbuf[0], &a2a_desc.sendcounts[0], &a2a_desc.sdispls[0], &rbuf[0], &a2a_desc.recvcounts[0],
//     &a2a_desc.rdispls[0]);
//
//     PRINT("test_alltoall_v3 done");
//     comm.barrier();
// }

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--num_gkvec=", "{int} number of Gk-vectors");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--repeat=", "{int} repeat test number of times");
    args.register_key("--num_groups=", "{int} number of MPI groups");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    int num_gkvec  = args.value<int>("num_gkvec");
    int num_bands  = args.value<int>("num_bands");
    int repeat     = args.value<int>("repeat", 1);
    int num_groups = args.value<int>("num_groups", 1);

    sirius::initialize(1);

    std::vector<double> perf(repeat);
    double avg = 0;
    for (int i = 0; i < repeat; i++) {
        perf[i] = test_alltoall(num_gkvec, num_bands, num_groups);
        avg += perf[i];
    }
    avg /= repeat;
    double variance = 0;
    for (int i = 0; i < repeat; i++)
        variance += std::pow(perf[i] - avg, 2);
    variance /= repeat;
    double sigma = std::sqrt(variance);
    if (mpi_comm_world().rank() == 0) {
        printf("average performance: %12.4f GB/sec.\n", avg);
        printf("sigma: %12.4f GB/sec.\n", sigma);
    }

    // test_alltoall_v2();
    // test_alltoall_v3();

    sirius::finalize();
}

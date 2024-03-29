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
test_redistr(std::vector<int> mpi_grid_dims, int M, int N, int bs)
{
    if (mpi_grid_dims.size() != 2) {
        RTE_THROW("2d MPI grid is expected");
    }

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims[0], mpi_grid_dims[1]);

    dmatrix<double> mtrx(M, N, blacs_grid, bs, bs);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mtrx.set(j, i, double((j + 1) * (i + 1)));
        }
    }

    splindex<block> spl_row(M, mpi_comm_world().size(), mpi_comm_world().rank());
    matrix_storage<double, matrix_storage_t::slab> mtrx2(spl_row.local_size(), N, CPU);

    mtrx2.remap_from(mtrx, 0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < spl_row.local_size(); j++) {
            int jglob = spl_row[j];
            if (std::abs(mtrx2.prime(j, i) - double((jglob + 1) * (i + 1))) > 1e-14) {
                RTE_THROW("error");
            }
        }
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--M=", "{int} global number of matrix rows");
    args.register_key("--N=", "{int} global number of matrix columns");
    args.register_key("--bs=", "{int} block size");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto M             = args.value<int>("M", 10000);
    auto N             = args.value<int>("N", 1000);
    auto bs            = args.value<int>("bs", 16);

    sirius::initialize(1);
    test_redistr(mpi_grid_dims, M, N, bs);
    runtime::Timer::print();
    sirius::finalize();
}

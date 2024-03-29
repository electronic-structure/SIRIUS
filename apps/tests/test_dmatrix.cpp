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
test_dmatrix(std::vector<int> mpi_grid_dims__)
{
    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims__[0], mpi_grid_dims__[1]);

    int bs{16};
    int N{4};

    dmatrix<double_complex> mtrx(4, 4, blacs_grid, bs, bs);
    mtrx.zero();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mtrx.set(j, i, type_wrapper<double_complex>::random());
        }
    }

    if (!Utils::check_hermitian(mtrx, N)) {
        printf("wrong: matrix is not hermitian\n");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            mtrx.set(j, i, double_complex(i + 1, j + 1));
        }
    }
    mtrx.serialize("mtrx", N);
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});

    sirius::initialize(1);
    test_dmatrix(mpi_grid_dims);
    sirius::finalize();

    return 0;
}

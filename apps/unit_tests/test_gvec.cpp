/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

/* test G-vectors */

using namespace sirius;

int
test_gvec(cmd_args& args)
{
    auto vd = args.value("dims", std::vector<int>({132, 132, 132}));
    r3::vector<int> dims(vd[0], vd[1], vd[2]);
    double cutoff = args.value<double>("cutoff", 50);

    r3::matrix<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    M(0, 1)                     = 0.1;
    M(0, 2)                     = 0.2;
    M(2, 0)                     = 0.3;

    fft::Gvec gvec(M, cutoff, mpi::Communicator::world(), false);
    fft::Gvec gvec_r(M, cutoff, mpi::Communicator::world(), true);

    if (gvec_r.num_gvec() * 2 != gvec.num_gvec() + 1) {
        return 1;
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"dims=", "{vector<int>} FFT dimensions"}, {"cutoff=", "{double} cutoff radius in G-space"}});

    sirius::initialize(true);
    int result = call_test(argv[0], test_gvec, args);
    sirius::finalize();
    return result;
}

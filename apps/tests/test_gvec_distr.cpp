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

int
test_gvec_distr(cmd_args const& args__)
{
    auto cutoff = args__.value<double>("cutoff", 10.0);

    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    fft::Gvec gvec_coarse(M, cutoff, mpi::Communicator::world(), false);

    fft::Gvec gvec1(M, cutoff + 2, mpi::Communicator::world(), false);
    fft::Gvec gvec2(cutoff + 2, gvec_coarse);

    mpi::pstdout pout(mpi::Communicator::world());
    pout << "rank: " << mpi::Communicator::world().rank() << std::endl;
    pout << "-------------------------\n";
    pout << "num_gvec_coarse     : " << gvec_coarse.num_gvec() << "\n";
    pout << "num_gvec_coarse_loc : " << gvec_coarse.count() << "\n";
    pout << "num_zcols_coarse    : " << gvec_coarse.num_zcol() << "\n";
    pout << ".........................\n";
    pout << "num_gvec            : " << gvec1.num_gvec() << "\n";
    pout << "num_gvec_loc        : " << gvec1.count() << "\n";
    pout << "num_zcols           : " << gvec1.num_zcol() << "\n";
    pout << ".........................\n";
    pout << "num_gvec            : " << gvec2.num_gvec() << "\n";
    pout << "num_gvec_loc        : " << gvec2.count() << "\n";
    pout << "num_zcols           : " << gvec2.num_zcol() << "\n";

    std::cout << pout.flush(0);

    if (gvec1.num_gvec() != gvec2.num_gvec()) {
        RTE_THROW("wrong number of G-vectors in gvec2");
    }

    int ierr{0};
    for (int igloc = 0; igloc < gvec_coarse.count(); igloc++) {
        int ig = gvec_coarse.offset() + igloc;
        auto G = gvec_coarse.gvec(gvec_index_t::global(ig));

        int igloc2 = gvec2.gvec_base_mapping(igloc);
        auto G2    = gvec2.gvec(gvec_index_t::global(gvec2.offset() + igloc2));
        if (!(G[0] == G2[0] && G[1] == G2[1] && G[2] == G2[2])) {
            printf("wrong order of G-vectors: %i %i %i vs. %i %i %i\n", G[0], G[1], G[2], G2[0], G2[1], G2[2]);
            printf("ig=%i, ig2=%i\n", ig, gvec2.offset() + igloc2);
            ierr = 1;
        }
    }
    return ierr;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"cutoff=", "{double} wave-functions cutoff"}});

    sirius::initialize(1);
    int result = call_test("test_gvec_distr", test_gvec_distr, args);
    sirius::finalize();
    return result;
}

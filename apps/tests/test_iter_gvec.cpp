/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "core/fft/gvec.hpp"
#include "sirius.hpp"

using namespace sirius;

int
main(int argn, char** argv)
{
    sirius::initialize();

    r3::matrix<double> M({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    fft::Gvec gv(M, 10, mpi::Communicator::world(), false);

    #pragma omp parallel for
    for (auto it : gv) {
        std::cout << it.ig << " " << it.igloc << std::endl;
    }

    sirius::finalize();
    return 0;
}

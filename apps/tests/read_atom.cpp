/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

void
read_atom(std::string fname)
{
    Simulation_parameters params;
    params.electronic_structure_method("pseudopotential");

    Atom_type atype(params, 0, "", fname);
    atype.init();
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    read_atom(argv[1]);
    sirius::finalize();
}

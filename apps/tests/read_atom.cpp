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
read_atom(cmd_args const& args__)
{
    Simulation_parameters params;
    params.electronic_structure_method(args__.value<std::string>("method"));

    Atom_type atype(params, 0, "X", args__.value<std::string>("file"));
    atype.init();

    if (params.full_potential()) {
        Atom_symmetry_class a1(0, atype);
        std::vector<double> veff(atype.radial_grid().num_points());
        for (int i = 0; i < atype.radial_grid().num_points(); i++) {
            veff[i] = -atype.zn() / atype.radial_grid().x(i);
        }
        a1.set_spherical_potential(veff);

        a1.generate_radial_functions(relativity_t::none);
        a1.check_lo_linear_independence(1e-5);
    }
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"file=", "(string) atomic file name"}, {"method=", "(string) electronic structure method"}});
    sirius::initialize(1);
    read_atom(args);
    sirius::finalize();
}

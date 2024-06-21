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
read_atom(cmd_args const& args__)
{
    if (!args__.exist("method")) {
        std::cout << "electronic structure method is not specified\n" << std::endl;
        return 0;
    }
    if (!args__.exist("file")) {
        std::cout << "file name is not provided\n" << std::endl;
        return 0;
    }
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
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"file=", "(string) atomic file name"}, {"method=", "(string) electronic structure method"}});
    sirius::initialize(1);
    int result = call_test("read_atom", read_atom, args);
    sirius::finalize();
    return result;
}

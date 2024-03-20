/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

void
test1()
{
    /*if (mpi_comm_world().rank() == 0) {
        sirius::HDF5_tree f("f.h5", true);
        std::vector<double> buf(100, 0);
        f.create_node("aaa");
        f["aaa"].write("vec", buf);
    */
    mpi_comm_world().barrier();
    sirius::HDF5_tree f("f.h5", false);
}

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    test1();
    sirius::finalize();
}

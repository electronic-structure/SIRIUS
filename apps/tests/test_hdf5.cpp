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
test1()
{
    sirius::HDF5_tree f("f1.h5", sirius::hdf5_access_t::truncate);

    mdarray<double, 2> dat({2, 4});
    dat.zero();

    dat(0, 0) = 1.1;
    dat(0, 1) = 2.2;
    dat(0, 2) = 3.3;
    dat(1, 0) = 4.4;
    dat(1, 1) = 5.5;
    dat(1, 2) = 6.6;

    std::cout << "hash  = " << dat.hash() << std::endl;

    f.create_node("aaa");
    f["aaa"].write("dat_name", dat);
    dat.zero();
    f["aaa"].read("dat_name", dat);
    std::cout << "hash  = " << dat.hash() << std::endl;

    f.write("dat_name", dat);
}

void
test2()
{
    sirius::HDF5_tree f("f2.h5", sirius::hdf5_access_t::truncate);
    f.create_node("node1");

    mdarray<double, 2> md1({2, 4});
    md1.zero();
    f["node1"].write("md1", md1);
    f["node1"].write(0, md1);

    mdarray<std::complex<double>, 2> md2({2, 4});
    md2.zero();
    f["node1"].write("md2", md2);
    f["node1"].write(1, md2);

    mdarray<int, 2> md3({2, 4});
    md3.zero();
    f["node1"].write("md3", md3);
    f["node1"].write(2, md3);
}

void
test3()
{
    sirius::HDF5_tree f("f2.h5", sirius::hdf5_access_t::read_only);

    mdarray<double, 2> md1({2, 4});
    f["node1"].read("md1", md1);
    f["node1"].read(0, md1);

    mdarray<std::complex<double>, 2> md2({2, 4});
    md2.zero();
    f["node1"].read("md2", md2);
    f["node1"].read(1, md2);

    mdarray<int, 2> md3({2, 4});
    md3.zero();
    f["node1"].read("md3", md3);
    f["node1"].read(2, md3);
}

void
test4()
{
    using namespace std::string_literals;
    sirius::HDF5_tree f("qe.h5", sirius::hdf5_access_t::truncate);

    mdarray<int, 2> miller({2, 3}); // Dataset
    miller.zero();

    mdarray<double, 2> evc({3, 4}); // Dataset
    evc.zero();

    std::vector<double> xk = {0.00, 0.13, 0.10}; // Group '/' attribute

    f.write_attribute("gamma_only", ".FALSE.");
    f.write_attribute("igwx", 4572);
    f.write_attribute("scale_factor", 1.0);
    f.write_attribute("xk", xk);
    f.write("MillerIndices", miller);
    f.write_attribute("bg1", {0.67, 0.39, 0.00}, "MillerIndices");
    f.write_attribute("doc", "Miller Indices of the wave-vectors", "MillerIndices");
    f.write("evc", evc);
    f.write_attribute("doc", "Wave Functions, (npwx, nbnd)"s, "evc");
}

int
main(int argn, char** argv)
{
    test1();
    test2();
    test3();
    test4();
}

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;
using namespace mpi;

int
test_reduce()
{
    auto const& comm = Communicator::world();

    bool pr = (comm.rank() == 0);

    mdarray<std::complex<double>, 2> tmp_d({100, 200});
    mdarray<std::complex<double>, 2> tmp({100, 200});

    std::cout << comm.rank() << " tmp_d " << tmp_d.at(memory_t::host) << std::endl;
    std::cout << comm.rank() << " tmp " << tmp.at(memory_t::host) << std::endl;

    if (pr) {
        std::cout << "allreduce<double> " << std::endl;
    }
    comm.allreduce(tmp_d.at(memory_t::host), static_cast<int>(tmp.size()));
    comm.barrier();
    if (pr) {
        std::cout << "allreduce<double> : OK" << std::endl;
    }

    if (pr) {
        std::cout << "reduce<double> " << std::endl;
    }
    comm.reduce(tmp_d.at(memory_t::host), static_cast<int>(tmp.size()), 1);
    comm.barrier();
    if (pr) {
        std::cout << "reduce<double> : OK" << std::endl;
    }

    if (pr) {
        std::cout << "allreduce<reinterpret_cast<std::complex<double>>> " << std::endl;
    }
    comm.allreduce(reinterpret_cast<double*>(tmp.at(memory_t::host)), 2 * static_cast<int>(tmp.size()));

    if (pr) {
        std::cout << "reduce<reinterpret_cast<std::complex<double>>> " << std::endl;
    }
    comm.reduce(reinterpret_cast<double*>(tmp.at(memory_t::host)), 2 * static_cast<int>(tmp.size()), 1);

    if (pr) {
        std::cout << "allreduce<std::complex<double>> " << std::endl;
    }
    comm.allreduce(tmp.at(memory_t::host), static_cast<int>(tmp.size()));

    if (pr) {
        std::cout << "reduce<std::complex<double>> " << std::endl;
    }
    comm.reduce(tmp.at(memory_t::host), static_cast<int>(tmp.size()), 1);

    return 0;
}

int
main(int argn, char** argv)
{
    sirius::initialize(true);
    int result = call_test("test_reduce", test_reduce);
    sirius::finalize(true);
    return result;
}

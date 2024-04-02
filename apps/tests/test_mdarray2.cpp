/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

mdarray<double_complex, 1>&
buf(size_t size__)
{
    static mdarray<double_complex, 1> buf_;
    if (buf_.size() < size__) {
        buf_ = mdarray<double_complex, 1>(size__, memory_t::host | memory_t::device);
    }
    return buf_;
}

struct A
{
    static mdarray<double, 1> a;

    A()
    {
        a = mdarray<double, 1>(100);
        a.allocate(memory_t::device);
    }

    ~A()
    {
        a.deallocate_on_device();
    }
};

int
main(int argn, char** argv)
{
    sirius::initialize(1);

    buf(10);
    buf(20);
    buf(10);

    auto& b = buf(20);

    b.deallocate_on_device();

    A a;

#ifndef NDEBUG
    std::cout << "Allocated memory : " << mdarray_mem_count::allocated().load() << std::endl;
#endif

    sirius::finalize();
}

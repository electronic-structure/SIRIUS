/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "memory.hpp"

namespace sirius {

/// Return a memory pool.
/** A memory pool is created when this function called for the first time. */
memory_pool&
get_memory_pool(memory_t M__)
{
    static std::map<memory_t, memory_pool> memory_pool_;
    if (memory_pool_.count(M__) == 0) {
        memory_pool_.emplace(M__, memory_pool(M__));
    }
    return memory_pool_.at(M__);
}

} // namespace sirius

#include <map>

namespace sirius {

memory_t
get_memory_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    std::map<std::string, memory_t> const m = {{"none", memory_t::none},
                                               {"host", memory_t::host},
                                               {"host_pinned", memory_t::host_pinned},
                                               {"managed", memory_t::managed},
                                               {"device", memory_t::device}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "get_memory_t(): wrong label of the memory_t enumerator: " << name__;
        throw std::runtime_error(s.str());
    }
    return m.at(name__);
}

device_t
get_device_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    std::map<std::string, device_t> const m = {{"cpu", device_t::CPU}, {"gpu", device_t::GPU}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "get_device_t(): wrong label of the device_t enumerator: " << name__;
        throw std::runtime_error(s.str());
    }
    return m.at(name__);
}

template class mdarray<double, 1>;
template class mdarray<double, 2>;
template class mdarray<double, 3>;
template class mdarray<double, 4>;
template class mdarray<std::complex<double>, 1>;
template class mdarray<std::complex<double>, 2>;
template class mdarray<std::complex<double>, 3>;
template class mdarray<std::complex<double>, 4>;
template class mdarray<int, 1>;
template class mdarray<int, 2>;
template class mdarray<int, 3>;
template class mdarray<float, 1>;
template class mdarray<float, 2>;
template class mdarray<float, 3>;
template class mdarray<std::complex<float>, 1>;
template class mdarray<std::complex<float>, 2>;
template class mdarray<std::complex<float>, 3>;
template class mdarray<uint32_t, 1>;
template class mdarray<char, 1>;

} // namespace sirius

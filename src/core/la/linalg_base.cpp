/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file linalg_base.cpp
 *
 *  \brief Conversion of the linear algebra library type from a string.
 */

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include "core/la/linalg_base.hpp"

namespace sirius {

namespace la {

lib_t
get_lib_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    static const std::map<std::string, lib_t> map_to_type = {
            {"blas", lib_t::blas},      {"lapack", lib_t::lapack},   {"scalapack", lib_t::scalapack},
            {"cublas", lib_t::gpublas}, {"gpublas", lib_t::gpublas}, {"cublasxt", lib_t::cublasxt},
            {"magma", lib_t::magma},
    };

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of linear algebra type: " << name__;
        throw std::runtime_error(s.str());
    }

    return map_to_type.at(name__);
}

} // namespace la

} // namespace sirius

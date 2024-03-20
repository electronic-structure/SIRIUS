/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sirius_version.cpp
 *
 *  \brief Get version number and related quantities.
 */

#include <string>
#include "core/version.hpp"

namespace sirius {

int
major_version()
{
    return version::major_version;
};

int
minor_version()
{
    return version::minor_version;
}

int
revision()
{
    return version::revision;
}

std::string
git_hash()
{
    return std::string(version::git_hash);
}

std::string
git_branchname()
{
    return std::string(version::git_branchname);
}

std::string
build_date()
{
    return std::string(version::build_date);
}

} // namespace sirius

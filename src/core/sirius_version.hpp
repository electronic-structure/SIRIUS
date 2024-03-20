/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sirius_version.hpp
 *
 *  \brief Get version number and related quantities.
 */

#ifndef __SIRIUS_VERSION_HPP__
#define __SIRIUS_VERSION_HPP__

#include <string>

namespace sirius {

int
major_version();

int
minor_version();

int
revision();

std::string
git_hash();

std::string
git_branchname();

std::string
build_date();

} // namespace sirius

#endif

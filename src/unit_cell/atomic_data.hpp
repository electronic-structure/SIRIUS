/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Warning! This is an automatically generated header file!
/** \file atomic_data.hpp
 *  \brief Basic atomic data information.
 */

#ifndef __ATOMIC_DATA_HPP__
#define __ATOMIC_DATA_HPP__

#include <string>
#include <vector>
#include <map>

/// Describes single atomic level.
struct atomic_level_descriptor
{
    /// Principal quantum number.
    int n;

    /// Angular momentum quantum number.
    int l;

    /// Quantum number k.
    int k;

    /// Level occupancy.
    double occupancy;

    /// True if this is a core level.
    bool core{false};
};

/// Chemical symbols of the elements, index is Z - 1.
extern const std::vector<std::string> atomic_symb;

/// Names of the elements, index is Z - 1.
extern const std::vector<std::string> atomic_name;

/// Map from chemical symbol to Z.
extern const std::map<std::string, int> atomic_zn;

/// Ground state electronic configuration of the elements, index is Z - 1.
extern const std::vector<std::vector<atomic_level_descriptor>> atomic_conf;

/// Total energy of the free atoms (NIST, LDA), index is Z - 1.
extern const std::vector<double> atomic_energy_NIST_LDA;

#endif

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file debug.hpp
 *
 *  \brief Functions used for debug purposes.
 */

#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#include <mpi.h>
#include <fstream>

/// Return individual ofstream for each MPI rank.
inline std::ofstream
local_output()
{
    std::ofstream result;
    int i;
    MPI_Initialized(&i);
    if (i) {
        MPI_Comm_rank(MPI_COMM_WORLD, &i);
        result = std::ofstream("out" + std::to_string(i) + ".txt", std::ios::app);
    } else {
        result = std::ofstream("/dev/null");
    }
    return result;
}

#endif

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file make_sirius_comm.hpp
 *
 *  \brief Conversion between mpi4py and SIRIUS communicators.
 *
 *  The functions are defined in py_sirius.cpp: the mpi4py C-API is accessed through a table of function pointers
 *  which is local to each translation unit and only initialized there by import_mpi4py().
 */

#ifndef __MAKE_SIRIUS_COMM_HPP__
#define __MAKE_SIRIUS_COMM_HPP__

#include <pybind11/pybind11.h>
#include "core/mpi/communicator.hpp"

namespace sirius {

/// Create a SIRIUS communicator from an mpi4py.MPI.Comm object.
mpi::Communicator
make_sirius_comm(pybind11::object py_comm);

/// Create an mpi4py.MPI.Comm object from a SIRIUS communicator.
pybind11::object
make_pycomm(mpi::Communicator const& comm);

} // namespace sirius

#endif

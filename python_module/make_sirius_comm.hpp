/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/operators.h>
#include <iostream>

#include "core/mpi/communicator.hpp"
#include "core/rte/rte.hpp"

namespace sirius {

inline auto
make_sirius_comm(pybind11::object py_comm)
{
    PyObject* py_obj = py_comm.ptr();
    MPI_Comm* comm_p = PyMPIComm_Get(py_obj);
    if (comm_p == NULL) {
        RTE_THROW("invalid MPI_Comm object passed");
    }
    int rank = -1;
    MPI_Comm_rank(*comm_p, &rank);

    mpi::Communicator sirius_comm(*comm_p);
    return sirius_comm;
}

inline pybind11::handle
make_pycomm(const mpi::Communicator& comm)
{
    return pybind11::handle(PyMPIComm_New(comm.native()));
}

} // end namespace sirius

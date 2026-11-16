/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius.cpp
 *
 *  \brief Entry point of the py_sirius module.
 *
 *  The bindings are split into several translation units (py_sirius_*.cpp), see py_sirius.hpp.
 */

/* must come first, see py_sirius.hpp */
#include "py_sirius.hpp"
#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include "sirius.hpp"
#include "make_sirius_comm.hpp"

namespace py = pybind11;
using namespace sirius;

/* The mpi4py C-API (PyMPIComm_Get / PyMPIComm_New) must only be used in this translation unit, it is initialized
 * by import_mpi4py() in PYBIND11_MODULE below. */
mpi::Communicator
sirius::make_sirius_comm(py::object py_comm)
{
    MPI_Comm* comm_p = PyMPIComm_Get(py_comm.ptr());
    if (comm_p == nullptr) {
        throw py::error_already_set();
    }
    return mpi::Communicator(*comm_p);
}

py::object
sirius::make_pycomm(mpi::Communicator const& comm)
{
    PyObject* py_comm = PyMPIComm_New(comm.native());
    if (py_comm == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(py_comm);
}

PYBIND11_MODULE(py_sirius, m)
{
    // this is needed to be able to pass MPI_Comm from Python->C++
    if (import_mpi4py() < 0) {
        return;
    }
    // MPI_Init/Finalize
    int mpi_init_flag;
    MPI_Initialized(&mpi_init_flag);
    if (mpi_init_flag == true) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "loading SIRIUS python module, MPI already initialized\n";
        }
        sirius::initialize(false);
    } else {
        sirius::initialize(true);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "loading SIRIUS python module, initialize MPI\n";
        }
    }
    auto atexit = py::module::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        int mpi_finalized_flag;
        MPI_Finalized(&mpi_finalized_flag);
        if (mpi_finalized_flag == true) {
            sirius::finalize(false);
        } else {
            sirius::finalize(
                    /* call MPI_Finalize */
                    true,
                    /* reset device */
                    false);
        }
    }));

    try {
        py::module::import("numpy");
    } catch (...) {
        return;
    }

    init_operators(m);
    init_r3(m);

    init_context(m);
    init_dft(m);
    init_kpoints(m);
    init_arrays(m);
    init_smearing(m);

    m.def("num_devices", &acc::num_devices);
    m.def("omp_set_num_threads", &omp_set_num_threads);
    m.def("omp_get_num_threads", &omp_get_num_threads);
    m.def("make_sirius_comm", &make_sirius_comm);
    m.def("make_pycomm", &make_pycomm);
}

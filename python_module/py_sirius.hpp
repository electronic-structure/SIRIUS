/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius.hpp
 *
 *  \brief Declarations shared by the translation units of the py_sirius module.
 *
 *  The bindings are split into several translation units to reduce the peak memory usage of the compiler.
 *  Each init_* function registers one group of classes / functions; they are called from PYBIND11_MODULE in
 *  py_sirius.cpp in a fixed order (base classes have to be registered before derived classes).
 */

#ifndef __PY_SIRIUS_HPP__
#define __PY_SIRIUS_HPP__

/* This header must be included before any SIRIUS header: with GNU extensions (-std=gnu++NN) <complex.h>, pulled in
 * e.g. by vdwxc.h, includes the C99 header which defines the macro I; pybind11/stl.h uses I as a template parameter. */
#if defined(I)
#error "py_sirius.hpp must be included before headers that include <complex.h>"
#endif

#include <pybind11/pybind11.h>
/* every translation unit of the module must see the same type casters: std::vector, std::array, ... are
 * converted to / from python lists by the casters in stl.h */
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/json.hpp"

/// Convert a json object to the corresponding python object (dict, list, str, ...).
pybind11::object
pj_convert(nlohmann::json& node);

/// Register `__array__(dtype=None, copy=None)` on a class (numpy array protocol, numpy >= 2 passes `copy`).
/** `f` maps the (python) object to a py::array, the requested dtype / copy semantics are handled by numpy. */
template <typename Class, typename F>
inline Class&
def_array_protocol(Class& cls, F&& f)
{
    using namespace pybind11::literals;
    return cls.def(
            "__array__",
            [f](pybind11::object& self, pybind11::object dtype, pybind11::object copy) -> pybind11::object {
                pybind11::array arr = f(self);
                auto np             = pybind11::module_::import("numpy");
                if (copy.is_none()) {
                    return np.attr("asarray")(arr, "dtype"_a = dtype);
                }
                return np.attr("array")(arr, "dtype"_a = dtype, "copy"_a = copy);
            },
            "dtype"_a = pybind11::none(), "copy"_a = pybind11::none());
}

/// Communicator, Simulation_context, Atom, Atom_type, Unit_cell.
void
init_context(pybind11::module& m);

/// Field4D, Potential, Density, DFT_ground_state, Stress, Force and energy functions.
void
init_dft(pybind11::module& m);

/// K_point, K_point_set, application of the Hamiltonian and subspace initialization.
void
init_kpoints(pybind11::module& m);

/// FFT grid, mdarray wrappers, enums, G-vectors, wave-functions and periodic functions.
void
init_arrays(pybind11::module& m);

/// sirius.smearing submodule.
void
init_smearing(pybind11::module& m);

/// Hamiltonian / overlap / preconditioner operators.
void
init_operators(pybind11::module& m);

/// r3::vector and r3::matrix wrappers.
void
init_r3(pybind11::module& m);

#endif

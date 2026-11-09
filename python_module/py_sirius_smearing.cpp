/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius_smearing.cpp
 *
 *  \brief Python bindings: sirius.smearing submodule.
 */

#include "py_sirius.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dft/smearing.hpp"

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using complex_double = std::complex<double>;

void
init_smearing(py::module& m)
{
    /* sirius.smearing submodules */
    py::module smearing_module = m.def_submodule("smearing");
    py::enum_<smearing::smearing_t>(smearing_module, "SmearingEnum")
            .value("gauss", smearing::smearing_t::gaussian)
            .value("methfessel_paxton", smearing::smearing_t::methfessel_paxton)
            .value("fermi_dirac", smearing::smearing_t::fermi_dirac)
            .value("cold", smearing::smearing_t::cold);
    {
        py::module mpm = smearing_module.def_submodule("methfessel_paxton");
        mpm.def("entropy", py::vectorize(&smearing::methfessel_paxton::entropy), "x"_a, "w"_a, "n"_a = int{1});
        mpm.def("delta", py::vectorize(&smearing::methfessel_paxton::delta), "x"_a, "w"_a, "n"_a = int{1});
        mpm.def("occupancy", py::vectorize(&smearing::methfessel_paxton::occupancy), "x"_a, "w"_a, "n"_a = int{1});
        mpm.def("dxdelta", py::vectorize(&smearing::methfessel_paxton::dxdelta), "x"_a, "w"_a, "n"_a = int{1});
    }
    {
        py::module mcold = smearing_module.def_submodule("cold");
        mcold.def("entropy", py::vectorize(&smearing::cold::entropy), "x"_a, "w"_a);
        mcold.def("occupancy", py::vectorize(&smearing::cold::occupancy), "x"_a, "w"_a);
        mcold.def("delta", py::vectorize(&smearing::cold::delta), "x"_a, "w"_a);
        mcold.def("dxdelta", py::vectorize(&smearing::cold::dxdelta), "x"_a, "w"_a);
    }
    {
        py::module mfd = smearing_module.def_submodule("fermi_dirac");
        mfd.def("entropy", py::vectorize(&smearing::fermi_dirac::entropy), "x"_a, "w"_a);
        mfd.def("occupancy", py::vectorize(&smearing::fermi_dirac::occupancy), "x"_a, "w"_a);
        mfd.def("delta", py::vectorize(&smearing::fermi_dirac::delta), "x"_a, "w"_a);
        mfd.def("dxdelta", py::vectorize(&smearing::fermi_dirac::dxdelta), "x"_a, "w"_a);
    }
    {
        py::module mgauss = smearing_module.def_submodule("gaussian");
        mgauss.def("delta", py::vectorize(&smearing::gaussian::delta), "x"_a, "w"_a);
        mgauss.def("occupancy", py::vectorize(&smearing::gaussian::occupancy), "x"_a, "w"_a);
        mgauss.def("entropy", py::vectorize(&smearing::gaussian::entropy), "x"_a, "w"_a);
    }
    /* sirius.smearing submodules (end) */
}

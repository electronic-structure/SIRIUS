/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius_arrays.cpp
 *
 *  \brief Python bindings: FFT grid, mdarray, G-vectors, wave-functions and periodic functions.
 */

#include "py_sirius.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "core/fft/fft3d_grid.hpp"
#include "core/fft/gvec.hpp"
#include "core/memory.hpp"
#include "core/wf/wave_functions.hpp"
#include "function3d/smooth_periodic_function.hpp"
#include "function3d/periodic_function.hpp"

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using complex_double = std::complex<double>;

void
init_arrays(py::module& m)
{
    py::class_<fft::Grid>(m, "FFT3D_grid")
            .def_property_readonly("num_points", py::overload_cast<>(&fft::Grid::num_points, py::const_))
            .def_property_readonly("shape",
                                   [](const fft::Grid& obj) -> std::array<int, 3> { return {obj[0], obj[1], obj[2]}; });

    py::class_<mdarray<complex_double, 1>> mdarray1c(m, "mdarray1c");
    mdarray1c.def("on_device", &mdarray<complex_double, 1>::on_device)
            .def("copy_to_host", [](mdarray<complex_double, 1>& mdarray) { mdarray.copy_to(memory_t::host); });
    def_array_protocol(mdarray1c, [](py::object& obj) -> py::array {
        mdarray<complex_double, 1>& arr = obj.cast<mdarray<complex_double, 1>&>();
        int nrows                       = arr.size(0);
        return py::array_t<complex_double>({nrows}, {1 * sizeof(complex_double)}, arr.at(memory_t::host), obj);
    });

    py::class_<mdarray<double, 1>> mdarray1r(m, "mdarray1r");
    mdarray1r.def("on_device", &mdarray<double, 1>::on_device).def("copy_to_host", [](mdarray<double, 1>& mdarray) {
        mdarray.copy_to(memory_t::host);
    });
    def_array_protocol(mdarray1r, [](py::object& obj) -> py::array {
        mdarray<double, 1>& arr = obj.cast<mdarray<double, 1>&>();
        int nrows               = arr.size(0);
        return py::array_t<double>({nrows}, {1 * sizeof(double)}, arr.at(memory_t::host), obj);
    });

    py::class_<mdarray<complex_double, 2>> mdarray2c(m, "mdarray2c");
    mdarray2c.def("on_device", &mdarray<complex_double, 2>::on_device)
            .def("copy_to_host", [](mdarray<complex_double, 2>& mdarray) { mdarray.copy_to(memory_t::host); });
    def_array_protocol(mdarray2c, [](py::object& obj) -> py::array {
        mdarray<complex_double, 2>& arr = obj.cast<mdarray<complex_double, 2>&>();
        int nrows                       = arr.size(0);
        int ncols                       = arr.size(1);
        return py::array_t<complex_double>({nrows, ncols}, {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                           arr.at(memory_t::host), obj);
    });

    py::class_<mdarray<double, 2>> mdarray2(m, "mdarray2");
    mdarray2.def("on_device", &mdarray<double, 2>::on_device).def("copy_to_host", [](mdarray<double, 2>& mdarray) {
        mdarray.copy_to(memory_t::host, 0, mdarray.size(1));
    });
    def_array_protocol(mdarray2, [](py::object& obj) -> py::array {
        mdarray<double, 2>& arr = obj.cast<mdarray<double, 2>&>();
        int nrows               = arr.size(0);
        int ncols               = arr.size(1);
        return py::array_t<double>({nrows, ncols}, {1 * sizeof(double), nrows * sizeof(double)}, arr.at(memory_t::host),
                                   obj);
    });

    py::enum_<device_t>(m, "DeviceEnum").value("CPU", device_t::CPU).value("GPU", device_t::GPU);
    py::enum_<memory_t>(m, "MemoryEnum").value("device", memory_t::device).value("host", memory_t::host);
    py::enum_<wf::copy_to>(m, "CopyEnum")
            .value("none", wf::copy_to::none)
            .value("device", wf::copy_to::device)
            .value("host", wf::copy_to::host);
    py::class_<wf::num_mag_dims>(m, "num_mag_dims").def(py::init<int>());
    py::class_<wf::num_bands>(m, "num_bands").def(py::init<int>());
    py::class_<fft::Gvec_fft, std::shared_ptr<fft::Gvec_fft>>(m, "Gvec_fft");
    py::class_<fft::Gvec, std::shared_ptr<fft::Gvec>>(m, "Gvec")
            .def("count", py::overload_cast<>(&fft::Gvec::count, py::const_))
            .def("gkvec_cart", [](py::object& gvec_obj, int i) {
                auto& gvec = gvec_obj.cast<fft::Gvec&>();
                return gvec.gkvec_cart(gvec_index_t::global(i));
            });
    py::class_<wf::device_memory_guard>(m, "device_memory_guard");

    py::class_<wf::Wave_functions_base<double>, std::shared_ptr<wf::Wave_functions_base<double>>>(m,
                                                                                                  "Wave_functions_base")
            .def("copy_to", &wf::Wave_functions_base<double>::copy_to)
            .def("allocate", &wf::Wave_functions_base<double>::allocate)
            .def("memory_guard", &wf::Wave_functions<double>::memory_guard, "mem"_a, "copy_to"_a)
            .def("deallocate", &wf::Wave_functions_base<double>::deallocate);

    py::class_<wf::Wave_functions<double>, wf::Wave_functions_base<double>,
               std::shared_ptr<wf::Wave_functions<double>>>(m, "Wave_functions")
            .def(py::init<std::shared_ptr<fft::Gvec>, wf::num_mag_dims, wf::num_bands, memory_t>(), "gvecp"_a,
                 "num_mag_dims"_a, "mum_bands"_a, "memory_t"_a)
            .def("num_sc", &wf::Wave_functions<double>::num_sc)
            .def("num_wf", &wf::Wave_functions<double>::num_wf)
            .def("zero",
                 [](wf::Wave_functions<double>& wf) {
                     for (int i = 0; i < wf.num_md(); ++i)
                         wf.pw_coeffs(wf::spin_index(i)).zero();
                 })
            .def(
                    "pw_coeffs",
                    [](py::object& obj, int i) -> py::array_t<complex_double> {
                        auto& wf             = obj.cast<wf::Wave_functions<double>&>();
                        auto& matrix_storage = wf.pw_coeffs(wf::spin_index(i));
                        int nrows            = matrix_storage.size(0);
                        int ncols            = matrix_storage.size(1);
                        /* return underlying data as numpy.ndarray view */
                        return py::array_t<complex_double>({nrows, ncols},
                                                           {1 * sizeof(complex_double), nrows * sizeof(complex_double)},
                                                           matrix_storage.at(memory_t::host), obj);
                    },
                    py::keep_alive<0, 1>())

            .def("allocated_on_device", [](wf::Wave_functions<double>& wf) {
                bool is_on_device = true;
                for (int i = 0; i < wf.num_sc(); ++i) {
                    is_on_device = is_on_device && wf.pw_coeffs(wf::spin_index(i)).on_device();
                }
                return is_on_device;
            });

    py::class_<Smooth_periodic_function<double>>(m, "Smooth_periodic_function")
            .def("fft", [](Smooth_periodic_function<double>& obj) { return obj.fft_transform(-1); })
            .def("ifft", [](Smooth_periodic_function<double>& obj) { return obj.fft_transform(1); })
            .def_property("pw", py::overload_cast<>(&Smooth_periodic_function<double>::f_pw_local, py::const_),
                          py::overload_cast<>(&Smooth_periodic_function<double>::f_pw_local),
                          py::return_value_policy::reference_internal)
            .def_property("rg", py::overload_cast<>(&Smooth_periodic_function<double>::values, py::const_),
                          py::overload_cast<>(&Smooth_periodic_function<double>::values),
                          py::return_value_policy::reference_internal);

    py::class_<Periodic_function<double>>(m, "RPeriodic_function");
}

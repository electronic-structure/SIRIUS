/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "py_sirius.hpp"
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "core/r3/r3.hpp"

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using complex_double = std::complex<double>;

using PT = double;

template <class T>
std::string
show_vec(const r3::vector<T>& vec)
{
    std::string str = "[" + std::to_string(vec[0]) + "," + std::to_string(vec[1]) + "," + std::to_string(vec[2]) + "]";
    return str;
}

std::string
show_mat(const r3::matrix<double>& mat)
{
    std::string str = "[";
    for (int i = 0; i < 2; ++i) {
        str = str + "[" + std::to_string(mat(i, 0)) + "," + std::to_string(mat(i, 1)) + "," +
              std::to_string(mat(i, 2)) + "]" + "\n";
    }
    str = str + "[" + std::to_string(mat(2, 0)) + "," + std::to_string(mat(2, 1)) + "," + std::to_string(mat(2, 2)) +
          "]" + "]";
    return str;
}

void
init_r3(py::module& m)
{
    py::class_<r3::vector<int>> vector3d_int(m, "vector3d_int");
    vector3d_int.def(py::init<std::vector<int>>())
            .def("__call__", [](const r3::vector<int>& obj, int x) { return obj[x]; })
            .def("__repr__", [](const r3::vector<int>& vec) { return show_vec(vec); })
            .def("__len__", &r3::vector<int>::length);
    def_array_protocol(vector3d_int, [](py::object& obj) -> py::array {
        auto& v3d = obj.cast<r3::vector<int>&>();
        return py::array_t<int>(3, &v3d[0]);
    });

    py::class_<r3::vector<double>> vector3d_double(m, "vector3d_double");
    def_array_protocol(vector3d_double, [](py::object& obj) -> py::array {
        auto& v3d = obj.cast<r3::vector<double>&>();
        return py::array_t<double>(3, &v3d[0]);
    });
    vector3d_double.def(py::init<std::vector<double>>())
            .def("__call__", [](const r3::vector<double>& obj, int x) { return obj[x]; })
            .def("__repr__", [](const r3::vector<double>& vec) { return show_vec(vec); })
            .def("__len__", &r3::vector<double>::length)
            .def(py::self - py::self)
            .def(py::self * float())
            .def(py::self + py::self)
            .def(py::init<r3::vector<double>>());

    py::class_<r3::matrix<double>> matrix3d(m, "matrix3d");
    def_array_protocol(matrix3d, [](py::object& obj) -> py::array {
        auto& mat = obj.cast<r3::matrix<double>&>();
        return py::array_t<double>({3, 3}, {3 * sizeof(double), sizeof(double)}, &mat(0, 0));
    });
    matrix3d.def(py::init<std::vector<std::vector<double>>>())
            .def(py::init<>())
            .def("__call__", [](const r3::matrix<double>& obj, int x, int y) { return obj(x, y); })
            .def("__getitem__", [](const r3::matrix<double>& obj, int x, int y) { return obj(x, y); })
            .def("__mul__",
                 [](const r3::matrix<double>& obj, r3::vector<double> const& b) {
                     r3::vector<double> res = dot(obj, b);
                     return res;
                 })
            .def("__repr__", [](const r3::matrix<double>& mat) { return show_mat(mat); })
            .def(py::init<r3::matrix<double>>())
            .def("det", &r3::matrix<double>::det);

    py::class_<r3::matrix<int>> matrix3di(m, "matrix3di");
    def_array_protocol(matrix3di, [](py::object& obj) -> py::array {
        auto& mat = obj.cast<r3::matrix<int>&>();
        return py::array_t<int>({3, 3}, {3 * sizeof(int), sizeof(int)}, &mat(0, 0));
    });
    matrix3di.def(py::init<std::vector<std::vector<int>>>())
            .def(py::init<>())
            .def("__call__", [](const r3::matrix<int>& obj, int x, int y) { return obj(x, y); })
            .def("__getitem__", [](const r3::matrix<int>& obj, int x, int y) { return obj(x, y); })
            .def("__mul__",
                 [](const r3::matrix<int>& obj, r3::vector<int> const& b) {
                     r3::vector<int> res = dot(obj, b);
                     return res;
                 })
            .def(py::init<r3::matrix<int>>())
            .def("det", &r3::matrix<int>::det);
}

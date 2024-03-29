/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "python_module_includes.hpp"

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
    py::class_<r3::vector<int>>(m, "vector3d_int")
            .def(py::init<std::vector<int>>())
            .def("__call__", [](const r3::vector<int>& obj, int x) { return obj[x]; })
            .def("__repr__", [](const r3::vector<int>& vec) { return show_vec(vec); })
            .def("__len__", &r3::vector<int>::length)
            .def("__array__", [](r3::vector<int>& v3d) {
                py::array_t<int> x(3);
                auto r = x.mutable_unchecked<1>();
                r(0)   = v3d[0];
                r(1)   = v3d[1];
                r(2)   = v3d[2];
                return x;
            });

    py::class_<r3::vector<double>>(m, "vector3d_double")
            .def(py::init<std::vector<double>>())
            .def("__call__", [](const r3::vector<double>& obj, int x) { return obj[x]; })
            .def("__repr__", [](const r3::vector<double>& vec) { return show_vec(vec); })
            .def("__array__",
                 [](r3::vector<double>& v3d) {
                     py::array_t<double> x(3);
                     auto r = x.mutable_unchecked<1>();
                     r(0)   = v3d[0];
                     r(1)   = v3d[1];
                     r(2)   = v3d[2];
                     return x;
                 })
            .def("__len__", &r3::vector<double>::length)
            .def(py::self - py::self)
            .def(py::self * float())
            .def(py::self + py::self)
            .def(py::init<r3::vector<double>>());

    py::class_<r3::matrix<double>>(m, "matrix3d")
            .def(py::init<std::vector<std::vector<double>>>())
            .def(py::init<>())
            .def("__call__", [](const r3::matrix<double>& obj, int x, int y) { return obj(x, y); })
            .def(
                    "__array__",
                    [](const r3::matrix<double>& mat) {
                        return py::array_t<double>({3, 3}, {3 * sizeof(double), sizeof(double)}, &mat(0, 0));
                    },
                    py::return_value_policy::reference_internal)
            .def("__getitem__", [](const r3::matrix<double>& obj, int x, int y) { return obj(x, y); })
            .def("__mul__",
                 [](const r3::matrix<double>& obj, r3::vector<double> const& b) {
                     r3::vector<double> res = dot(obj, b);
                     return res;
                 })
            .def("__repr__", [](const r3::matrix<double>& mat) { return show_mat(mat); })
            .def(py::init<r3::matrix<double>>())
            .def("det", &r3::matrix<double>::det);

    py::class_<r3::matrix<int>>(m, "matrix3di")
            .def(py::init<std::vector<std::vector<int>>>())
            .def(py::init<>())
            .def("__call__", [](const r3::matrix<int>& obj, int x, int y) { return obj(x, y); })
            .def(
                    "__array__",
                    [](const r3::matrix<int>& mat) {
                        return py::array_t<int>({3, 3}, {3 * sizeof(int), sizeof(int)}, &mat(0, 0));
                    },
                    py::return_value_policy::reference_internal)
            .def("__getitem__", [](const r3::matrix<int>& obj, int x, int y) { return obj(x, y); })
            .def("__mul__",
                 [](const r3::matrix<int>& obj, r3::vector<int> const& b) {
                     r3::vector<int> res = dot(obj, b);
                     return res;
                 })
            .def(py::init<r3::matrix<int>>())
            .def("det", &r3::matrix<int>::det);
}

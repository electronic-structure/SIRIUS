/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius_dft.cpp
 *
 *  \brief Python bindings: potential, density, DFT ground state, forces and stress.
 */

#include "py_sirius.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dft/dft_ground_state.hpp"
#include "magnetization.hpp"

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using complex_double = std::complex<double>;

using json = nlohmann::json;

// inspired by: https://github.com/mdcb/python-jsoncpp11/blob/master/extension.cpp
py::object
pj_convert(json& node)
{
    switch (node.type()) {
        case json::value_t::null: {
            return py::reinterpret_borrow<py::object>(Py_None);
        }
        case json::value_t::boolean: {
            return py::bool_(node.get<bool>());
        }
        case json::value_t::string: {
            return py::str(node.get<std::string>());
        }
        case json::value_t::number_integer: {
            return py::int_(node.get<int>());
        }
        case json::value_t::number_unsigned: {
            return py::int_(node.get<unsigned int>());
        }
        case json::value_t::number_float: {
            return py::float_(node.get<double>());
        }
        case json::value_t::object: {
            py::dict result;
            for (auto it = node.begin(); it != node.end(); ++it) {
                json my_key(it.key());
                result[pj_convert(my_key)] = pj_convert(*it);
            }
            return result;
        }
        case json::value_t::array: {
            py::list result;
            for (auto it = node.begin(); it != node.end(); ++it) {
                result.append(pj_convert(*it));
            }
            return result;
        }
        default: {
            RTE_THROW("undefined json value");
            /* make compiler happy */
            return py::reinterpret_borrow<py::object>(Py_None);
        }
    }
}

void
init_dft(py::module& m)
{
    py::class_<Field4D>(m, "Field4D")
            .def(
                    "f_pw_local",
                    [](py::object& obj, int i) -> py::array_t<complex_double> {
                        Field4D& field       = obj.cast<Field4D&>();
                        auto& matrix_storage = field.component_raise(i).rg().f_pw_local();
                        int nrows            = matrix_storage.size(0);
                        /* return underlying data as numpy.ndarray view */
                        return py::array_t<complex_double>({nrows}, {1 * sizeof(complex_double)},
                                                           matrix_storage.at(memory_t::host), obj);
                    },
                    py::keep_alive<0, 1>())
            .def("f_rg",
                 [](py::object& obj, int i) -> py::array_t<double> {
                     Field4D& field       = obj.cast<Field4D&>();
                     auto& matrix_storage = field.component_raise(i).rg().values();
                     int nrows            = matrix_storage.size(0);
                     /* return underlying data as numpy.ndarray view */
                     return py::array_t<double>({nrows}, {1 * sizeof(double)}, matrix_storage.at(memory_t::host), obj);
                 })
            .def("component", py::overload_cast<int>(&Field4D::component), py::return_value_policy::reference_internal);

    /* opaque handle, only needed for full-potential calculations; None for pseudopotential calculations */
    py::class_<LAPW_radial_basis, std::shared_ptr<LAPW_radial_basis>>(m, "LAPW_radial_basis");

    py::class_<Potential, Field4D>(m, "Potential")
            .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
            .def("create_lapw_basis", &Potential::create_lapw_basis)
            .def("generate", &Potential::generate, "density"_a, "use_sym"_a, "transform_to_rg"_a)
            .def("fft_transform", &Potential::fft_transform)
            .def_property("vxc", py::overload_cast<>(&Potential::xc_potential),
                          py::overload_cast<>(&Potential::xc_potential), py::return_value_policy::reference_internal)
            .def_property("exc", py::overload_cast<>(&Potential::xc_energy_density),
                          py::overload_cast<>(&Potential::xc_energy_density),
                          py::return_value_policy::reference_internal)
            .def_property("vha", py::overload_cast<>(&Potential::hartree_potential),
                          py::overload_cast<>(&Potential::hartree_potential),
                          py::return_value_policy::reference_internal)
            .def_property("vloc", py::overload_cast<>(&Potential::local_potential),
                          py::overload_cast<>(&Potential::local_potential), py::return_value_policy::reference_internal)
            .def("energy_vha", &Potential::energy_vha)
            .def("energy_vxc", &Potential::energy_vxc)
            .def("energy_exc", &Potential::energy_exc)
            .def("PAW_total_energy", &Potential::PAW_total_energy)
            .def("PAW_one_elec_energy", &Potential::PAW_one_elec_energy);

    py::class_<Density, Field4D>(m, "Density")
            .def(py::init<Simulation_context&>(), py::keep_alive<1, 2>(), "ctx"_a)
            .def("initial_density", &Density::initial_density)
            .def("mixer_init", &Density::mixer_init)
            .def("check_num_electrons", &Density::check_num_electrons)
            .def("fft_transform", &Density::fft_transform)
            .def("mix", &Density::mix)
            .def(
                    "generate",
                    [](Density& density, K_point_set const& kset, std::shared_ptr<LAPW_radial_basis> lapw_basis,
                       bool symmetrize, bool add_core, bool transform_to_rg) {
                        density.generate<double>(kset, lapw_basis.get(), symmetrize, add_core, transform_to_rg);
                    },
                    "kpointset"_a, "lapw_basis"_a = nullptr, "symmetrize"_a = false, "add_core"_a = true,
                    "transform_to_rg"_a = false)
            .def("compute_atomic_mag_mom", &Density::compute_atomic_mag_mom)
            .def("save", &Density::save)
            .def("check_num_electrons", &Density::check_num_electrons)
            .def("get_magnetisation", &Density::get_magnetisation)
            .def("load", &Density::load);

    py::class_<DFT_ground_state>(m, "DFT_ground_state")
            .def(py::init<K_point_set&>(), py::keep_alive<1, 2>())
            .def("print_info", &DFT_ground_state::print_info)
            .def("initial_state", &DFT_ground_state::initial_state)
            .def("total_energy", &DFT_ground_state::total_energy)
            .def("serialize",
                 [](DFT_ground_state& dft) {
                     auto json = dft.serialize();
                     return pj_convert(json);
                 })
            .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
            .def(
                    "find",
                    [](DFT_ground_state& dft, double density_tol, double energy_tol, double initial_tol,
                       int num_dft_iter, bool write_state) {
                        json js = dft.find(density_tol, energy_tol, initial_tol, num_dft_iter, write_state);
                        return pj_convert(js);
                    },
                    "density_tol"_a, "energy_tol"_a, "initial_tol"_a, "num_dft_iter"_a, "write_state"_a)
            .def("check_scf_density",
                 [](DFT_ground_state& dft) {
                     json js = dft.check_scf_density();
                     return pj_convert(js);
                 })
            .def("k_point_set", &DFT_ground_state::k_point_set, py::return_value_policy::reference_internal)
            .def("potential", &DFT_ground_state::potential, py::return_value_policy::reference_internal)
            .def("forces", &DFT_ground_state::forces, py::return_value_policy::reference_internal)
            .def("stress", &DFT_ground_state::stress, py::return_value_policy::reference_internal)
            .def("update", &DFT_ground_state::update)
            .def("energy_kin_sum_pw", &DFT_ground_state::energy_kin_sum_pw);

    py::class_<Stress>(m, "Stress")
            .def(py::init<Simulation_context&, Density&, Potential&, K_point_set&>())
            .def("calc_stress_total", &Stress::calc_stress_total, py::return_value_policy::reference_internal)
            .def("calc_stress_har", &Stress::calc_stress_har, py::return_value_policy::reference_internal)
            .def("calc_stress_ewald", &Stress::calc_stress_ewald, py::return_value_policy::reference_internal)
            .def("calc_stress_xc", &Stress::calc_stress_xc, py::return_value_policy::reference_internal)
            .def("calc_stress_kin", &Stress::calc_stress_kin, py::return_value_policy::reference_internal)
            .def("calc_stress_vloc", &Stress::calc_stress_vloc, py::return_value_policy::reference_internal)
            .def("print_info", &Stress::print_info);

    py::class_<Force>(m, "Force")
            .def(py::init<Simulation_context&, Density&, Potential&, K_point_set&>())
            .def("calc_forces_total", py::overload_cast<bool>(&Force::calc_forces_total), "add_scf_corr"_a,
                 py::return_value_policy::reference_internal)
            .def_property_readonly("ewald", &Force::forces_ewald, py::return_value_policy::reference_internal)
            .def_property_readonly("hubbard", &Force::forces_hubbard, py::return_value_policy::reference_internal)
            .def_property_readonly("vloc", &Force::forces_vloc, py::return_value_policy::reference_internal)
            .def_property_readonly("nonloc", &Force::forces_nonloc, py::return_value_policy::reference_internal)
            .def_property_readonly("core", &Force::forces_core, py::return_value_policy::reference_internal)
            .def_property_readonly("scf_corr", &Force::forces_scf_corr, py::return_value_policy::reference_internal)
            .def_property_readonly("us", &Force::forces_us, py::return_value_policy::reference_internal)
            .def_property_readonly("total", &Force::forces_total, py::return_value_policy::reference_internal)
            .def("print_info", &Force::print_info);

    m.def("total_energy", &total_energy);
    m.def("ks_energy",
          py::overload_cast<Simulation_context const&, K_point_set const&, Density const&, Potential const&>(
                  &ks_energy));
    m.def("ewald_energy", &ewald_energy);
    m.def("energy_bxc", &energy_bxc);
    m.def("magnetization", &magnetization);
    m.def("sprint_magnetization", &sprint_magnetization);
}

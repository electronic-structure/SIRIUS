/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file py_sirius_context.cpp
 *
 *  \brief Python bindings: simulation context and unit cell.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "context/simulation_context.hpp"
#include "unit_cell_accessors.hpp"
#include "make_sirius_comm.hpp"
#include "py_sirius.hpp"

using namespace pybind11::literals;
namespace py = pybind11;
using namespace sirius;
using complex_double = std::complex<double>;

void
init_context(py::module& m)
{
    py::class_<mpi::Communicator>(m, "Communicator");

    py::class_<Simulation_context>(m, "Simulation_context")
            .def(py::init<std::string const&>())
            .def(py::init<std::string const&, mpi::Communicator const&>(), py::keep_alive<1, 3>())
            .def("initialize", &Simulation_context::initialize)
            .def("num_bands", py::overload_cast<>(&Simulation_context::num_bands, py::const_))
            .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands))
            .def("max_occupancy", &Simulation_context::max_occupancy)
            .def("num_fv_states", py::overload_cast<>(&Simulation_context::num_fv_states, py::const_))
            .def("num_spins", &Simulation_context::num_spins)
            .def("verbosity", py::overload_cast<>(&Simulation_context::verbosity, py::const_))
            .def("create_storage_file", &Simulation_context::create_storage_file)
            .def("processing_unit", py::overload_cast<>(&Simulation_context::processing_unit, py::const_))
            .def("processing_unit", py::overload_cast<std::string>(&Simulation_context::processing_unit))
            .def("gvec", &Simulation_context::gvec, py::return_value_policy::reference_internal)
            .def("full_potential", &Simulation_context::full_potential)
            .def("hubbard_correction", py::overload_cast<>(&Simulation_context::hubbard_correction, py::const_))
            .def("hubbard_correction", py::overload_cast<bool>(&Simulation_context::hubbard_correction))
            .def_property_readonly("kT", py::overload_cast<>(&Simulation_context::smearing_width, py::const_))
            .def_property_readonly("smearing_type", py::overload_cast<>(&Simulation_context::smearing, py::const_))
            .def("unit_cell", py::overload_cast<>(&Simulation_context::unit_cell, py::const_),
                 py::return_value_policy::reference)
            .def("pw_cutoff", py::overload_cast<>(&Simulation_context::pw_cutoff, py::const_))
            .def("pw_cutoff", py::overload_cast<double>(&Simulation_context::pw_cutoff))
            .def("gk_cutoff", py::overload_cast<>(&Simulation_context::gk_cutoff, py::const_))
            .def("gk_cutoff", py::overload_cast<double>(&Simulation_context::gk_cutoff))
            .def("aw_cutoff", py::overload_cast<>(&Simulation_context::aw_cutoff, py::const_))
            .def("aw_cutoff", py::overload_cast<double>(&Simulation_context::aw_cutoff))
            .def("num_spinors", &Simulation_context::num_spinors)
            .def("num_mag_dims", py::overload_cast<>(&Simulation_context::num_mag_dims, py::const_))
            .def("num_mag_dims", py::overload_cast<int>(&Simulation_context::num_mag_dims))
            .def_property(
                    "gamma_point", [](const Simulation_context& ctx) { return ctx.gamma_point(); },
                    [](Simulation_context& ctx, bool isGammaPoint) { ctx.gamma_point((isGammaPoint)); })
            .def("update", &Simulation_context::update)
            .def("use_symmetry", py::overload_cast<>(&Simulation_context::use_symmetry, py::const_))
            .def("processing_unit_memory_t", &Simulation_context::processing_unit_memory_t)
            .def(
                    "comm", [](Simulation_context& obj) { return make_pycomm(obj.comm()); },
                    py::return_value_policy::reference_internal)
            .def(
                    "comm_k", [](Simulation_context& obj) { return make_pycomm(obj.comm_k()); },
                    py::return_value_policy::reference_internal)
            .def(
                    "comm_fft", [](Simulation_context& obj) { return make_pycomm(obj.comm_fft()); },
                    py::return_value_policy::reference_internal);

    py::class_<Atom>(m, "Atom")
            .def("position", &Atom::position)
            .def("type_id", &Atom::type_id)
            .def("type", &Atom::type, py::return_value_policy::reference)
            .def_property_readonly("label", [](const Atom& obj) { return obj.type().label(); })
            .def_property_readonly("mass", [](const Atom& obj) { return obj.type().mass(); })
            .def("set_position", [](Atom& obj, const std::vector<double>& pos) {
                if (pos.size() != 3)
                    RTE_THROW("wrong input");
                obj.set_position({pos[0], pos[1], pos[2]});
            });

    py::class_<Atom_type>(m, "Atom_type")
            .def_property_readonly("augment", [](const Atom_type& atype) { return atype.augment(); })
            .def_property_readonly("mass", &Atom_type::mass)
            .def_property_readonly("num_atoms", [](const Atom_type& atype) { return atype.num_atoms(); });

    py::class_<Unit_cell>(m, "Unit_cell")
            .def("add_atom_type", &Unit_cell::add_atom_type, py::return_value_policy::reference)
            .def("add_atom",
                 [](Unit_cell& obj, std::string& label, std::array<double, 3>& v) { obj.add_atom(label, v); })
            .def("add_atom", py::overload_cast<const std::string, r3::vector<double>>(&Unit_cell::add_atom))
            .def("atom", py::overload_cast<int>(&Unit_cell::atom), py::return_value_policy::reference)
            .def("atom_type", py::overload_cast<int>(&Unit_cell::atom_type), py::return_value_policy::reference)
            .def("lattice_vectors", &Unit_cell::lattice_vectors)
            .def(
                    "set_lattice_vectors",
                    [](Unit_cell& obj, py::buffer l1, py::buffer l2, py::buffer l3) {
                        set_lattice_vectors(obj, l1, l2, l3);
                    },
                    "l1"_a, "l2"_a, "l3"_a)
            .def(
                    "set_lattice_vectors",
                    [](Unit_cell& obj, std::vector<double> l1, std::vector<double> l2, std::vector<double> l3) {
                        obj.set_lattice_vectors(r3::vector<double>(l1), r3::vector<double>(l2), r3::vector<double>(l3));
                    },
                    "l1"_a, "l2"_a, "l3"_a)
            .def("get_symmetry", &Unit_cell::get_symmetry)
            .def_property_readonly("num_electrons", &Unit_cell::num_electrons)
            .def_property_readonly("num_atoms", &Unit_cell::num_atoms)
            .def_property_readonly("num_valence_electrons", &Unit_cell::num_valence_electrons)
            .def_property_readonly("augmented", &Unit_cell::augment)
            .def_property_readonly("reciprocal_lattice_vectors", &Unit_cell::reciprocal_lattice_vectors)
            .def_property_readonly("min_mt_radius", &Unit_cell::min_mt_radius)
            .def_property_readonly("max_mt_radius", &Unit_cell::max_mt_radius)
            .def_property_readonly("omega", &Unit_cell::omega)
            .def("print_info", &Unit_cell::print_info);

    m.def("set_atom_positions", &set_atom_positions);
    m.def("atom_positions", &atom_positions);
}

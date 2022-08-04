#include "python_module_includes.hpp"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
// #include <stdexcept>

#include "unit_cell/atom_type_base.hpp"
#include "unit_cell/basis_functions_index.hpp"
#include "unit_cell/radial_functions_index.hpp"

using complex_double = std::complex<double>;
using namespace sddk;

using PT = double;

void
init_atom(py::module& m)
{
    py::class_<Atom>(m, "Atom")
        .def("position", &Atom::position)
        .def("type_id", &Atom::type_id)
        .def("type", &Atom::type, py::return_value_policy::reference)
        .def_property_readonly("label", [](const Atom& obj) { return obj.type().label(); })
        .def_property_readonly("mass", [](const Atom& obj) { return obj.type().mass(); })
        .def("set_position", [](Atom& obj, const std::vector<double>& pos) {
            if (pos.size() != 3)
                throw std::runtime_error("wrong input");
            obj.set_position({pos[0], pos[1], pos[2]});
        });

    py::class_<Atom_type_base>(m, "Atom_type_base")
        .def_property_readonly("name", &Atom_type_base::name)
        .def_property_readonly("zn", py::overload_cast<>(&Atom_type_base::zn, py::const_))
        .def_property_readonly("symbol", &Atom_type_base::symbol);

    py::class_<Atom_type>(m, "Atom_type")
        .def_property_readonly("name", &Atom_type::name)
        .def_property_readonly("zn", py::overload_cast<>(&Atom_type::zn, py::const_))
        .def_property_readonly("symbol", &Atom_type::symbol)
        .def("indexb", py::overload_cast<int>(&Atom_type::indexb, py::const_))
        .def("indexb_array", py::overload_cast<>(&Atom_type::indexb, py::const_),
             py::return_value_policy::reference_internal)
        .def("indexr", py::overload_cast<>(&Atom_type::indexr, py::const_), py::return_value_policy::reference_internal)
        .def_property_readonly("augment", [](const Atom_type& atype) { return atype.augment(); })
        .def_property_readonly("mass", &Atom_type::mass)
        .def_property_readonly("num_atoms", [](const Atom_type& atype) { return atype.num_atoms(); });

    py::class_<radial_functions_index>(m, "radial_functions_index")
        .def("__len__", [](radial_functions_index& obj) { return obj.size(); })
        .def("__getitem__", [](radial_functions_index& obj, int i) -> radial_function_index_descriptor {
            if (i >= obj.size())
                throw std::out_of_range("error");
            return obj[i];
        });

    py::class_<radial_function_index_descriptor>(m, "radial_function_index_descriptor")
        .def("__repr__",
             [](radial_function_index_descriptor& obj) -> std::string {
                 return "l: " + std::to_string(obj.l) + "\n" + "j: " + std::to_string(obj.j) + "\n" +
                        "order: " + std::to_string(obj.order) + "\n" + "idxlo: " + std::to_string(obj.idxlo) + "\n";
             })
        .def_readonly("j", &radial_function_index_descriptor::j)
        .def_readonly("l", &radial_function_index_descriptor::l)
        .def_readonly("order", &radial_function_index_descriptor::order)
        .def_readonly("idxlo", &radial_function_index_descriptor::idxlo);

    py::class_<basis_functions_index>(m, "basis_function_index")
        .def("__array__",
             [](basis_functions_index& bfi) -> std::vector<basis_function_index_descriptor> {
                 std::vector<basis_function_index_descriptor> vec;
                 int size = bfi.size();
                 for (int i = 0; i < size; ++i) {
                     vec.push_back(bfi[i]);
                 }
                 return vec;
             })
        .def("__len__", [](basis_functions_index& obj) { return obj.size(); })
        .def("__getitem__",
             [](basis_functions_index& obj, int i) {
                 if (i >= obj.size())
                     throw std::out_of_range("out of range in basis_function_index");
                 return obj[i];
             })
        .def_property_readonly("size", &basis_functions_index::size)
        .def_property_readonly("size_aw", &basis_functions_index::size_aw)
        .def_property_readonly("size_lo", &basis_functions_index::size_lo);

    py::class_<basis_function_index_descriptor>(m, "basis_function_index_descriptor")
        .def("__repr__",
             [](basis_function_index_descriptor& obj) {
                 return "lm: " + std::to_string(obj.lm) + "\n" + "m: " + std::to_string(obj.m) + "\n" +
                        "j: " + std::to_string(obj.j) + "\n";
             })
        .def_readonly("lm", &basis_function_index_descriptor::lm)
        .def_readonly("m", &basis_function_index_descriptor::m)
        .def_readonly("j", &basis_function_index_descriptor::j);

    m.def("lmmax", &utils::lmmax);
    m.def("l_by_lm", &utils::l_by_lm);
}

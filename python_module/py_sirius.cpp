#include <pybind11/pybind11.h>
#include <sirius.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "json.hpp"


using namespace pybind11::literals; // to bring in the `_a` literal
namespace py = pybind11;
using namespace sirius;
using namespace geometry3d;
using json = nlohmann::json;
using nlohmann::basic_json;

//inspired by: https://github.com/mdcb/python-jsoncpp11/blob/master/extension.cpp
py::object pj_convert(json& node)
{
    switch (node.type()) {
        case json::value_t::null: {
            return py::reinterpret_borrow<py::object>(Py_None);
        }
        case json::value_t::boolean: {
            bool b(node);
            return py::bool_(b);
        }
        case json::value_t::string: {
            std::string s;
            s = static_cast<std::string const&>(node);
            return py::str(s);
        }
        case json::value_t::number_integer: {
            int i(node);
            return py::int_(i);
        }
        case json::value_t::number_unsigned: {
            unsigned int u(node);
            return py::int_(u);
        }
        case json::value_t::number_float: {
            float f(node);
            return py::float_(f);
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
            throw std::runtime_error("undefined json value");
            /* make compiler happy */
            return py::reinterpret_borrow<py::object>(Py_None);
        }
    }
}

std::string show_mat(const matrix3d<double>& mat)
{
  std::string str = "[";
  for(int i=0; i<2; ++i)
  {str = str +"[" + std::to_string(mat(i,0)) + "," + std::to_string(mat(i,1)) + "," + std::to_string(mat(i,2)) + "]"+"\n";}
  str = str + "[" + std::to_string(mat(2,0)) + "," + std::to_string(mat(2,1)) + "," + std::to_string(mat(2,2)) + "]"+ "]";
  return str;
}

template<class T>
std::string show_vec(const vector3d<T>& vec)
{
  std::string str = "[" + std::to_string(vec[0]) + "," + std::to_string(vec[1]) + "," + std::to_string(vec[2]) + "]";
  return str;
}

PYBIND11_MODULE(py_sirius, m){

    m.def("initialize", []()
                        {
                            sirius::initialize();
                        });
    m.def("finalize", []()
                      {
                          sirius::finalize();
                      });

    py::class_<Unit_cell>(m, "Unit_cell")
        .def("add_atom", (void (Unit_cell::*)(const std::string, vector3d<double>)) &Unit_cell::add_atom)
        .def("get_symmetry", &Unit_cell::get_symmetry)
        .def("reciprocal_lattice_vectors", &Unit_cell::reciprocal_lattice_vectors)
        .def("add_atom_type", (void (Unit_cell::*)(const std::string, const std::string)) &Unit_cell::add_atom_type)
        .def("atom_type", (Atom_type& (Unit_cell::*)(int)) &Unit_cell::atom_type, py::return_value_policy::reference)
        .def("set_lattice_vectors", (void (Unit_cell::*)(matrix3d<double>)) &Unit_cell::set_lattice_vectors);

    py::class_<Parameters_input>(m, "Parameters_input")
        .def(py::init<>())
        .def_readwrite("potential_tol_", &Parameters_input::potential_tol_)
        .def_readwrite("energy_tol_", &Parameters_input::energy_tol_)
        .def_readwrite("num_dft_iter_", &Parameters_input::num_dft_iter_);

  py::class_<Simulation_parameters>(m, "Simulation_parameters")
    .def(py::init<>())
    .def("pw_cutoff", &Simulation_parameters::pw_cutoff)
    .def("parameters_input", (Parameters_input& (Simulation_parameters::*)()) &Simulation_parameters::parameters_input, py::return_value_policy::reference)
    .def("num_spin_dims", &Simulation_parameters::num_spin_dims)
    .def("num_mag_dims", &Simulation_parameters::num_mag_dims)
    .def("set_gamma_point", &Simulation_parameters::set_gamma_point)
    .def("set_iterative_solver_tolerance", &Simulation_parameters::set_iterative_solver_tolerance)
    .def("set_pw_cutoff", &Simulation_parameters::set_pw_cutoff);

  py::class_<Simulation_context_base, Simulation_parameters>(m, "Simulation_context_base")
    .def("gvec", &Simulation_context_base::gvec)
    .def("fft", &Simulation_context_base::fft)
    .def("unit_cell", (Unit_cell& (Simulation_context_base::*)()) &Simulation_context_base::unit_cell, py::return_value_policy::reference);

  py::class_<Simulation_context, Simulation_context_base>(m, "Simulation_context")
    .def(py::init<>())
    .def(py::init<std::string const&>())
    .def("initialize", &Simulation_context::initialize)
    .def("num_bands", py::overload_cast<>(&Simulation_context::num_bands, py::const_))
    .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands))
    .def("set_verbosity", &Simulation_context::set_verbosity);

  py::class_<z_column_descriptor> (m, "z_column_descriptor")
     .def_readwrite("x", &z_column_descriptor::x)
     .def_readwrite("y", &z_column_descriptor::y)
     .def_readwrite("z", &z_column_descriptor::z)
     .def(py::init<int, int , std::vector<int>>());

   py::class_<Gvec>(m, "Gvec")
     .def(py::init<matrix3d<double>, double, bool>())
     .def("num_gvec", &sddk::Gvec::num_gvec)
     .def("count", &sddk::Gvec::count)
     .def("offset", &sddk::Gvec::offset)
     .def("gvec", &sddk::Gvec::gvec)
     .def("num_zcol", &sddk::Gvec::num_zcol)
     .def("gvec_alt", [](Gvec &obj, int idx){vector3d<int> vec(obj.gvec(idx)); //alternative solution: returns an array.
       std::vector<int> retr;
       retr.push_back(vec[0]);
       retr.push_back(vec[1]);
       retr.push_back(vec[2]);
       return retr;})
     .def("index_by_gvec", [](Gvec &obj, std::vector<int> vec){vector3d<int> vec3d(vec);
       return obj.index_by_gvec(vec3d);})
     .def("zcol", [](Gvec &gvec, int idx){
       z_column_descriptor obj(gvec.zcol(idx));
       py::dict dict("x"_a = obj.x, "y"_a = obj.y, "z"_a = obj.z);
       return dict;})
     .def("index_by_gvec", &Gvec::index_by_gvec);

  py::class_<vector3d<int>>(m, "vector3d_int")
    .def(py::init<std::vector<int>>())
    .def("__call__", [](const vector3d<int> &obj, int x){return obj[x];})
    .def("__repr__", [](const vector3d<int> &vec){return show_vec(vec);})
    .def(py::init<vector3d<int>>());

  py::class_<vector3d<double>>(m, "vector3d_double")
    .def(py::init<std::vector<double>>())
    .def("__call__", [](const vector3d<double> &obj, int x){return obj[x];})
    .def("__repr__", [](const vector3d<double> &vec){return show_vec(vec);})
    .def("length", &vector3d<double>::length)
    .def(py::self - py::self)
    .def(py::self * float())
    .def(py::self + py::self)
    .def(py::init<vector3d<double>>());

  py::class_<matrix3d<double>>(m, "matrix3d")
    .def(py::init<std::vector<std::vector<double>>>())
    .def(py::init<>()) //to create a zero matrix
    .def("__call__", [](const matrix3d<double> &obj, int x, int y){return obj(x,y);})
    .def(py::self * py::self)
    .def("__getitem__", [](const matrix3d<double> &obj, int x, int y){return obj(x,y);})
    .def("__mul__", [](const matrix3d<double> & obj, vector3d<double> const& b){
      vector3d<double> res = obj * b;
      return res;})
    .def("__repr__", [](const matrix3d<double> &mat){return show_mat(mat);})
    .def(py::init<matrix3d<double>>())
    .def("det", &matrix3d<double>::det);

  py::class_<Potential>(m, "Potential")
    .def(py::init<Simulation_context&>())
    .def("generate", &Potential::generate)
    .def("allocate", &Potential::allocate)
    .def("load", &Potential::load);

  py::class_<Density>(m, "Density")
    .def(py::init<Simulation_context&>())
    .def("initial_density", &Density::initial_density)
    .def("allocate", &Density::allocate)
    .def("load", &Density::load);

  py::class_<Band>(m, "Band")
    .def(py::init<Simulation_context&>())
    .def("initialize_subspace", py::overload_cast<K_point_set&, Hamiltonian&>(&Band::initialize_subspace, py::const_))
    .def("solve", &Band::solve);

  py::class_<DFT_ground_state>(m, "DFT_ground_state")
    .def(py::init<Simulation_context&>())
    .def("print_info", &DFT_ground_state::print_info)
    .def("initial_state", &DFT_ground_state::initial_state)
    .def("print_magnetic_moment", &DFT_ground_state::print_magnetic_moment)
    .def("total_energy", &DFT_ground_state::total_energy)
    .def("band", &DFT_ground_state::band)
    .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
    .def("find", [](DFT_ground_state& dft, double potential_tol, double energy_tol, int num_dft_iter, bool write_state){
      json js = dft.find(potential_tol, energy_tol, num_dft_iter, write_state);
      return pj_convert(js);})
    .def("k_point_set", &DFT_ground_state::k_point_set, py::return_value_policy::reference_internal)
    .def("hamiltonian", &DFT_ground_state::hamiltonian, py::return_value_policy::reference)
    .def("potential", &DFT_ground_state::potential, py::return_value_policy::reference);

  py::class_<K_point>(m, "K_point")
    .def("band_energy", py::overload_cast<int, int>(&K_point::band_energy))
    .def("vk", &K_point::vk, py::return_value_policy::reference);

  py::class_<K_point_set>(m, "K_point_set")
    .def(py::init<Simulation_context&>())
    .def(py::init<Simulation_context&, std::vector<vector3d<double>>>())
    .def("initialize", py::overload_cast<>(&K_point_set::initialize))
    .def("num_kpoints", &K_point_set::num_kpoints)
    .def("energy_fermi", &K_point_set::energy_fermi)
    .def("get_band_energies", &K_point_set::get_band_energies, py::return_value_policy::reference)
    .def("sync_band_energies", &K_point_set::sync_band_energies)
    .def("__call__", &K_point_set::operator[], py::return_value_policy::reference)
    .def("add_kpoint", [](K_point_set &ks, std::vector<double> &v, double weight){
      vector3d<double> vec3d(v);
      ks.add_kpoint(&vec3d[0], weight);})
    .def("add_kpoint", [](K_point_set &ks, vector3d<double> &v, double weight){
      ks.add_kpoint(&v[0], weight);});

  py::class_<Hamiltonian>(m, "Hamiltonian")
    .def(py::init<Simulation_context&, Potential&>());

  py::class_<Stress>(m, "Stress")
    .def(py::init<Simulation_context&, K_point_set&, Density&, Potential&>())
    .def("calc_stress_total", &Stress::calc_stress_total, py::return_value_policy::reference_internal)
    .def("print_info", &Stress::print_info);

  py::class_<Force>(m, "Force")
    .def(py::init<Simulation_context&, Density&, Potential&, Hamiltonian&, K_point_set&>())
    .def("calc_forces_total", &Force::calc_forces_total, py::return_value_policy::reference_internal)
    .def("print_info", &Force::print_info);

  py::class_<Free_atom>(m, "Free_atom")
    .def(py::init<Simulation_parameters&, std::string>())
    .def(py::init<Simulation_parameters&, int>())
    .def("ground_state", [](Free_atom& atom, double energy_tol, double charge_tol, bool rel)
                         {
                             json js = atom.ground_state(energy_tol, charge_tol, rel);
                             return pj_convert(js);
                         });
}

//Last update: 21.03.2018, 12:39 pm
#include <pybind11/pybind11.h>
#include <sirius.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

using namespace pybind11::literals; // to bring in the `_a` literal
namespace py = pybind11;
using namespace sirius;
using namespace geometry3d;


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

void initializer()
{
  sirius::initialize(1);
}

void finalizer()
{
  sirius::finalize(1);
}


PYBIND11_MODULE(sirius, m){

   m.def("initialize", &initializer);
   m.def("finalize", &finalizer);

  py::class_<Atom_type>(m, "Atom_type")
    //.def("zn", (int (Atom_type::*)(int)) &Atom_type::zn, "Set zn")
    .def("zn", py::overload_cast<int>(&Atom_type::zn))
    .def("zn", py::overload_cast<>(&Atom_type::zn, py::const_))
    .def("add_beta_radial_function", &Atom_type::add_beta_radial_function)
    .def("num_mt_points", &Atom_type::num_mt_points);

  py::class_<Unit_cell>(m, "Unit_cell")
    .def("add_atom", (void (Unit_cell::*)(const std::string, vector3d<double>)) &Unit_cell::add_atom)
    .def("get_symmetry", &Unit_cell::get_symmetry)
    .def("add_atom_type", (void (Unit_cell::*)(const std::string, const std::string)) &Unit_cell::add_atom_type)
    .def("atom_type", (Atom_type& (Unit_cell::*)(int)) &Unit_cell::atom_type, py::return_value_policy::reference)
    .def("set_lattice_vectors", (void (Unit_cell::*)(matrix3d<double>)) &Unit_cell::set_lattice_vectors);

  py::class_<Parameters_input>(m, "Parameters_input")
    .def_readwrite("potential_tol_", &Parameters_input::potential_tol_)
    .def_readwrite("energy_tol_", &Parameters_input::energy_tol_)
    .def_readwrite("num_dft_iter_", &Parameters_input::num_dft_iter_)
    .def(py::init<>());

  py::class_<Simulation_parameters>(m, "Simulation_parameters")
    .def("pw_cutoff", &Simulation_parameters::pw_cutoff)
    .def("parameters_input", (Parameters_input& (Simulation_parameters::*)()) &Simulation_parameters::parameters_input, py::return_value_policy::reference)
    //.def("parameters_input", [](const py::object& obj){
      //Simulation_context *ctx = obj.cast<Simulation_context *>();
      //return ctx->parameters_input();})
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
    .def("num_bands", py::overload_cast<int>(&Simulation_context::num_bands));

  py::class_<z_column_descriptor> (m, "z_column_descriptor")
     .def_readwrite("x", &z_column_descriptor::x)
     .def_readwrite("y", &z_column_descriptor::y)
     .def_readwrite("z", &z_column_descriptor::z)
     //.def(py::init<>())
     //.def("__repr__", [](const z_column_descriptor &obj){return show_z_colum_descriptor(obj);)
     .def(py::init<int, int , std::vector<int>>());



   py::class_<Gvec>(m, "Gvec")
     .def(py::init<matrix3d<double>, double, bool>())
     .def("num_gvec", &sddk::Gvec::num_gvec)
     .def("count", &sddk::Gvec::count)
     .def("offset", &sddk::Gvec::offset)
     .def("gvec", &sddk::Gvec::gvec)
     .def("gvec_alt", [](Gvec &obj, int idx){vector3d<int> vec(obj.gvec(idx)); //alternative solution: returns an array.
       std::vector<int> retr;
       retr.push_back(vec[0]);
       retr.push_back(vec[1]);
       retr.push_back(vec[2]);
       return retr;})
     .def("index_by_gvec", [](Gvec &obj, std::vector<int> vec){vector3d<int> vec3d(vec);
       return obj.index_by_gvec(vec3d);})
     .def("zcol", [](Gvec &gvec, int idx){ //Error on Python side: TypeError: zcol(): incompatible function arguments. The following argument types are supported:
       z_column_descriptor obj(gvec.zcol(idx));
       py::dict dict("x"_a = obj.x, "y"_a = obj.y, "z"_a = obj.z);
       return dict;})
     .def("zcol", &sddk::Gvec::zcol)
     //.def("__repr__", [](int idx, const Gvec &gvec){std::cout<<idx;})
     .def("index_by_gvec", &Gvec::index_by_gvec);
     //.def("index_by_gvec", py::overload_cast<>);

  py::class_<matrix3d<double>>(m, "matrix3d") //py::class_ constructor
    .def(py::init<std::vector<std::vector<double>>>())
    .def(py::init<>()) //to create a zero matrix
    .def("__call__", [](const matrix3d<double> &obj, int x, int y){return obj(x,y);})
    .def(py::self * py::self)
    .def("__repr__", [](const matrix3d<double> &mat){return show_mat(mat);})
    .def(py::init<matrix3d<double>>())
    .def("det", &matrix3d<double>::det);

  py::class_<vector3d<int>>(m, "vector3d_int")
    .def(py::init<std::vector<int>>())
    .def("__call__", [](const vector3d<int> &obj, int x){return obj[x];})
    .def("__repr__", [](const vector3d<int> &vec){return show_vec(vec);})
    .def(py::init<vector3d<int>>());

  py::class_<vector3d<double>>(m, "vector3d_double")
    .def(py::init<std::vector<double>>())
    .def("__call__", [](const vector3d<double> &obj, int x){return obj[x];})
    .def("__repr__", [](const vector3d<double> &vec){return show_vec(vec);})
    .def(py::init<vector3d<double>>());

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

  py::class_<DFT_ground_state>(m, "DFT_ground_state")
    .def(py::init<Simulation_context&>())
    .def("print_info", &DFT_ground_state::print_info)
    .def("initial_state", &DFT_ground_state::initial_state)
    .def("print_magnetic_moment", &DFT_ground_state::print_magnetic_moment)
    .def("total_energy", &DFT_ground_state::total_energy)
    //.def("serialize", &DFT_ground_state::serialize, py::return_value_policy::reference)
    .def("density", &DFT_ground_state::density, py::return_value_policy::reference)
    .def("find", &DFT_ground_state::find)
    .def("potential", &DFT_ground_state::potential, py::return_value_policy::reference);
}

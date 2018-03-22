//Last update: 21.03.2018, 12:39 pm
#include <pybind11/pybind11.h>
#include <sirius.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

namespace py = pybind11;
using namespace sirius;
using namespace geometry3d;

void vector_wrapper(std::vector<std::vector<double>> a, double (&x)[3][3])
{
  for(int i = 0; i < 3; ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      x[i][j] = a[i][j];
    }
  }
}

matrix3d<double> factory(std::vector<std::vector<double>> x)
{
    double arr[3][3];//empty array
    vector_wrapper(x, arr);
    return matrix3d<double>(arr);
}


std::string show_mat(const matrix3d<double>& mat)
{
  std::string str = "[";
  for(int i=0; i<2; ++i)
  {str = str +"[" + std::to_string(mat(i,0)) + "," + std::to_string(mat(i,1)) + "," + std::to_string(mat(i,2)) + "]"+"\n";}
  str = str + "[" + std::to_string(mat(2,0)) + "," + std::to_string(mat(2,1)) + "," + std::to_string(mat(2,2)) + "]"+ "]";
  return str;
}

std::string show_vec(const vector3d<double>& vec)
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

/*
Atom_type atom_type_wrapper(Atom_type&& src)
{
  return Atom_type(src);
}
*/
PYBIND11_MODULE(sirius, m){
  py::class_<Atom_type>(m, "Atom_type")
    //.def(py::init([](Atom_type src){return atom_type_wrapper(std::move(src));}))
    //.def("set_radial_grid", )set_radial_grid(radial_grid_t grid_type__, int num_points__, double rmin__, double rmax__)
    //.def("zn", (int (Atom_type::*)(int)) &Atom_type::zn)
    //.def("zn", [](Simulation_context& ctx, const int id, const int charge){
      //Atom_type& atype = ctx.unit_cell().atom_type(0);
      //atype.zn(charge);})
    .def("zn", (int (Atom_type::*)(int)) &Atom_type::zn)
    .def("add_beta_radial_function", &Atom_type::add_beta_radial_function)
    .def("num_mt_points", &Atom_type::num_mt_points);

  py::class_<Unit_cell>(m, "Unit_cell")
    .def("add_atom", (void (Unit_cell::*)(const std::string, vector3d<double>)) &Unit_cell::add_atom)
    .def("get_symmetry", &Unit_cell::get_symmetry)
    .def("add_atom_type", (void (Unit_cell::*)(const std::string, const std::string)) &Unit_cell::add_atom_type)
    //.def("atom_type", (Atom_type& (Unit_cell::*)(int)) &Unit_cell::atom_type, py::return_value_policy::reference)
    //.def("atom_type", &Unit_cell::atom_type, py::return_value_policy::reference)
    .def("atom_type", (Atom_type& (Unit_cell::*)(int)) &Unit_cell::atom_type, py::return_value_policy::reference)
    .def("set_lattice_vectors", (void (Unit_cell::*)(matrix3d<double>)) &Unit_cell::set_lattice_vectors);

  py::class_<Simulation_parameters>(m, "Simulation_parameters")
    .def("pw_cutoff", &Simulation_parameters::pw_cutoff)
    .def("set_pw_cutoff", &Simulation_parameters::set_pw_cutoff);

  py::class_<Simulation_context_base, Simulation_parameters>(m, "Simulation_context_base")
    .def("gvec", &Simulation_context_base::gvec)
    .def("fft", &Simulation_context_base::fft)
    .def("unit_cell", (Unit_cell& (Simulation_context_base::*)()) &Simulation_context_base::unit_cell, py::return_value_policy::reference);

  py::class_<Simulation_context, Simulation_context_base>(m, "Simulation_context")
    .def(py::init<std::string const&>())
    .def("initialize", &Simulation_context::initialize);

   py::class_<Gvec>(m, "Gvec")
     .def(py::init<matrix3d<double>, double, bool>())
     .def("count", &Gvec::count)
     .def("offset", &Gvec::offset)
     .def("gvec", &Gvec::gvec)
     .def("index_by_gvec", &Gvec::index_by_gvec);
   m.def("initialize", &initializer);
   m.def("finalize", &finalizer);

  py::class_<matrix3d<double>>(m, "matrix3d") //py::class_ constructor
    .def(py::init(&factory))
    .def(py::init<>()) //to create a zero matrix
    .def("__call__", [](const matrix3d<double> &obj, int x, int y){return obj(x,y);})
    .def(py::self * py::self)
    .def("__repr__", [](const matrix3d<double> &mat){return show_mat(mat);})
    .def(py::init<matrix3d<double>>())
    .def("det", &matrix3d<double>::det);

  py::class_<vector3d<double>>(m, "vector3d")
    .def(py::init<std::vector<double>>())
    .def("__call__", [](const vector3d<double> &obj, int x){return obj[x];})
    .def("__repr__", [](const vector3d<double> &vec){return show_vec(vec);})
    .def(py::init<vector3d<double>>());
}

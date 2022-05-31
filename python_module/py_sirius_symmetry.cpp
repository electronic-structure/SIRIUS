#include "python_module_includes.hpp"
#include "symmetry/lattice.hpp"

using namespace sirius;
namespace py = pybind11;


void init_symmetry(py::module& m)
{
    m.def("find_lat_sym", &find_lat_sym);
}

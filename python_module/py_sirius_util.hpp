#ifndef PY_SIRIUS_UTIL_H
#define PY_SIRIUS_UTIL_H

#include <iostream>
#include <mpi4py/mpi4py.h>
#include <mpi/communicator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include "unit_cell/unit_cell.hpp"
#include <density/density.hpp>
#include <k_point/k_point_set.hpp>

namespace sirius {
void set_atom_positions(Unit_cell& unit_cell, pybind11::buffer positions);

pybind11::array_t<double> atom_positions(Unit_cell& unit_cell);

void set_lattice_vectors(Unit_cell& unit_cell, pybind11::buffer l1buf, pybind11::buffer l2buf, pybind11::buffer l3buf);

sddk::Communicator make_sirius_comm(pybind11::object py_comm);

pybind11::handle make_pycomm(const sddk::Communicator& comm);

std::vector<double> magnetization(Density& density);

std::string sprint_magnetization(K_point_set& kset, const Density& density);

} // namespace sirius

#endif /* PY_SIRIUS_UTIL_H */

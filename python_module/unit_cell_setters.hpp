#ifndef SET_ATOM_POSITIONS_H
#define SET_ATOM_POSITIONS_H

#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>

#include "Unit_cell/unit_cell.hpp"


namespace sirius {
void set_atom_positions(Unit_cell& unit_cell, pybind11::buffer positions)
{
    using namespace pybind11;

    buffer_info info = positions.request();
    int npositions = info.shape[0];
    if (info.shape[1] != 3) {
        throw std::runtime_error("set_atom_positions: wrong shape");
    }
    if (npositions != unit_cell.num_atoms()) {
        throw std::runtime_error("number of positions != number of atoms");
    }
    if (info.format != format_descriptor<double>::format())
        throw std::runtime_error("Incompatible format: expected a double array!");

    for (int ia = 0; ia < npositions; ++ia) {
        auto& atom = unit_cell.atom(ia);
        double* pos = static_cast<double*>(info.ptr) + 3*ia;
        atom.set_position({pos[0], pos[1], pos[2]});
    }
}

void set_lattice_vectors(Unit_cell& unit_cell,
                         pybind11::buffer l1buf,
                         pybind11::buffer l2buf,
                         pybind11::buffer l3buf)
{
    using namespace pybind11;
    auto to_vector = [](pybind11::buffer in)
                       {
                           buffer_info info = in.request();
                           int npositions = info.shape[0];
                           // checks
                           if (info.ndim != 1) {
                               throw std::runtime_error("Error: set_lattice_vectors expected a vector");
                           }
                           if (info.shape[0] != 3) {
                               throw std::runtime_error("Error: set_lattice_vector ");
                           }
                           if (info.format != format_descriptor<double>::format())
                               throw std::runtime_error("Error: set_lattice_vector: expected type float64");

                           double* ptr = static_cast<double*>(info.ptr);
                           int stride = info.strides[0] / sizeof(double);
                           return vector3d<double>({ptr[0], ptr[stride], ptr[2*stride]});
                       };

    auto l1 = to_vector(l1buf);
    auto l2 = to_vector(l2buf);
    auto l3 = to_vector(l3buf);

    unit_cell.set_lattice_vectors(l1, l2, l3);

}

}  // sirius

#endif /* SET_ATOM_POSITIONS_H */

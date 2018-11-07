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
    unit_cell.update();
}
}  // sirius

#endif /* SET_ATOM_POSITIONS_H */

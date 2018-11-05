#ifndef SET_ATOM_POSITIONS_H
#define SET_ATOM_POSITIONS_H

#include <stdexcept>

#include "Unit_cell/unit_cell.hpp"

void set_atom_positions(Unit_cell& unit_cell, const std::vector<std::vector<double>>& positions)
{
    int npositions = positions.size();
    if (npositions != unit_cell.num_atoms()) {
        throw std::runtime_error("number of positions != number of atoms");
    }
    for (int ia = 0; ia < npositions; ++ia) {
        auto& atom = unit_cell.atom(ia);
        const auto& pos = position[ia];
        if (pos.size() != 3) {
            throw std::runtime_error("wrong number of entries");
        }
        atom.set_position({pos[0], pos[1], pos[2]});
    }
    unit_cell.update();
}

#endif /* SET_ATOM_POSITIONS_H */

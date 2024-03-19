/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef SET_ATOM_POSITIONS_H
#define SET_ATOM_POSITIONS_H

#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>

#include "unit_cell/unit_cell.hpp"

namespace sirius {
void
set_atom_positions(Unit_cell& unit_cell, pybind11::buffer positions)
{
    using namespace pybind11;

    buffer_info info = positions.request();
    int npositions   = info.shape[0];
    if (info.ndim != 2) {
        RTE_THROW("set_atom_positions: wrong dimension (expected 2d array)");
    }
    if (info.shape[1] != 3) {
        RTE_THROW("set_atom_positions: wrong shape");
    }
    if (npositions != unit_cell.num_atoms()) {
        RTE_THROW("number of positions != number of atoms");
    }
    if (info.format != format_descriptor<double>::format())
        RTE_THROW("Incompatible format: expected a double array!");

    const double* ptr = static_cast<double*>(info.ptr);
    int s0            = info.strides[0] / sizeof(double);
    int s1            = info.strides[1] / sizeof(double);
    for (int ia = 0; ia < npositions; ++ia) {
        auto& atom = unit_cell.atom(ia);
        double x   = *(ptr + s0 * ia + s1 * 0);
        double y   = *(ptr + s0 * ia + s1 * 1);
        double z   = *(ptr + s0 * ia + s1 * 2);
        atom.set_position({x, y, z});
    }
}

pybind11::array_t<double>
atom_positions(Unit_cell& unit_cell)
{
    using namespace pybind11;

    int ntot = 0;
    for (int ia = 0; ia < unit_cell.num_atom_types(); ++ia) {
        ntot += unit_cell.atom_coord(ia).size(0);
    }

    array_t<double> positions;
    positions.resize({ntot, 3});
    auto pos = positions.mutable_unchecked<2>();

    int offset = 0;
    for (int ia = 0; ia < unit_cell.num_atom_types(); ++ia) {
        const auto& atom_coords = unit_cell.atom_coord(ia);
        int n                   = atom_coords.size(0);
        for (int k = 0; k < n; ++k) {
            pos(offset + k, 0) = atom_coords(k, 0);
            pos(offset + k, 1) = atom_coords(k, 1);
            pos(offset + k, 2) = atom_coords(k, 2);
        }
        offset += n;
    }

    return positions;
}

void
set_lattice_vectors(Unit_cell& unit_cell, pybind11::buffer l1buf, pybind11::buffer l2buf, pybind11::buffer l3buf)
{
    using namespace pybind11;
    auto to_vector = [](pybind11::buffer in) {
        buffer_info info = in.request();
        // checks
        if (info.ndim != 1) {
            RTE_THROW("Error: set_lattice_vectors expected a vector");
        }
        if (info.shape[0] != 3) {
            RTE_THROW("Error: set_lattice_vector ");
        }
        if (info.format != format_descriptor<double>::format())
            RTE_THROW("Error: set_lattice_vector: expected type float64");

        double* ptr = static_cast<double*>(info.ptr);
        int stride  = info.strides[0] / sizeof(double);
        return r3::vector<double>({ptr[0], ptr[stride], ptr[2 * stride]});
    };

    auto l1 = to_vector(l1buf);
    auto l2 = to_vector(l2buf);
    auto l3 = to_vector(l3buf);

    unit_cell.set_lattice_vectors(l1, l2, l3);
}
} // namespace sirius

#endif /* SET_ATOM_POSITIONS_H */

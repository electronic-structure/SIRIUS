// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file input.cpp
 *
 *  \brief Contains input parameters structures.
 *
 */

#include "input.hpp"

namespace sirius {

void
Hubbard_input::read(json const& parser)
{
    if (!parser.count("hubbard")) {
        return;
    }
    auto section = parser["hubbard"];

    orthogonalize_hubbard_orbitals_ = section.value("orthogonalize", orthogonalize_hubbard_orbitals_);

    normalize_hubbard_orbitals_ = section.value("normalize", normalize_hubbard_orbitals_);

    simplified_hubbard_correction_ = section.value("simplified", simplified_hubbard_correction_);

    if (section.count("projection_method")) {
        std::string projection_method = parser["hubbard"]["projection_method"].get<std::string>();
        if (projection_method == "file") {
            // they are provided by a external file
            if (parser["hubbard"].count("wave_function_file")) {
                this->wave_function_file_ = parser["hubbard"]["wave_function_file"].get<std::string>();
                this->projection_method_  = 1;
            } else {
                throw std::runtime_error(
                    "The hubbard projection method 'file' requires the option 'wave_function_file' to be defined");
            }
        }

        if (projection_method == "pseudo") {
            this->projection_method_ = 2;
        }
    }

    hubbard_U_plus_V_ = parser["hubbard"].value("hubbard_u_plus_v", hubbard_U_plus_V_);

    auto v = parser["unit_cell"]["atom_types"].get<std::vector<std::string>>();

    auto sec = parser["hubbard"]["local"];

    if (sec.size() == 0) {
        throw std::runtime_error(
            "The Hubbard correction section is defined but contain no information about atoms with U-correction");
    }

    for (int elem = 0; elem < sec.size(); elem++) {
        std::string label;
        label = sec[elem].value("atom_type", label);
        if (species_with_U.count(label)) {
            throw std::runtime_error("U-correction for atom " + label + " has already been defined");
        }

        // check that the atom type is actually defined

        auto found = std::find(v.begin(), v.end(), label);

        if (found == v.end()) {
            throw std::runtime_error("The atom type " + label + " can not be found in the unit cell declaration");
        }

        hubbard_correction_ = true;

        hubbard_orbital_t ho;

        ho.coeff[0] = sec[elem].value("U", ho.coeff[0]);
        ho.coeff[1] = sec[elem].value("J", ho.coeff[1]);
        ho.coeff[2] = sec[elem].value("B", ho.coeff[2]);
        ho.coeff[2] = sec[elem].value("E2", ho.coeff[2]);
        ho.coeff[3] = sec[elem].value("E3", ho.coeff[3]);
        ho.coeff[4] = sec[elem].value("alpha", ho.coeff[4]);
        ho.coeff[5] = sec[elem].value("beta", ho.coeff[5]);

        /* now convert eV in Ha */
        for (int s = 0; s < 6; s++) {
            ho.coeff[s] /= ha2ev;
        }
        ho.l = sec[elem].value("l", ho.l);
        ho.n = sec[elem].value("n", ho.n);
        if (ho.l == -1 || ho.n == -1) {
            std::string level;
            sec[elem].value("hubbard_orbital", level);
            std::map<char, int> const map_l = {{'s', 0}, {'p', 1}, {'d', 2}, {'f', 3}};

            std::istringstream iss(std::string(1, level[0]));
            iss >> ho.n;
            if (ho.n <= 0 || iss.fail()) {
                std::stringstream s;
                s << "wrong principal quantum number : " << std::string(1, level[0]);
                throw std::runtime_error(s.str());
            }
            ho.l = map_l.at(level[1]);
        }

        if (!sec[elem].count("occupancy")) {
            throw std::runtime_error("initial occupancy of the Hubbard orbital is not set");
        }
        ho.occupancy         = sec[elem].value("occupancy", ho.occupancy);
        ho.initial_occupancy = sec[elem].value("initial_occupancy", ho.initial_occupancy);

        int sz    = static_cast<int>(ho.initial_occupancy.size());
        int lmmax = 2 * ho.l + 1;

        if (!(sz == 0 || sz == lmmax || sz == 2 * lmmax)) {
            std::stringstream s;
            s << "wrong size of initial occupacies vector (" << sz << ") for l = " << ho.l;
            throw std::runtime_error(s.str());
        }

        species_with_U[label] = ho;
    }
}

}; // namespace sirius

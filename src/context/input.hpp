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

/** \file input.hpp
 *
 *  \brief Contains input parameters structures.
 *
 *  \todo Some of the parameters belong to SCF ground state mini-app. Mini-app should parse this values itself.
 *  \todo parse atomic coordinates and magnetic field separtely, not as 6D vector.
 */

#ifndef __INPUT_HPP__
#define __INPUT_HPP__

#include <list>
#include "constants.hpp"
#include "SDDK/geometry3d.hpp"
#include "utils/json.hpp"
#include <iostream>

using namespace geometry3d;
using namespace nlohmann;

namespace sirius {

struct Hubbard_input
{
    int number_of_species{0};
    bool hubbard_correction_{false};
    //bool simplified_hubbard_correction_{false};
    //bool orthogonalize_hubbard_orbitals_{false};
    //bool normalize_hubbard_orbitals_{false};
    bool hubbard_U_plus_V_{false};

    /** by default we use the atomic orbitals given in the pseudo potentials */
    //int projection_method_{0};

    struct hubbard_orbital_t
    {
        int l{-1};
        int n{-1};
        std::string level;
        std::array<double, 6> coeff{0, 0, 0, 0, 0, 0};
        double occupancy{0};
        std::vector<double> initial_occupancy;
    };

    //std::string wave_function_file_;
    std::map<std::string, hubbard_orbital_t> species_with_U;

    bool hubbard_correction() const
    {
        return hubbard_correction_;
    }

    void read(json const& parser);
};

}; // namespace sirius

#endif // __INPUT_HPP__

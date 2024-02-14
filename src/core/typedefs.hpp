// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file typedefs.hpp
 *
 *  \brief Contains typedefs, enums and simple descriptors.
 */

#ifndef __TYPEDEFS_HPP__
#define __TYPEDEFS_HPP__

#include <cstdlib>
#include <complex>
#include <cstdint>
#include <vector>
#include <array>
#include <limits>
#include <map>
#include <algorithm>
#include <type_traits>

namespace sirius {

// define type traits that return real type
// general case for real type
template <typename T>
struct Real
{
    using type = T;
};

// special case for complex type
template <typename T>
struct Real<std::complex<T>>
{
    using type = T;
};

template <typename T>
using real_type = typename Real<T>::type;

template <class T>
constexpr bool is_real_v = std::is_same<T, real_type<T>>::value;

/// Spin-blocks of the Hamiltonian.
enum class spin_block_t
{
    /// Non-magnetic case.
    nm,

    /// Up-up block.
    uu,

    /// Down-donw block.
    dd,

    /// Up-down block.
    ud,

    /// Down-up block.
    du
};

/// Type of electronic structure methods.
enum class electronic_structure_method_t
{
    /// Full potential linearized augmented plane waves with local orbitals.
    full_potential_lapwlo,

    /// Pseudopotential (ultrasoft, norm-conserving, PAW).
    pseudopotential
};

/// Type of a function domain.
enum class function_domain_t
{
    /// Spatial domain.
    spatial,
    /// Spectral domain.
    spectral
};

/// Type of relativity treatment in the case of LAPW.
enum class relativity_t
{
    none,

    koelling_harmon,

    zora,

    iora,

    dirac
};

inline relativity_t
get_relativity_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    std::map<std::string, relativity_t> const m = {{"none", relativity_t::none},
                                                   {"koelling_harmon", relativity_t::koelling_harmon},
                                                   {"zora", relativity_t::zora},
                                                   {"iora", relativity_t::iora},
                                                   {"dirac", relativity_t::dirac}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "get_relativity_t(): wrong label of the relativity_t enumerator: " << name__;
        throw std::runtime_error(s.str());
    }
    return m.at(name__);
}

/// Describes radial solution.
struct radial_solution_descriptor
{
    /// Principal quantum number.
    int n;

    /// Angular momentum quantum number.
    int l;

    /// Order of energy derivative.
    int dme;

    /// Energy of the solution.
    double enu;

    /// Automatically determine energy.
    int auto_enu;
};

inline std::ostream&
operator<<(std::ostream& out, radial_solution_descriptor const& rsd)
{
    out << "{l: " << rsd.l << ", n: " << rsd.n << ", enu: " << rsd.enu << ", dme: " << rsd.dme
        << ", auto: " << rsd.auto_enu << "}";
    return out;
}

/// Set of radial solution descriptors, used to construct augmented waves or local orbitals.
using radial_solution_descriptor_set = std::vector<radial_solution_descriptor>;

/// Descriptor of an atom in a list of nearest neighbours for each atom.
/** See sirius::Unit_cell::find_nearest_neighbours() for the details of usage. */
struct nearest_neighbour_descriptor
{
    /// Index of the neighbouring atom.
    int atom_id;

    /// Translation in fractional coordinates.
    std::array<int, 3> translation;

    /// Distance from the central atom.
    double distance;

    /// Vector connecting central atom with the neighbour in Cartesian coordinates.
    std::array<double, 3> rc;
};

struct unit_cell_parameters_descriptor
{
    double a;
    double b;
    double c;
    double alpha;
    double beta;
    double gamma;
};

/// Descriptor of the local-orbital part of the LAPW+lo basis.
struct lo_basis_descriptor
{
    /// Index of atom.
    uint16_t ia;

    /// Index of orbital quantum number \f$ \ell \f$.
    uint8_t l;

    /// Combined lm index.
    uint16_t lm;

    /// Order of the local orbital radial function for the given orbital quantum number \f$ \ell \f$.
    /** All radial functions for the given orbital quantum number \f$ \ell \f$ are ordered in the following way:
     *  augmented radial functions come first followed by the local orbital radial function. */
    uint8_t order;

    /// Index of the local orbital radial function.
    uint8_t idxrf;
};

template <typename T>
struct spheric_function_set_ptr_t
{
    T* ptr{nullptr};
    int lmmax{0};
    int nrmtmax{0};
    int num_atoms{0};

    spheric_function_set_ptr_t()
    {
    }

    spheric_function_set_ptr_t(T* ptr__, int lmmax__, int nrmtmax__, int num_atoms__)
        : ptr{ptr__}
        , lmmax{lmmax__}
        , nrmtmax{nrmtmax__}
        , num_atoms{num_atoms__}
    {
    }
};

template <typename T>
struct smooth_periodic_function_ptr_t
{
    T* ptr{nullptr};
    int size_x{0};
    int size_y{0};
    int size_z{0};
    /* if offset_z is negative, FFT buffer is not distributed */
    /* if offset_z >= 0. FFT buffer is treated as distributed and size_z is a local size along z-dimension */
    int offset_z{0};

    smooth_periodic_function_ptr_t()
    {
    }

    smooth_periodic_function_ptr_t(T* ptr__, int size_x__, int size_y__, int size_z__, int offset_z__)
        : ptr{ptr__}
        , size_x{size_x__}
        , size_y{size_y__}
        , size_z{size_z__}
        , offset_z{offset_z__}
    {
    }
};

/// Describe external pointers to periodic function.
/** In case when data is allocated by the calling code, the pointers to muffin-tin and real-space grids
    can be passed to Periodic_function to avoid allocation on the SIRIUS side.*/
template <typename T>
struct periodic_function_ptr_t
{
    spheric_function_set_ptr_t<T> mt;
    smooth_periodic_function_ptr_t<T> rg;

    periodic_function_ptr_t()
    {
    }

    periodic_function_ptr_t(spheric_function_set_ptr_t<T> mt__, smooth_periodic_function_ptr_t<T> rg__)
        : mt{mt__}
        , rg{rg__}
    {
    }
};

} // namespace sirius

#endif // __TYPEDEFS_HPP__

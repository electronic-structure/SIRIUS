// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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

/** \file hubbard_matrix.hpp
 *
 *  \brief Base class for Hubbard occupancy and potential matrices.
 */

#ifndef __HUBBARD_MATRIX_HPP__
#define __HUBBARD_MATRIX_HPP__

#include "context/simulation_context.hpp"

namespace sirius {

class Hubbard_matrix
{
  protected:
    Simulation_context& ctx_;
    std::vector<sddk::mdarray<double_complex, 3>> local_;
    std::vector<sddk::mdarray<double_complex, 3>> nonlocal_;
    std::vector<std::pair<int, int>> atomic_orbitals_;
    std::vector<int> offset_;

  public:
    Hubbard_matrix(Simulation_context& ctx__);

    /// Retrieve or set elements of the Hubbard matrix.
    /** This functions helps retrieving or setting up the hubbard occupancy
     *  tensors from an external tensor. Retrieving it is done by specifying
     *  "get" in the first argument of the method while setting it is done
     *  with the parameter set up to "set". The second parameter is the
     *  output pointer and the last parameter is the leading dimension of the
     *  tensor.
     *
     *  The returned result has the same layout than SIRIUS layout, * i.e.,
     *  the harmonic orbitals are stored from m_z = -l..l. The occupancy
     *  matrix can also be accessed through the method occupation_matrix()
     *
     * \param [in]    what String to set to "set" for initializing sirius ccupancy tensor and "get" for retrieving it.
     * \param [inout] occ  Pointer to external occupancy tensor.
     * \param [in]    ld   Leading dimension of the outside tensor.
     * \return return the occupancy matrix if the first parameter is set to "get". */
    void access(std::string const& what__, double_complex* ptr__, int ld__);

    void print_local(int ia__, std::ostream& out__) const;

    void print_nonlocal(int idx__, std::ostream& out__) const;

    void zero();

    sddk::mdarray<double_complex, 3>& local(int ia__)
    {
        return local_[ia__];
    }

    const std::vector<std::pair<int, int>>& atomic_orbitals() const
    {
        return atomic_orbitals_;
    }

    const std::pair<int, int>& atomic_orbitals(const int idx__) const
    {
        return atomic_orbitals_[idx__];
    }

    const int offset(const int idx__) const
    {
        return offset_[idx__];
    }

    const std::vector<int>& offset() const
    {
        return offset_;
    }

    sddk::mdarray<double_complex, 3> const& local(int ia__) const
    {
        return local_[ia__];
    }

    /// return a vector containing the occupation numbers for each atomic orbitals
    auto& local() const
    {
        return local_;
    }

    sddk::mdarray<double_complex, 3>& nonlocal(int idx__)
    {
        return nonlocal_[idx__];
    }

    sddk::mdarray<double_complex, 3> const& nonlocal(int idx__) const
    {
        return nonlocal_[idx__];
    }

    auto const& ctx() const
    {
        return ctx_;
    }

    const int find_orbital_index(const int ia__, const int n__, const int l__) const
    {
        int at_lvl = 0;
        for (at_lvl = 0; at_lvl < static_cast<int>(atomic_orbitals_.size()); at_lvl++) {
            int lo_ind  = atomic_orbitals_[at_lvl].second;
            int atom_id = atomic_orbitals_[at_lvl].first;

            if ((atomic_orbitals_[at_lvl].first == ia__) &&
                (ctx_.unit_cell().atom(atom_id).type().lo_descriptor_hub(lo_ind).n() == n__) &&
                (ctx_.unit_cell().atom(atom_id).type().lo_descriptor_hub(lo_ind).l == l__))
                break;
        }

        if (at_lvl == static_cast<int>(atomic_orbitals_.size())) {
            std::cout << "atom: " << ia__ << "n: " << n__ << ", l: " << l__ << std::endl;
            RTE_THROW("Found an arbital that is not listed\n");
        }
        return at_lvl;
    }
};

inline void
copy(Hubbard_matrix const& src__, Hubbard_matrix& dest__)
{
    for (int at_lvl = 0; at_lvl < static_cast<int>(src__.atomic_orbitals().size()); at_lvl++) {
        ::sddk::copy(src__.local(at_lvl), dest__.local(at_lvl));
    }
    for (int i = 0; i < static_cast<int>(src__.ctx().cfg().hubbard().nonlocal().size()); i++) {
        ::sddk::copy(src__.nonlocal(i), dest__.nonlocal(i));
    }
}

} // namespace sirius

#endif

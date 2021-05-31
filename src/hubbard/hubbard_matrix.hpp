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

class Hubbard_matrix {
  protected:
    Simulation_context& ctx_;
    std::vector<sddk::mdarray<double_complex, 3>> local_;
    //sddk::mdarray<double_complex, 4> data_;
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

    //sddk::mdarray<double_complex, 4>& data()
    //{
    //    return data_;
    //}

    //sddk::mdarray<double_complex, 4> const& data() const
    //{
    //    return data_;
    //}

    sddk::mdarray<double_complex, 3>& local(int ia__)
    {
        return local_[ia__];
    }

    sddk::mdarray<double_complex, 3> const& local(int ia__) const
    {
        return local_[ia__];
    }

    void zero()
    {
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            if (ctx_.unit_cell().atom(ia).type().hubbard_correction()) {
                local_[ia].zero();
            }
        }
    }

    auto const& ctx() const
    {
        return ctx_;
    }
};

inline void copy(Hubbard_matrix const& src__, Hubbard_matrix& dest__)
{
    for (int ia = 0; ia < src__.ctx().unit_cell().num_atoms(); ia++) {
        if (src__.ctx().unit_cell().atom(ia).type().hubbard_correction()) {
            copy(src__.local(ia), dest__.local(ia));
        }
    }
}

}

#endif

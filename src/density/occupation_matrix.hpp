// Copyright (c) 2013-2020 Mathieu Taillefumier, Mthieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file occupation_matrix.hpp
 *
 *  \brief Occupation matrix of the LDA+U method.
 */

#include "SDDK/memory.hpp"
#include "SDDK/wf_inner.hpp"
#include "simulation_context.hpp"
#include "k_point/k_point.hpp"

namespace sirius {

class Occupation_matrix {
  private:
    Simulation_context& ctx_;
    sddk::mdarray<double_complex, 4> data_;
  public:
    Occupation_matrix(Simulation_context& ctx__);

    void add_k_point_contribution(K_point& kp__);

    /// Retrieve or set the Hubbard occupancies.
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
    void access(std::string const& what__, double_complex* occ__, int ld__);

    /** The initial occupancy is calculated following Hund rules. We first
     *  fill the d (f) states according to the hund's rules and with majority
     *  spin first and the remaining electrons distributed among the minority states. */
    void init();

    void print_occupancies() const;

    sddk::mdarray<double_complex, 4>& data()
    {
        return data_;
    }

    sddk::mdarray<double_complex, 4> const& data() const
    {
        return data_;
    }

    void zero()
    {
        data_.zero();
    }

    void reduce()
    {
        if (data_.size()) {
        /* global reduction over k points */
            ctx_.comm_k().allreduce(data_.at(memory_t::host), static_cast<int>(data_.size()));
        }
    }
};

inline void copy(Occupation_matrix const& src__, Occupation_matrix& dest__)
{
    copy(src__.data(), dest__.data());
}

} // namespace

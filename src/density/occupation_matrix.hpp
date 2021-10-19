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
#include "k_point/k_point.hpp"
#include "hubbard/hubbard_matrix.hpp"

namespace sirius {

class Occupation_matrix : public Hubbard_matrix
{
  private:
    std::map<vector3d<int>, sddk::mdarray<double_complex, 3>> occ_mtrx_T_;

  public:
    Occupation_matrix(Simulation_context& ctx__);

    template <typename T>
    void add_k_point_contribution(K_point<T>& kp__);

    /** The initial occupancy is calculated following Hund rules. We first
     *  fill the d (f) states according to the hund's rules and with majority
     *  spin first and the remaining electrons distributed among the minority states. */
    void init();

    void reduce()
    {
        if (!ctx_.hubbard_correction())
            return;

        /* global reduction over k points */
        for (int at_lvl = 0; at_lvl < (int)this->local_.size(); at_lvl++) {
            const int ia     = atomic_orbitals_[at_lvl].first;
            auto const& atom = ctx_.unit_cell().atom(ia);
            if (atom.type().lo_descriptor_hub(atomic_orbitals_[at_lvl].second).use_for_calculation()) {
                ctx_.comm().allreduce(this->local(at_lvl).at(memory_t::host),
                                      static_cast<int>(this->local(at_lvl).size()));
            }
        }

        // we must reduce occ_mtrx_T_ not nonlocal (it non zero only after symmetrization)
        for (auto& T : this->occ_mtrx_T_) {
            ctx_.comm().allreduce(T.second.at(memory_t::host), static_cast<int>(T.second.size()));
        }

        // std::cout  << "Before : " << this->nonlocal(0)(0, 0, 0) << std::endl;

        // for (int link = 0; link < (int)ctx_.cfg().hubbard().nonlocal().size(); link++) {
        //   ctx_.comm().allreduce(this->nonlocal_[link].at(memory_t::host),
        //   static_cast<int>(this->nonlocal_[link].size()));
        // }

        // std::cout  << "After : " << this->nonlocal(0)(0, 0, 0) << std::endl;
    }

    void zero()
    {
        Hubbard_matrix::zero();
        for (auto& e : occ_mtrx_T_) {
            e.second.zero();
        }
    }

    void symmetrize();

    void print_occupancies(int verbosity__) const;
};

} // namespace sirius

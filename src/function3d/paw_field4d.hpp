// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file paw_field4d.hpp
 *
 *  \brief Contains definition and implementation of PAW_field4D class.
 */

#ifndef __PAW_FIELD4D_HPP__
#define __PAW_FIELD4D_HPP__

#include "function3d/spheric_function_set.hpp"

namespace sirius {

/// PAW density and potential storage.
/** In PAW density and potential are represented with two counterpart: all-electron (AE) and pseudo (PS) */
template <typename T>
class PAW_field4D
{
  private:
    /// Unit cell.
    Unit_cell const& uc_;
    /// All-electron part.
    std::array<Spheric_function_set<T>, 4> ae_components_;
    /// Pseudo-potential part.
    std::array<Spheric_function_set<T>, 4> ps_components_;
    /* copy constructor is forbidden */
    PAW_field4D(PAW_field4D const& src__) = delete;
    /* copy assignment operator is forbidden */
    PAW_field4D& operator=(PAW_field4D const& src__) = delete;
  public:
    /// Constructor
    PAW_field4D(Unit_cell const& uc__, bool is_global__)
        : uc_{uc__}
    {
        if (!uc__.num_paw_atoms()) {
            return;
        }

        auto ptr = (is_global__) ? nullptr : &uc__.spl_num_paw_atoms();

        for (int j = 0; j < uc__.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j] = Spheric_function_set<T>(uc__, uc__.paw_atoms(),
                    [&uc__](int ia){return lmax_t(2 * uc__.atom(ia).type().indexr().lmax());}, ptr);
            ps_components_[j] = Spheric_function_set<T>(uc__, uc__.paw_atoms(),
                    [&uc__](int ia){return lmax_t(2 * uc__.atom(ia).type().indexr().lmax());}, ptr);
        }
    }

    void sync()
    {
        for (int j = 0; j < uc_.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j].sync(uc_.spl_num_paw_atoms());
            ps_components_[j].sync(uc_.spl_num_paw_atoms());
        }
    }

    void zero(int ia__)
    {
        for (int j = 0; j < uc_.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j][ia__].zero(sddk::memory_t::host);
            ps_components_[j][ia__].zero(sddk::memory_t::host);
        }
    }

    void zero()
    {
        for (int j = 0; j < uc_.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j].zero();
            ps_components_[j].zero();
        }
    }

    auto& ae_component(int i__)
    {
        return ae_components_[i__];
    }

    auto const& ae_component(int i__) const
    {
        return ae_components_[i__];
    }

    auto& ps_component(int i__)
    {
        return ps_components_[i__];
    }

    auto const& ps_component(int i__) const
    {
        return ps_components_[i__];
    }

    auto const& unit_cell() const
    {
        return uc_;
    }
};


} // namespace sirius

#endif

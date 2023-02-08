// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file paw_density.hpp
 *
 *  \brief Contains definition and implementation of helper class sirius::paw_density.
 */

#ifndef __PAW_DENSITY_HPP__
#define __PAW_DENSITY_HPP__

namespace sirius {

template <typename T>
class paw_density
{
  private:
    Unit_cell const* uc_{nullptr};
    std::array<Spheric_function_set<T>, 4> ae_density_;
    std::array<Spheric_function_set<T>, 4> ps_density_;

    /* copy constructor is forbidden */
    paw_density(paw_density const& src__) = delete;
    /* copy assignment operator is forbidden */
    paw_density& operator=(paw_density const& src__) = delete;

  public:

    paw_density(Unit_cell const& uc__)
        : uc_{&uc__}
    {
        if (!uc__.num_paw_atoms()) {
            return;
        }

        for (int j = 0; j < uc__.parameters().num_mag_dims() + 1; j++) {
            ae_density_[j] = Spheric_function_set<T>(uc__, uc__.paw_atoms(),
                    [&uc__](int ia){return 2 * uc__.atom(ia).type().indexr().lmax();}, &uc__.spl_num_paw_atoms());
            ps_density_[j] = Spheric_function_set<T>(uc__, uc__.paw_atoms(),
                    [&uc__](int ia){return 2 * uc__.atom(ia).type().indexr().lmax();}, &uc__.spl_num_paw_atoms());
        }
    }

    void zero(int ia__)
    {
        for (int j = 0; j < uc_->parameters().num_mag_dims() + 1; j++) {
            ae_density_[j][ia__].zero(sddk::memory_t::host);
            ps_density_[j][ia__].zero(sddk::memory_t::host);
        }
    }

    auto& ae_density(int i__, int ja__)
    {
        return ae_density_[i__][ja__];
    }

    auto const& ae_density(int i__, int ja__) const
    {
        return ae_density_[i__][ja__];
    }

    auto& ps_density(int i__, int ja__)
    {
        return ps_density_[i__][ja__];
    }

    auto const& ps_density(int i__, int ja__) const
    {
        return ps_density_[i__][ja__];
    }

    auto const& unit_cell() const
    {
        return *uc_;
    }
};

} // namespace sirius

#endif

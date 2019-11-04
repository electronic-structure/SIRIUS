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

class paw_density
{
  private:
    Simulation_context* ctx_{nullptr};
    sddk::mdarray<Spheric_function<function_domain_t::spectral, double>, 2> ae_density_;
    sddk::mdarray<Spheric_function<function_domain_t::spectral, double>, 2> ps_density_;
    std::vector<int> atoms_;

    paw_density(paw_density const& src__) = delete;
    paw_density& operator=(paw_density const& src__) = delete;

  public:
    paw_density()
    {
    }

    paw_density(Simulation_context& ctx__)
        : ctx_(&ctx__)
    {
        auto& uc = ctx__.unit_cell();
        if (!uc.num_paw_atoms()) {
            return;
        }

        using sf = Spheric_function<function_domain_t::spectral, double>;

        ae_density_ = sddk::mdarray<sf, 2>(ctx__.num_mag_dims() + 1, uc.spl_num_paw_atoms().local_size());
        ps_density_ = sddk::mdarray<sf, 2>(ctx__.num_mag_dims() + 1, uc.spl_num_paw_atoms().local_size());

        for (int i = 0; i < uc.spl_num_paw_atoms().local_size(); i++) {
            int ia_paw      = uc.spl_num_paw_atoms(i);
            int ia          = uc.paw_atom_index(ia_paw);
            auto& atom      = uc.atom(ia);
            auto& atom_type = atom.type();

            int l_max      = 2 * atom_type.indexr().lmax_lo();
            int lm_max_rho = utils::lmmax(l_max);

            atoms_.push_back(ia);

            /* allocate density arrays */
            for (int j = 0; j < ctx__.num_mag_dims() + 1; j++) {
                ae_density_(j, i) = sf(lm_max_rho, atom.radial_grid());
                ps_density_(j, i) = sf(lm_max_rho, atom.radial_grid());
            }
        }
    }

    paw_density(paw_density&& src__) = default;
    paw_density& operator=(paw_density&& src__) = default;

    Spheric_function<function_domain_t::spectral, double>& ae_density(int i, int j)
    {
        return ae_density_(i, j);
    }

    Spheric_function<function_domain_t::spectral, double> const& ae_density(int i, int j) const
    {
        return ae_density_(i, j);
    }

    Spheric_function<function_domain_t::spectral, double>& ps_density(int i, int j)
    {
        return ps_density_(i, j);
    }

    Spheric_function<function_domain_t::spectral, double> const& ps_density(int i, int j) const
    {
        return ps_density_(i, j);
    }

    Simulation_context const& ctx() const
    {
        return *ctx_;
    }
};

} // namespace sirius

#endif

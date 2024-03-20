/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
    /// Text label of the field.
    std::string label_;
    /// All-electron part.
    std::array<Spheric_function_set<T, paw_atom_index_t>, 4> ae_components_;
    /// Pseudo-potential part.
    std::array<Spheric_function_set<T, paw_atom_index_t>, 4> ps_components_;
    /* copy constructor is forbidden */
    PAW_field4D(PAW_field4D const& src__) = delete;
    /* copy assignment operator is forbidden */
    PAW_field4D&
    operator=(PAW_field4D const& src__) = delete;

  public:
    /// Constructor
    PAW_field4D(std::string label__, Unit_cell const& uc__, bool is_global__)
        : uc_{uc__}
        , label_{label__}
    {
        if (!uc__.num_paw_atoms()) {
            return;
        }

        auto ptr = (is_global__) ? nullptr : &uc__.spl_num_paw_atoms();

        for (int j = 0; j < uc__.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j] = Spheric_function_set<T, paw_atom_index_t>(
                    label__ + std::to_string(j), uc__, uc__.paw_atoms(),
                    [&uc__](int ia) { return lmax_t(2 * uc__.atom(ia).type().indexr().lmax()); }, ptr);
            ps_components_[j] = Spheric_function_set<T, paw_atom_index_t>(
                    label__ + std::to_string(j), uc__, uc__.paw_atoms(),
                    [&uc__](int ia) { return lmax_t(2 * uc__.atom(ia).type().indexr().lmax()); }, ptr);
        }
    }

    void
    sync()
    {
        for (int j = 0; j < uc_.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j].sync(uc_.spl_num_paw_atoms());
            ps_components_[j].sync(uc_.spl_num_paw_atoms());
        }
    }

    void
    zero(int ia__)
    {
        for (int j = 0; j < uc_.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j][ia__].zero(memory_t::host);
            ps_components_[j][ia__].zero(memory_t::host);
        }
    }

    void
    zero()
    {
        for (int j = 0; j < uc_.parameters().num_mag_dims() + 1; j++) {
            ae_components_[j].zero();
            ps_components_[j].zero();
        }
    }

    auto&
    ae_component(int i__)
    {
        return ae_components_[i__];
    }

    auto const&
    ae_component(int i__) const
    {
        return ae_components_[i__];
    }

    auto&
    ps_component(int i__)
    {
        return ps_components_[i__];
    }

    auto const&
    ps_component(int i__) const
    {
        return ps_components_[i__];
    }

    auto const&
    unit_cell() const
    {
        return uc_;
    }

    template <typename T_>
    friend T_
    inner(PAW_field4D<T_> const& x__, PAW_field4D<T_> const& y__);
};

template <typename T>
T
inner(PAW_field4D<T> const& x__, PAW_field4D<T> const& y__)
{
    T result{0};
    for (int j = 0; j < x__.uc_.parameters().num_mag_dims() + 1; j++) {
        result += inner(x__.ae_components_[j], y__.ae_components_[j]);
        result += inner(x__.ps_components_[j], y__.ps_components_[j]);
    }
    return result;
}

} // namespace sirius

#endif

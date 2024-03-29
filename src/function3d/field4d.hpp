/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file field4d.hpp
 *
 *  \brief Base class for sirius::Density and sirius::Potential.
 */

#ifndef __FIELD4D_HPP__
#define __FIELD4D_HPP__

#include <memory>
#include <array>
#include <stdexcept>
#include "core/memory.hpp"
#include "function3d/periodic_function.hpp"
#include "core/typedefs.hpp"

namespace sirius {

/// Four-component function consisting of scalar and vector parts.
/** This class is used to represents density/magnetisation and potential/magentic filed of the system. */
class Field4D
{
  private:
    /// Four components of the field: scalar, vector_z, vector_x, vector_y
    std::array<std::unique_ptr<Periodic_function<double>>, 4> components_;

  protected:
    Simulation_context& ctx_;

  public:
    /// Constructor.
    Field4D(Simulation_context& ctx__, lmax_t lmax__,
            std::array<periodic_function_ptr_t<double> const*, 4> ptr__ = {nullptr, nullptr, nullptr, nullptr})
        : ctx_{ctx__}
    {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            smooth_periodic_function_ptr_t<double> const* ptr_rg{nullptr};
            spheric_function_set_ptr_t<double> const* ptr_mt{nullptr};
            if (ptr__[i] && ptr__[i]->rg.ptr) {
                ptr_rg = &ptr__[i]->rg;
            }
            if (ptr__[i] && ptr__[i]->mt.ptr) {
                ptr_mt = &ptr__[i]->mt;
            }
            if (ctx_.full_potential()) {
                /* allocate with global MT part */
                components_[i] = std::make_unique<Periodic_function<double>>(
                        ctx_, [&](int ia) { return lmax__; }, nullptr, ptr_rg, ptr_mt);
            } else {
                components_[i] = std::make_unique<Periodic_function<double>>(ctx_, ptr_rg);
            }
        }
    }

    /// Return scalar part of the field.
    inline auto&
    scalar()
    {
        return *(components_[0]);
    }

    /// Return scalar part of the field.
    inline auto const&
    scalar() const
    {
        return *(components_[0]);
    }

    /// Return component of the vector part of the field.
    inline auto&
    vector(int i)
    {
        RTE_ASSERT(i >= 0 && i <= 2);
        return *(components_[i + 1]);
    }

    /// Return component of the vector part of the field.
    inline auto const&
    vector(int i) const
    {
        RTE_ASSERT(i >= 0 && i <= 2);
        return *(components_[i + 1]);
    }

    inline auto&
    component(int i)
    {
        RTE_ASSERT(i >= 0 && i <= 3);
        return *(components_[i]);
    }

    /// Throws error in case of invalid access.
    inline auto&
    component_raise(int i)
    {
        if (components_[i] == nullptr) {
            throw std::runtime_error("invalid access");
        }
        return *(components_[i]);
    }

    inline auto const&
    component(int i) const
    {
        RTE_ASSERT(i >= 0 && i <= 3);
        return *(components_[i]);
    }

    inline void
    zero()
    {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            component(i).zero();
        }
    }

    inline void
    fft_transform(int direction__)
    {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            component(i).rg().fft_transform(direction__);
        }
    }

    inline auto&
    ctx()
    {
        return ctx_;
    }

    inline auto const&
    ctx() const
    {
        return ctx_;
    }

    auto
    mt_components()
    {
        std::vector<Spheric_function_set<double, atom_index_t>*> result;
        result.push_back(&components_[0]->mt());
        switch (ctx_.num_mag_dims()) {
            case 1: {
                /* z-component */
                result.push_back(&components_[1]->mt());
                break;
            }
            case 3: {
                /* x-component */
                result.push_back(&components_[2]->mt());
                /* y-component */
                result.push_back(&components_[3]->mt());
                /* z-component */
                result.push_back(&components_[1]->mt());
                break;
            }
        }
        return result;
    }

    auto
    pw_components()
    {
        std::vector<Smooth_periodic_function<double>*> result;
        result.push_back(&components_[0]->rg());
        switch (ctx_.num_mag_dims()) {
            case 1: {
                /* z-component */
                result.push_back(&components_[1]->rg());
                break;
            }
            case 3: {
                /* x-component */
                result.push_back(&components_[2]->rg());
                /* y-component */
                result.push_back(&components_[3]->rg());
                /* z-component */
                result.push_back(&components_[1]->rg());
                break;
            }
        }
        return result;
    }
};

} // namespace sirius

#endif

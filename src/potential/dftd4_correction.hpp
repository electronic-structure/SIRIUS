/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __DFTD4_CORRECTION_HPP__
#define __DFTD4_CORRECTION_HPP__

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>

#ifdef SIRIUS_USE_DFTD4
#include <dftd4.h>
#else
typedef void* dftd4_error;
typedef void* dftd4_structure;
typedef void* dftd4_model;
typedef void* dftd4_param;
#endif
#include "core/splindex.hpp"
#include "unit_cell/unit_cell.hpp"
#include "context/simulation_context.hpp"
#include "xc_functional.hpp"

namespace sirius {
class dftd4
{
  private:
    Simulation_context& ctx_;
    Unit_cell& unit_cell_;
    // cartesian coordinates of the atoms in bohr
    std::vector<double> atom_positions_;
    // lattice vectors in bohr
    std::vector<double> lattice_vectors_;
    // atomic forces
    mdarray<double, 2> forces_;
    // stress tensor
    r3::matrix<double> stress_;
    // total number of electrons of each atom
    std::vector<int> z_charges_;
    // functional. Unfortunately dft-d3 library does not use libxc numbering scheme
    std::string xc_method_;
    // energy
    double energy_{0.0};

    dftd4_error error_{nullptr};
    dftd4_structure mol_{nullptr};
    dftd4_model disp_{nullptr};
    dftd4_param param_{nullptr};

  public:
    dftd4(Simulation_context& ctx__, Unit_cell& unit_cell__);

    /* forbid assignment operator */
    dftd4&
    operator=(const dftd4& src) = delete;

    /* forbid copy constructor */
    dftd4(const dftd4& src) = delete;

    dftd4(dftd4* src__)
        : ctx_(src__->ctx_)
        , unit_cell_(src__->unit_cell_)
    {
#if defined(SIRIUS_USE_DFTD3)
        this->lattice_vectors_ = src__->lattice_vectors_;
        this->forces_          = std::move(src__->forces_);
        this->stress_          = std::move(src__->stress_);
        this->z_charges_       = std::move(src__->z_charges_);
        this->xc_method_       = src__->xc_method_;
        this->error_           = src__->error_;
        this->param_           = src__->param_;
        this->mol_             = src__->mol_;
        this->disp_            = src__->disp_;
        src__->error_          = nullptr;
        src__->param_          = nullptr;
        src__->mol_            = nullptr;
        src__->disp_           = nullptr;
#endif
    }

    dftd4(dftd4&& src__)
        : ctx_(src__.ctx_)
        , unit_cell_(src__.unit_cell_)
    {
#if defined(SIRIUS_USE_DFTD4)
        this->lattice_vectors_ = src__.lattice_vectors_;
        this->forces_          = std::move(src__.forces_);
        this->stress_          = std::move(src__.stress_);
        this->z_charges_       = std::move(src__.z_charges_);
        this->xc_method_       = src__.xc_method_;
        this->error_           = src__.error_;
        this->param_           = src__.param_;
        this->mol_             = src__.mol_;
        this->disp_            = src__.disp_;
        src__.error_           = nullptr;
        src__.param_           = nullptr;
        src__.mol_             = nullptr;
        src__.disp_            = nullptr;
#endif
    }

    void
    update_dftd4_ctx();
    void
    calculate_energy_forces_stress();
    ~dftd4()
    {
        if (!ctx_.cfg().parameters().dftd4_correction()) {
            return;
        }
#ifdef SIRIUS_USE_DFTD4
        if (error_ != nullptr) {
            dftd4_delete_error(&error_);
        }
        if (mol_ != nullptr) {
            dftd4_delete_structure(&mol_);
        }
        if (disp_ != nullptr) {
            dftd4_delete_model(&disp_);
        }
        if (param_ != nullptr) {
            dftd4_delete_param(&param_);
        }
        atom_positions_.clear();
        lattice_vectors_.clear();
#endif
    }

    double
    energy() const
    {
        return energy_;
    }

    inline auto const&
    forces() const
    {
        return forces_;
    }
    auto&
    stress() const
    {
        return stress_;
    }
};
} // namespace sirius
#endif

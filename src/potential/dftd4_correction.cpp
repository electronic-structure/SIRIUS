/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "dftd4_correction.hpp"

namespace sirius {
dftd4::dftd4(Simulation_context& ctx__, Unit_cell& unit_cell__)
    : ctx_(ctx__)
    , unit_cell_(unit_cell__)
{
#ifndef SIRIUS_USE_DFTD4
    RTE_THROW("SIRIUS is compiled without dft-d4 support");
#endif
    if (!ctx_.cfg().parameters().dftd4_correction()) {
        return;
    }
    xc_method_ = ctx_.cfg().dftd4().method();
    update_dftd4_ctx();
}

void
dftd4::update_dftd4_ctx()
{
    if (!ctx_.cfg().parameters().dftd4_correction()) {
        return;
    }
#ifdef SIRIUS_USE_DFTD4
    auto lat = unit_cell_.lattice_vectors();
    forces_  = mdarray<double, 2>({3, unit_cell_.num_atoms()});
    lattice_vectors_.resize(9);
    z_charges_.resize(unit_cell_.num_atoms());
    atom_positions_.resize(3 * unit_cell_.num_atoms());
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            lattice_vectors_[3 * i + j] = lat(i, j);
        }
    }

    for (int i = 0; i < unit_cell_.num_atoms(); i++) {
        z_charges_[i] = unit_cell_.atom(i).type().zn();
    }

    // calculate the cartesian coordinates of the atoms in bohr. SIRIUS works with fractional coordinates.
    for (int i = 0; i < unit_cell_.num_atoms(); i++) {
        const auto& coord          = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(i).position());
        atom_positions_[3 * i]     = coord[0];
        atom_positions_[3 * i + 1] = coord[1];
        atom_positions_[3 * i + 2] = coord[2];
    }

    // we only need to initialize it once for the entire simulation. Updating the context does not affect it at all
    if (error_ == nullptr) {
        error_ = dftd4_new_error();
    }

    std::string buffer_error;
    buffer_error.resize(1024, 0);

    if (mol_ == nullptr) {
        const bool periodic_[3] = {true, true, true};
        mol_ = dftd4_new_structure(error_, ctx_.unit_cell().num_atoms(), z_charges_.data(), atom_positions_.data(),
                                   nullptr, lattice_vectors_.data(), periodic_);
    } else {
        dftd4_update_structure(error_, mol_, atom_positions_.data(), lattice_vectors_.data());
    }

    if (dftd4_check_error(error_)) {
        dftd4_get_error(error_, buffer_error.data(), nullptr);
        RTE_THROW(buffer_error);
    }

    if (disp_ != nullptr) {
        dftd4_delete_model(&disp_);
    }
    disp_ = dftd4_new_d4_model(error_, mol_);

    if (dftd4_check_error(error_)) {
        dftd4_get_error(error_, buffer_error.data(), nullptr);
        RTE_THROW(buffer_error);
    }

    if (param_ != nullptr) {
        dftd4_delete_param(&param_);
    }
    if (ctx_.cfg().dftd4().method() == "none") {
        RTE_THROW("DFT-D4: The method parameter in the dftd4 section of the input file should\n"
                  "be set and match the XC functional. See the simple-dftd4 documentation for more details");
    }

    if (ctx_.cfg().dftd4().damping() == "rational") {
        if (ctx_.cfg().dftd4().damping_values() == "auto") {
            param_ = dftd4_load_rational_damping(error_, ctx_.cfg().dftd4().method().data(),
                                                 ctx_.cfg().dftd4().three_body());
        } else {
            param_ = dftd4_new_rational_damping(
                    error_, ctx_.cfg().dftd4().parameters().s6(), ctx_.cfg().dftd4().parameters().s8(),
                    ctx_.cfg().dftd4().parameters().s9(), ctx_.cfg().dftd4().parameters().a1(),
                    ctx_.cfg().dftd4().parameters().a2(), ctx_.cfg().dftd4().parameters().alp());
        }
    }

    if (dftd4_check_error(error_)) {
        dftd4_get_error(error_, buffer_error.data(), nullptr);
        RTE_THROW(buffer_error);
    }

    calculate_energy_forces_stress();
#endif
}

void
dftd4::calculate_energy_forces_stress()
{
    if (!ctx_.cfg().parameters().dftd4_correction()) {
        return;
    }
#ifdef SIRIUS_USE_DFTD4
    PROFILE("sirius::Potential::dft_d4");
    // do the actual calculations. It is needed only once since this correction does not depend on the density.
    std::vector<double> forces_tmp(unit_cell_.num_atoms() * 3);
    std::vector<double> virial_tmp(9);
    dftd4_get_dispersion(error_, mol_, disp_, param_, &energy_, forces_tmp.data(), virial_tmp.data());

    // the library returns the gradients NOT the forces. We need to multiply by -1 to get the forces
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        forces_(0, ia) = -forces_tmp[3 * ia];
        forces_(1, ia) = -forces_tmp[3 * ia + 1];
        forces_(2, ia) = -forces_tmp[3 * ia + 2];
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            stress_(i, j) = -virial_tmp[3 * i + j] / ctx_.unit_cell().omega();
        }
    }
#endif
}
} // namespace sirius

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "dftd3_correction.hpp"

namespace sirius {
dftd3::dftd3(Simulation_context& ctx__, Unit_cell& unit_cell__)
    : ctx_(ctx__)
    , unit_cell_(unit_cell__)
{
#ifndef SIRIUS_USE_DFTD3
    RTE_THROW("SIRIUS is compiled without dft-d3 support");
#endif
    if (!ctx_.cfg().parameters().dftd3_correction())
        return;

    xc_method_ = ctx_.cfg().dftd3().method();

    update_dftd3_ctx();
}

void
dftd3::update_dftd3_ctx()
{
    if (!ctx_.cfg().parameters().dftd3_correction())
        return;
#ifdef SIRIUS_USE_DFTD3
    lattice_vectors_.resize(9);
    atom_positions_.resize(3 * unit_cell_.num_atoms());
    forces_ = mdarray<double, 2>({3, unit_cell_.num_atoms()});
    lattice_vectors_.resize(9);
    z_charges_.resize(unit_cell_.num_atoms());
    z_charges_.resize(unit_cell_.num_atoms());
    auto lat = unit_cell_.lattice_vectors();
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
    if (!error_) {
        error_ = dftd3_new_error();
    }
    std::string buffer_error;
    buffer_error.resize(1024, 0);
    if (!mol_) {
        const bool periodic[3] = {true, true, true};
        mol_ = dftd3_new_structure(error_, ctx_.unit_cell().num_atoms(), z_charges_.data(), atom_positions_.data(),
                                   lattice_vectors_.data(), periodic);
    } else {
        dftd3_update_structure(error_, mol_, atom_positions_.data(), lattice_vectors_.data());
    }

    if (dftd3_check_error(error_)) {
        dftd3_get_error(error_, buffer_error.data(), nullptr);
        RTE_THROW(buffer_error);
    }

    if (disp_ != nullptr) {
        dftd3_delete_model(&disp_);
    }
    disp_ = dftd3_new_d3_model(error_, mol_);

    if (dftd3_check_error(error_)) {
        dftd3_get_error(error_, buffer_error.data(), nullptr);
        RTE_THROW(buffer_error);
    }

    if (param_ != nullptr) {
        dftd3_delete_param(&param_);
    }
    if (ctx_.cfg().dftd3().method() == "none") {
        RTE_THROW("DFT-D3: The method parameter in the dftd3 section of the input file should\n"
                  "be set and match the XC functional. See the simple-dftd3 documentation for more details");
    }

    if (ctx_.cfg().dftd3().damping() == "rational") {
        if (ctx_.cfg().dftd3().damping_values() == "auto") {
            param_ = dftd3_load_rational_damping(error_, ctx_.cfg().dftd3().method().data(),
                                                 ctx_.cfg().dftd3().three_body());
        } else {
            param_ = dftd3_new_rational_damping(
                    error_, ctx_.cfg().dftd3().parameters().s6(), ctx_.cfg().dftd3().parameters().s8(),
                    ctx_.cfg().dftd3().parameters().s9(), ctx_.cfg().dftd3().parameters().a1(),
                    ctx_.cfg().dftd3().parameters().a2(), ctx_.cfg().dftd3().parameters().alp());
        }
    }

    if (ctx_.cfg().dftd3().damping() == "zero") {
        if (ctx_.cfg().dftd3().damping_values() == "auto") {
            param_ = dftd3_load_zero_damping(error_, ctx_.cfg().dftd3().method().data(),
                                             ctx_.cfg().dftd3().three_body());
        } else {
            param_ = dftd3_new_zero_damping(
                    error_, ctx_.cfg().dftd3().parameters().s6(), ctx_.cfg().dftd3().parameters().s8(),
                    ctx_.cfg().dftd3().parameters().s9(), ctx_.cfg().dftd3().parameters().rs6(),
                    ctx_.cfg().dftd3().parameters().rs8(), ctx_.cfg().dftd3().parameters().alp());
        }
    }

    if (ctx_.cfg().dftd3().damping() == "mzero") {
        if (ctx_.cfg().dftd3().damping_values() == "auto") {
            param_ = dftd3_load_mzero_damping(error_, ctx_.cfg().dftd3().method().data(),
                                              ctx_.cfg().dftd3().three_body());
        } else {
            param_ = dftd3_new_mzero_damping(
                    error_, ctx_.cfg().dftd3().parameters().s6(), ctx_.cfg().dftd3().parameters().s8(),
                    ctx_.cfg().dftd3().parameters().s9(), ctx_.cfg().dftd3().parameters().rs6(),
                    ctx_.cfg().dftd3().parameters().rs8(), ctx_.cfg().dftd3().parameters().alp(),
                    ctx_.cfg().dftd3().parameters().beta());
        }
    }

    if (ctx_.cfg().dftd3().damping() == "mrational") {
        if (ctx_.cfg().dftd3().damping_values() == "auto") {
            param_ = dftd3_load_mrational_damping(error_, ctx_.cfg().dftd3().method().data(),
                                                  ctx_.cfg().dftd3().three_body());
        } else {
            param_ = dftd3_new_mrational_damping(
                    error_, ctx_.cfg().dftd3().parameters().s6(), ctx_.cfg().dftd3().parameters().s8(),
                    ctx_.cfg().dftd3().parameters().s9(), ctx_.cfg().dftd3().parameters().a1(),
                    ctx_.cfg().dftd3().parameters().a2(), ctx_.cfg().dftd3().parameters().alp());
        }
    }

    if (ctx_.cfg().dftd3().damping() == "optimizedpower") {
        if (ctx_.cfg().dftd3().damping_values() == "auto") {
            param_ = dftd3_load_optimizedpower_damping(error_, ctx_.cfg().dftd3().method().data(),
                                                       ctx_.cfg().dftd3().three_body());
        } else {
            param_ = dftd3_new_optimizedpower_damping(
                    error_, ctx_.cfg().dftd3().parameters().s6(), ctx_.cfg().dftd3().parameters().s8(),
                    ctx_.cfg().dftd3().parameters().s9(), ctx_.cfg().dftd3().parameters().a1(),
                    ctx_.cfg().dftd3().parameters().a2(), ctx_.cfg().dftd3().parameters().alp(),
                    ctx_.cfg().dftd3().parameters().beta());
        }
    }

    if (dftd3_check_error(error_)) {
        dftd3_get_error(error_, buffer_error.data(), nullptr);
        RTE_THROW(buffer_error);
    }

    calculate_energy_forces_stress();
#endif
}

void
dftd3::calculate_energy_forces_stress()
{
    if (!ctx_.cfg().parameters().dftd3_correction()) {
        return;
    }
#ifdef SIRIUS_USE_DFTD3
    PROFILE("sirius::Potential::dft_d3");
    // do the actual calculations. It is needed only once since this correction does not depend on the density.
    std::vector<double> forces_tmp(unit_cell_.num_atoms() * 3);
    std::vector<double> stress_tmp(9);
    dftd3_get_dispersion(error_, mol_, disp_, param_, &energy_, forces_tmp.data(), stress_tmp.data());

    // the library returns the gradients NOT the forces. We need to multiply by -1 to get the forces
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        forces_(0, ia) = -forces_tmp[3 * ia];
        forces_(1, ia) = -forces_tmp[3 * ia + 1];
        forces_(2, ia) = -forces_tmp[3 * ia + 2];
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            stress_(i, j) = stress_tmp[3 * i + j];
        }
    }
#endif
}
} // namespace sirius

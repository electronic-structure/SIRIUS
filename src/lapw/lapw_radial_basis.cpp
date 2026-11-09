/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file lapw_radial_basis.cpp
 *
 *  \brief Radial basis functions of the LAPW method.
 */

#include "lapw/lapw_radial_basis.hpp"
#include "core/json.hpp"

namespace sirius {

void
lapw_radial_basis_t::save_spherical_potential() const
{
    nlohmann::json dict;
    dict["x"]                   = atom_type_.radial_grid().values();
    dict["z"]                   = atom_type_.zn();
    dict["rmt"]                 = atom_type_.mt_radius();
    dict["spherical_potential"] = spherical_potential_;
    write_json_to_file(dict, "spherical_potential_" + atom_type_.label() + ".json");
}

void
lapw_radial_basis_t::save_radial_functions(std::string const& fname__) const
{
    nlohmann::json dict;
    dict["x"]                   = atom_type_.radial_grid().values();
    dict["z"]                   = atom_type_.zn();
    dict["rmt"]                 = atom_type_.mt_radius();
    dict["spherical_potential"] = spherical_potential_;

    std::vector<double> veff(spherical_potential_.size());
    for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
        veff[ir] = spherical_potential_[ir] + atom_type_.zn() / atom_type_.radial_grid(ir);
    }
    dict["spherical_potential_el"] = veff;

    dict["radial_functions"] = nlohmann::json::array();
    for (int idxrf = 0; idxrf < atom_type_.indexr().size(); idxrf++) {
        std::vector<double> u(atom_type_.num_mt_points());
        std::vector<double> rdudr(atom_type_.num_mt_points());

        for (int ir = 0; ir < atom_type_.num_mt_points(); ir++) {
            u[ir]     = radial_functions_(ir, idxrf, 0);
            rdudr[ir] = radial_functions_(ir, idxrf, 1);
        }

        auto const& rfd = atom_type_.indexr(idxrf);
        dict["radial_functions"].push_back({{"idxrf", idxrf},
                                            {"l", rfd.am.l()},
                                            {"order", rfd.order},
                                            {"idxlo", static_cast<int>(rfd.idxlo)},
                                            {"u", u},
                                            {"rdudr", rdudr}});
    }

    write_json_to_file(dict, fname__);
}

} // namespace sirius

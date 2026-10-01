/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_lapw_atom_input()
{
    nlohmann::json config = {{"parameters", {{"electronic_structure_method", "full_potential_lapwlo"}}}};
    Simulation_context ctx(config);
    auto& type = ctx.unit_cell().add_atom_type("He-site");
    type.set_zn(2);
    type.set_symbol("He");
    auto basis =
            nlohmann::json::array({{{"enu", -0.3}, {"dme", 0}, {"auto", 0}}, {{"enu", -0.3}, {"dme", 1}, {"auto", 0}}});
    nlohmann::json atom = {
            {"name", "Helium input fixture"},
            {"symbol", "He"},
            {"number", 2},
            {"mass", 4.002602},
            {"rmin", 1e-6},
            {"rmt", 1.8},
            {"nrmt", 33},
            {"core", ""},
            {"valence", nlohmann::json::array({{{"basis", basis}}})},
            {"free_atom", {{"radial_grid", {1e-6, 0.1, 1.0, 2.0, 10.0}}, {"density", {5.0, 3.0, 0.1, 0.001, 1e-17}}}}};
    for (bool wrong_charge : {false, true}) {
        auto wrong = atom;
        if (wrong_charge) {
            wrong["number"] = 3;
        } else {
            wrong["symbol"] = "Li";
        }
        bool rejected{false};
        try {
            type.read_input(wrong);
        } catch (std::runtime_error const& error) {
            rejected = std::string(error.what()).find("does not match the requested element") != std::string::npos;
        }
        if (!rejected || type.zn() != 2 || type.symbol() != "He" || type.radial_grid().num_points() != 0) {
            RTE_THROW("Mismatched LAPW atom input was accepted or changed the requested atom type");
        }
    }
    type.read_input(atom);
    if (type.zn() != 2 || type.symbol() != "He" || type.radial_grid().num_points() != 33 ||
        std::abs(type.mt_radius() - 1.8) > 1e-14) {
        RTE_THROW("Matching LAPW atom data did not load");
    }
    // A file-only caller does not have to supply redundant element metadata.
    auto& from_file = ctx.unit_cell().add_atom_type("file-only");
    from_file.read_input(atom);
    if (from_file.zn() != 2 || from_file.symbol() != "He") {
        RTE_THROW("File-only LAPW atom loading regressed");
    }
    return 0;
}

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result = call_test(argv[0], test_lapw_atom_input);
    sirius::finalize();
    return result;
}

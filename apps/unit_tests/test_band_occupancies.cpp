/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_empty_spin_channel()
{
    auto conf = R"({"parameters":{"electronic_structure_method":"pseudopotential","num_mag_dims":1,
        "fixed_mag":11,"num_bands":12,"smearing":"gaussian","smearing_width":0.01,
        "gk_cutoff":2.5,"pw_cutoff":6.0,"use_symmetry":false}})"_json;
    auto ctx  = create_simulation_context(conf, {{7, 0, 0}, {0, 7, 0}, {0, 0, 7}}, 1, {{0, 0, 0}}, false, false);
    K_point_set ks(*ctx);
    ks.add_kpoint({0, 0, 0}, 1);
    ks.initialize();
    auto kp = ks.get<double>(0);
    for (int s = 0; s < 2; s++) {
        for (int b = 0; b < ctx->num_bands(); b++) {
            kp->band_energy(b, s, 0);
        }
    }
    ks.find_band_occupancies<double>();
    for (int b = 0; b < ctx->num_bands(); b++) {
        if (std::abs(kp->band_occupancy(b, 0) - 11.0 / 12) > 1e-10 || kp->band_occupancy(b, 1) > 1e-10) {
            return 1;
        }
    }
    return 0;
}

int
main()
{
    sirius::initialize(true);
    int status = call_test("empty spin channel and degenerate bands", test_empty_spin_channel);
    sirius::finalize();
    return status;
}

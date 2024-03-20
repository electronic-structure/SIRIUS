/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <spla/spla.hpp>
#include <spla/context.hpp>
#include "core/wf/wave_functions.hpp"

using namespace sirius;

int
test_wf_ortho(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int use_gpu__, int bs__)
{
    auto pu = use_gpu__ ? device_t::GPU : device_t::CPU;
    spla::Context spla_ctx(pu == device_t::GPU ? SPLA_PU_GPU : SPLA_PU_HOST);

    la::BLACS_grid blacs_grid(mpi::Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]);

    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    /* create G-vectors */
    auto gvec = std::make_shared<fft::Gvec>(M, cutoff__, mpi::Communicator::world(), false);

    int num_atoms = 100;
    std::vector<int> nmt;
    for (int ia = 0; ia < num_atoms; ia++) {
        nmt.push_back(ia);
    }
    wf::Wave_functions<double> phi(gvec, nmt, wf::num_mag_dims(0), wf::num_bands(2 * num_bands__), memory_t::host);
    wf::Wave_functions<double> tmp(gvec, nmt, wf::num_mag_dims(0), wf::num_bands(2 * num_bands__), memory_t::host);

    sirius::randomize(phi);

    la::dmatrix<std::complex<double>> ovlp(2 * num_bands__, 2 * num_bands__, blacs_grid, bs__, bs__);

    memory_t mem{memory_t::host};
    if (pu == device_t::GPU) {
        mem = memory_t::device;
    }

    wf::orthogonalize(spla_ctx, mem, wf::spin_range(0), wf::band_range(0, 0), wf::band_range(0, num_bands__), phi, phi,
                      {&phi}, ovlp, tmp, true);

    wf::orthogonalize(spla_ctx, mem, wf::spin_range(0), wf::band_range(0, num_bands__),
                      wf::band_range(num_bands__, 2 * num_bands__), phi, phi, {&phi}, ovlp, tmp, true);

    wf::inner(spla_ctx, mem, wf::spin_range(0), phi, wf::band_range(0, 2 * num_bands__), phi,
              wf::band_range(0, 2 * num_bands__), ovlp, 0, 0);

    auto diff = la::check_identity(ovlp, 2 * num_bands__);
    if (diff > 1e-12) {
        printf("test_wf_ortho: wrong overlap");
        return 1;
    }
    return 0;
}

int
run_test(cmd_args const& args)
{
    auto mpi_grid_dims = args.value("mpi_grid_dims", std::vector<int>({1, 1}));
    auto cutoff        = args.value<double>("cutoff", 8.0);
    auto use_gpu       = args.value<int>("use_gpu", 0);

    auto result{0};
    for (int bs = 1; bs < 16; bs++) {
        for (int i = 30; i < 60; i++) {
            result += test_wf_ortho(mpi_grid_dims, cutoff, i, use_gpu, bs);
        }
    }
    return result;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    sirius::initialize(1);
    int result = sirius::call_test(argv[0], run_test, args);
    sirius::finalize();
    return result;
}

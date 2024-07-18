/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

using namespace sirius;

int
test_wf_inner_impl(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int bs__, memory_t mem__)
{
    spla::Context spla_ctx(is_host_memory(mem__) ? SPLA_PU_HOST : SPLA_PU_GPU);

    std::unique_ptr<la::BLACS_grid> blacs_grid;
    if (mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
        blacs_grid = std::make_unique<la::BLACS_grid>(mpi::Communicator::self(), 1, 1);
    } else {
        blacs_grid =
                std::make_unique<la::BLACS_grid>(mpi::Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]);
    }

    /* create G-vectors */
    auto gvec = fft::gkvec_factory(cutoff__, mpi::Communicator::world());

    if (mpi::Communicator::world().rank() == 0) {
        printf("number of bands          : %i\n", num_bands__);
        printf("total number of G-vectors: %i\n", gvec->num_gvec());
        printf("local number of G-vectors: %i\n", gvec->count());
    }

    wf::Wave_functions<double> phi1(gvec, wf::num_mag_dims(3), wf::num_bands(num_bands__), memory_t::host);
    wf::Wave_functions<double> phi2(gvec, wf::num_mag_dims(3), wf::num_bands(num_bands__), memory_t::host);

    auto sr = wf::spin_range(0, 2);

    for (auto s = sr.begin(); s != sr.end(); s++) {
        for (int i = 0; i < num_bands__; i++) {
            for (int igloc = 0; igloc < gvec->count(); igloc++) {
                int ig                                      = igloc + gvec->offset();
                phi1.pw_coeffs(igloc, s, wf::band_index(i)) = static_cast<double>(i + 1) / (ig + 1);
                phi2.pw_coeffs(igloc, s, wf::band_index(i)) = static_cast<double>(ig + 1) / (i + 1) / gvec->num_gvec();
            }
        }
    }

    auto mg1 = phi1.memory_guard(mem__, wf::copy_to::device);
    auto mg2 = phi2.memory_guard(mem__, wf::copy_to::device);

    la::dmatrix<std::complex<double>> ovlp(num_bands__, num_bands__, *blacs_grid, bs__, bs__);

    /* warmup call */
    wf::inner(spla_ctx, mem__, sr, phi1, wf::band_range(0, num_bands__), phi2, wf::band_range(0, num_bands__), ovlp, 0,
              0);
    mpi::Communicator::world().barrier();

    double t = -wtime();
    wf::inner(spla_ctx, mem__, sr, phi1, wf::band_range(0, num_bands__), phi2, wf::band_range(0, num_bands__), ovlp, 0,
              0);
    mpi::Communicator::world().barrier();
    t += wtime();

    double perf = sr.size() * 8e-9 * num_bands__ * num_bands__ * gvec->num_gvec() / t;
    if (mpi::Communicator::world().rank() == 0) {
        printf("execution time (sec) : %12.6f\n", t);
        printf("performance (GFlops) : %12.6f\n", perf);
    }

    double max_diff{0};
    for (int j = 0; j < ovlp.num_cols_local(); j++) {
        auto jcol = ovlp.icol(j);
        for (int i = 0; i < ovlp.num_rows_local(); i++) {
            auto irow = ovlp.irow(i);
            /* 2 is accumulated from two spins */
            std::complex<double> z = ovlp(i, j) - 2 * static_cast<double>(irow + 1) / (jcol + 1);
            max_diff               = std::max(max_diff, std::abs(z));
        }
    }
    mpi::Communicator::world().reduce<double, mpi::op_t::max>(&max_diff, 1, 0);
    if (max_diff > 1e-10) {
        return 1;
    }
    return 0;
}

int
test_wf_inner(cmd_args const& args)
{
    auto mpi_grid_dims       = args.value("mpi_grid_dims", std::vector<int>({1, 1}));
    auto cutoff              = args.value<double>("cutoff", 8.0);
    auto bs                  = args.value<int>("bs", 32);
    auto num_bands           = args.value<int>("num_bands", 100);
    std::string memory_t_str = args.value<std::string>("memory_t", "host");

    return test_wf_inner_impl(mpi_grid_dims, cutoff, num_bands, bs, get_memory_t(memory_t_str));
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"mpi_grid_dims=", "{int int} dimensions of MPI grid"},
                   {"cutoff=", "{double} wave-functions cutoff"},
                   {"bs=", "{int} block size"},
                   {"num_bands=", "{int} number of bands"},
                   {"memory_t=", "{string} type of the memory"}});

    sirius::initialize(1);
    int result = call_test("test_wf_inner", test_wf_inner, args);
    sirius::finalize(1);
    return result;
}

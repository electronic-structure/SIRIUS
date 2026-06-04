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
    /* we have plenty of gpu memory, allow a larger tile size */
    spla_ctx.set_tile_size_gpu(2096);

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

    double pref = 1.0 / std::sqrt(gvec->num_gvec());

    for (auto s = sr.begin(); s != sr.end(); s++) {
        for (int i = 0; i < num_bands__; i++) {
            for (int igloc = 0; igloc < gvec->count(); igloc++) {
                int ig                                      = igloc + gvec->offset();
                phi1.pw_coeffs(igloc, s, wf::band_index(i)) = pref * (i + 1) / (ig + 1);
                phi2.pw_coeffs(igloc, s, wf::band_index(i)) = pref * (ig + 1) / (i + 1);
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
    Measurement stat;

    int ierr{0};
    for (int k = 0; k < 4; k++) {
        if (mpi::Communicator::world().rank() == 0) {
            std::cout << "step " << k << std::endl;
        }
        double t = -wtime();
        wf::inner(spla_ctx, mem__, sr, phi1, wf::band_range(0, num_bands__), phi2, wf::band_range(0, num_bands__), ovlp,
                  0, 0);
        mpi::Communicator::world().barrier();
        t += wtime();
        double perf = sr.size() * 8e-9 * num_bands__ * num_bands__ * gvec->num_gvec() / t;
        stat.push_back(perf);
        if (mpi::Communicator::world().rank() == 0) {
            std::cout << "execution time : " << t << " sec." << std::endl;
            std::cout << "performance : " << perf << " GFlops" << ", " << perf / mpi::Communicator::world().size()
                      << " GFlops/rank" << std::endl;
        }
        double max_diff{0};
        for (int j = 0; j < ovlp.num_cols_local(); j++) {
            auto jcol = ovlp.icol(j);
            for (int i = 0; i < ovlp.num_rows_local(); i++) {
                auto irow = ovlp.irow(i);
                /* factor 1 or 2 is accumulated from spin components */
                auto z   = ovlp(i, j) - sr.size() * static_cast<double>(irow + 1) / (jcol + 1);
                max_diff = std::max(max_diff, std::abs(z));
            }
        }
        mpi::Communicator::world().allreduce<double, mpi::op_t::max>(&max_diff, 1);
        if (mpi::Communicator::world().rank() == 0) {
            std::cout << "max diff : " << max_diff << std::endl;
        }
        if (max_diff > 1e-8) {
            ierr++;
        }
        if (true && mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
            mpi::Communicator::world().barrier();
            double t0 = ::sirius::wtime();
            la::wrap(la::lib_t::blas)
                    .gemm('C', 'N', num_bands__, num_bands__, gvec->count(), &la::constant<std::complex<double>>::one(),
                          phi1.at(memory_t::host, 0, wf::spin_index(0), wf::band_index(0)), phi1.ld(),
                          phi2.at(memory_t::host, 0, wf::spin_index(0), wf::band_index(0)), phi2.ld(),
                          &la::constant<std::complex<double>>::zero(), ovlp.at(memory_t::host), ovlp.ld());
            mpi::Communicator::world().barrier();
            double t1 = ::sirius::wtime();
            mpi::Communicator::world().allreduce(ovlp.at(memory_t::host), num_bands__ * num_bands__);
            mpi::Communicator::world().barrier();
            double t2 = ::sirius::wtime();
            std::cout << "local zgemm time : " << t1 - t0 << ", allreduce time : " << t2 - t1
                      << ", effective performance : " << 8e-9 * num_bands__ * num_bands__ * gvec->num_gvec() / (t2 - t0)
                      << " gflops" << std::endl;
        }
    }
    if (mpi::Communicator::world().rank() == 0) {
        std::cout << "average performance (GFlops) : " << stat.average() << ", sigma : " << stat.sigma() << std::endl;
    }

    return ierr;
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

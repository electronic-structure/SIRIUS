/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

template <typename T, typename F>
void
test_wf_trans(la::BLACS_grid const& blacs_grid__, double cutoff__, int num_bands__, int bs__, int num_mag_dims__,
              memory_t mem__)
{
    spla::Context spla_ctx(is_host_memory(mem__) ? SPLA_PU_HOST : SPLA_PU_GPU);

    /* create G-vectors */
    auto gvec = fft::gkvec_factory(cutoff__, mpi::Communicator::world());

    if (mpi::Communicator::world().rank() == 0) {
        printf("number of bands          : %i\n", num_bands__);
        printf("num_mag_dims             : %i\n", num_mag_dims__);
        printf("total number of G-vectors: %i\n", gvec->num_gvec());
        printf("local number of G-vectors: %i\n", gvec->count());
    }

    int num_atoms = 31;
    std::vector<int> num_mt_coeffs(num_atoms);
    for (auto& n : num_mt_coeffs) {
        n = 123;
    }

    wf::Wave_functions<T> phi(gvec, num_mt_coeffs, wf::num_mag_dims(num_mag_dims__), wf::num_bands(num_bands__),
                              memory_t::host);

    wf::Wave_functions<T> psi(gvec, num_mt_coeffs, wf::num_mag_dims(num_mag_dims__), wf::num_bands(num_bands__),
                              memory_t::host);

    auto sr = num_mag_dims__ == 3 ? wf::spin_range(0, 2) : wf::spin_range(0);

    for (auto s = sr.begin(); s != sr.end(); s++) {
        for (int i = 0; i < num_bands__; i++) {
            for (int igloc = 0; igloc < gvec->count(); igloc++) {
                phi.pw_coeffs(igloc, s, wf::band_index(i)) = random<std::complex<T>>();
            }
            for (auto it : phi.spl_num_atoms()) {
                for (int xi = 0; xi < num_mt_coeffs[it.i]; xi++) {
                    phi.mt_coeffs(xi, it.li, s, wf::band_index(i)) = random<std::complex<T>>();
                }
            }
        }
    }

    la::dmatrix<F> tmtrx(num_bands__, num_bands__, blacs_grid__, bs__, bs__);
    tmtrx.zero();
    for (int i = 0; i < num_bands__; i++) {
        tmtrx.set(i, num_bands__ - i - 1, 1.0);
    }

    {
        auto mg1 = phi.memory_guard(mem__, wf::copy_to::device);
        auto mg2 = psi.memory_guard(mem__, wf::copy_to::host);

        /* warmup call */
        for (auto s = sr.begin(); s != sr.end(); s++) {
            wf::transform(spla_ctx, mem__, tmtrx, 0, 0, 1.0, phi, s, wf::band_range(0, num_bands__), 0.0, psi, s,
                          wf::band_range(0, num_bands__));
        }
        mpi::Communicator::world().barrier();

        double t = -wtime();
        for (auto s = sr.begin(); s != sr.end(); s++) {
            wf::transform(spla_ctx, mem__, tmtrx, 0, 0, 1.0, phi, s, wf::band_range(0, num_bands__), 0.0, psi, s,
                          wf::band_range(0, num_bands__));
        }
        mpi::Communicator::world().barrier();
        t += wtime();

        double perf = 8e-9 * num_bands__ * num_bands__ * gvec->num_gvec() * sr.size() / t;
        if (mpi::Communicator::world().rank() == 0) {
            printf("execution time (sec) : %12.6f\n", t);
            printf("performance (GFlops) : %12.6f\n", perf);
        }
    }

    double diff{0};
    for (auto s = sr.begin(); s != sr.end(); s++) {
        for (int i = 0; i < num_bands__; i++) {
            for (int igloc = 0; igloc < gvec->count(); igloc++) {
                diff += std::abs(phi.pw_coeffs(igloc, s, wf::band_index(i)) -
                                 psi.pw_coeffs(igloc, s, wf::band_index(num_bands__ - i - 1)));
            }
            for (auto it : phi.spl_num_atoms()) {
                for (int xi = 0; xi < num_mt_coeffs[it.i]; xi++) {
                    diff += std::abs(phi.mt_coeffs(xi, it.li, s, wf::band_index(i)) -
                                     psi.mt_coeffs(xi, it.li, s, wf::band_index(num_bands__ - i - 1)));
                }
            }
        }
    }
    std::cout << "diff = " << diff << std::endl;
}

template <typename T>
void
call_test(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int bs__, int num_mag_dims__,
          memory_t mem__, int repeat__)
{
    std::unique_ptr<la::BLACS_grid> blacs_grid;
    if (mpi_grid_dims__[0] * mpi_grid_dims__[1] == 1) {
        blacs_grid = std::make_unique<la::BLACS_grid>(mpi::Communicator::self(), 1, 1);
    } else {
        blacs_grid =
                std::make_unique<la::BLACS_grid>(mpi::Communicator::world(), mpi_grid_dims__[0], mpi_grid_dims__[1]);
    }
    for (int i = 0; i < repeat__; i++) {
        test_wf_trans<T, std::complex<double>>(*blacs_grid, cutoff__, num_bands__, bs__, num_mag_dims__, mem__);
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--bs=", "{int} block size");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--num_mag_dims=", "{int} number of magnetic dimensions");
    args.register_key("--memory_t=", "{string} type of memory");
    args.register_key("--fp32", "use FP32 arithmetics");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value("mpi_grid_dims", std::vector<int>({1, 1}));
    auto cutoff        = args.value<double>("cutoff", 8.0);
    auto bs            = args.value<int>("bs", 32);
    auto num_bands     = args.value<int>("num_bands", 100);
    auto num_mag_dims  = args.value<int>("num_mag_dims", 0);
    auto mem           = get_memory_t(args.value<std::string>("memory_t", "host"));

    sirius::initialize(1);
    if (args.exist("fp32")) {
#if defined(SIRIUS_USE_FP32)
        call_test<float>(mpi_grid_dims, cutoff, num_bands, bs, num_mag_dims, mem, 1);
#else
        RTE_THROW("Not compiled with FP32 support");
#endif
    } else {
        call_test<double>(mpi_grid_dims, cutoff, num_bands, bs, num_mag_dims, mem, 1);
    }

    int my_rank = mpi::Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0) {
        const auto timing_result = global_rtgraph_timer.process();
        std::cout << timing_result.print();
        // std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
        // ofs << timing_result.json();
    }
}

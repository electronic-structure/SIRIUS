/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>

/* test FFT: tranfrom random function to real space, transform back and compare with the original function */

using namespace sirius;
using namespace mpi;

template <typename T>
int
test_fft_impl(cmd_args const& args, device_t fft_pu__)
{
    double cutoff = args.value<double>("cutoff", 40);

    double eps = (typeid(T) == typeid(float)) ? 1e-6 : 1e-12;

    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    auto fft_grid = fft::get_min_grid(cutoff, M);

    auto spl_z = fft::split_z_dimension(fft_grid[2], Communicator::world());

    fft::Gvec gvec(M, cutoff, Communicator::world(), false);
    fft::Gvec_fft gvp(gvec, Communicator::world(), Communicator::self());

    fft::Gvec gvec_r(M, cutoff, Communicator::world(), true);
    fft::Gvec_fft gvp_r(gvec_r, Communicator::world(), Communicator::self());

    fft::spfft_grid_type<T> spfft_grid(fft_grid[0], fft_grid[1], fft_grid[2],
                                       std::max(gvp.zcol_count(), gvp_r.zcol_count()), spl_z.local_size(),
                                       SPFFT_PU_HOST, -1, Communicator::world().native(), SPFFT_EXCH_DEFAULT);

    auto const& gv = gvp.gvec_array();
    fft::spfft_transform_type<T> spfft(
            spfft_grid.create_transform(SPFFT_PU_HOST, SPFFT_TRANS_C2C, fft_grid[0], fft_grid[1], fft_grid[2],
                                        spl_z.local_size(), gvp.count(), SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host)));

    auto const& gv_r = gvp_r.gvec_array();
    fft::spfft_transform_type<T> spfft_r(spfft_grid.create_transform(
            SPFFT_PU_HOST, SPFFT_TRANS_R2C, fft_grid[0], fft_grid[1], fft_grid[2], spl_z.local_size(), gvp_r.count(),
            SPFFT_INDEX_TRIPLETS, gv_r.at(memory_t::host)));

    /* the test is doing the following:
     * 1. create PW coefficients of the real function
     * 2. transform it to real space
     * 3. copy back as complex function with only real part
     * 4. transform back to PW coefficients of a complex function
     * 5. compare original and final PW coefficients
     */

    mdarray<std::complex<T>, 1> f({gvp_r.count()});
    for (int ig = 0; ig < gvp_r.count(); ig++) {
        if (gv_r(0, ig) == 0 && gv_r(1, ig) == 0 && gv_r(2, ig) == 0) {
            f[ig] = 1.0;
        } else {
            f[ig] = random<std::complex<T>>();
        }
    }
    /* transform to real space */
    spfft_r.backward(reinterpret_cast<T const*>(&f[0]), spfft.processing_unit());
    mdarray<T, 1> f_rg({fft_grid[0] * fft_grid[1] * spl_z.local_size()});
    /* unload to real-space buffer */
    fft::spfft_output(spfft_r, f_rg.at(memory_t::host));

    /* copy to complex data */
    mdarray<std::complex<T>, 1> g_rg({fft_grid[0] * fft_grid[1] * spl_z.local_size()});
    for (size_t i = 0; i < f_rg.size(); i++) {
        g_rg[i] = f_rg[i];
    }
    /* load to spfft buffer */
    fft::spfft_input(spfft, g_rg.at(memory_t::host));
    /* tranform to reciprocal space */
    mdarray<std::complex<T>, 1> g({gvp.count()});
    spfft.forward(spfft.processing_unit(), reinterpret_cast<T*>(&g[0]), SPFFT_FULL_SCALING);

    mdarray<std::complex<T>, 1> f_glob({gvec_r.num_gvec()});
    gvp_r.gather_pw_global(&f[0], &f_glob[0]);

    mdarray<std::complex<T>, 1> g_glob({gvec.num_gvec()});
    gvp.gather_pw_global(&g[0], &g_glob[0]);

    double diff{0};
    for (int ig = 0; ig < gvec_r.num_gvec(); ig++) {
        auto G    = gvec_r.gvec(gvec_index_t::global(ig));
        auto idx1 = gvec.index_by_gvec(G);
        auto idx2 = gvec.index_by_gvec(-1 * G);
        diff      = std::max(diff, static_cast<double>(std::abs(f_glob[ig] - g_glob[idx1])));
        diff      = std::max(diff, static_cast<double>(std::abs(f_glob[ig] - std::conj(g_glob[idx2]))));
    }

    if (diff > eps) {
        return 1;
    } else {
        return 0;
    }
}

template <typename T>
int
test_fft(cmd_args const& args)
{
    int result = test_fft_impl<T>(args, device_t::CPU);
#ifdef SIRIUS_GPU
    result += test_fft_impl<T>(args, device_t::GPU);
#endif
    return result;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"cutoff=", "{double} cutoff radius in G-space"}, {"fp32", "run in FP32 arithmetics"}});

    sirius::initialize(true);
    int result{0};
    if (args.exist("fp32")) {
#if defined(SIRIUS_USE_FP32)
        result = call_test(argv[0], test_fft<float>, args);
#else
        RTE_THROW("not compiled with FP32 support");
#endif
    } else {
        result = call_test(argv[0], test_fft<double>, args);
    }
    sirius::finalize();

    return result;
}

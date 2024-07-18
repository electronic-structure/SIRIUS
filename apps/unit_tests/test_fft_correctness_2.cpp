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
test_fft_complex_impl(cmd_args const& args, device_t fft_pu__)
{
    double cutoff = args.value<double>("cutoff", 40);

    double eps = (typeid(T) == typeid(float)) ? 1e-6 : 1e-12;

    r3::matrix<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    auto fft_grid = fft::get_min_grid(cutoff, M);

    auto spl_z = fft::split_z_dimension(fft_grid[2], Communicator::world());

    fft::Gvec gvec(M, cutoff, Communicator::world(), false);

    fft::Gvec_fft gvp(gvec, Communicator::world(), Communicator::self());

    fft::spfft_grid_type<T> spfft_grid(fft_grid[0], fft_grid[1], fft_grid[2], gvp.zcol_count(), spl_z.local_size(),
                                       SPFFT_PU_HOST, -1, Communicator::world().native(), SPFFT_EXCH_DEFAULT);

    const auto fft_type = gvec.reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;

    auto const& gv = gvp.gvec_array();
    fft::spfft_transform_type<T> spfft(spfft_grid.create_transform(SPFFT_PU_HOST, fft_type, fft_grid[0], fft_grid[1],
                                                                   fft_grid[2], spl_z.local_size(), gvp.count(),
                                                                   SPFFT_INDEX_TRIPLETS, gv.at(memory_t::host)));

    mdarray<std::complex<T>, 1> f({gvp.count()});
    for (int ig = 0; ig < gvp.count(); ig++) {
        f[ig] = random<std::complex<T>>();
    }
    mdarray<std::complex<T>, 1> g({gvp.count()});

    spfft.backward(reinterpret_cast<T const*>(&f[0]), spfft.processing_unit());
    spfft.forward(spfft.processing_unit(), reinterpret_cast<T*>(&g[0]), SPFFT_FULL_SCALING);

    double diff{0};
    for (int ig = 0; ig < gvp.count(); ig++) {
        diff = std::max(diff, static_cast<double>(std::abs(f[ig] - g[ig])));
    }
    Communicator::world().allreduce<double, mpi::op_t::max>(&diff, 1);

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
    int result = test_fft_complex_impl<T>(args, device_t::CPU);
#ifdef SIRIUS_GPU
    result += test_fft_complex_impl<T>(args, device_t::GPU);
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

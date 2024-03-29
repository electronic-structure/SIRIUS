/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
test_fft_perf(std::vector<int> mpi_grid_dims__, double cutoff__, int repeat__, int use_gpu__, int gpu_ptr__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D_grid fft_box(find_translations(2.01 * cutoff__, M));

    FFT3D fft(fft_box, mpi_comm_world(), pu);

    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("number of z-columns: %i\n", gvec.num_zcol());
        printf("FFT grid size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
        printf("number of FFT threads: %i\n", omp_get_max_threads());
    }

    fft.prepare(gvec.partition());

    mdarray<double_complex, 1> vin(gvec.partition().gvec_count_fft());
    vin = [](size_t i) { return type_wrapper<double_complex>::random(); };
    mdarray<double_complex, 1> vout(gvec.partition().gvec_count_fft());

    if (pu == GPU) {
        vin.allocate(memory_t::device);
        vin.copy<memory_t::host, memory_t::device>();
        vout.allocate(memory_t::device);
    }

    mpi_comm_world().barrier();
    sddk::timer t1("test_fft_perf");
    for (int i = 0; i < repeat__; i++) {
        if (gpu_ptr__) {
            fft.transform<1, GPU>(gvec.partition(), vin.at<GPU>());
            fft.transform<-1, GPU>(gvec.partition(), vout.at<GPU>());
        } else {
            fft.transform<1, CPU>(gvec.partition(), vin.at<CPU>());
            fft.transform<-1, CPU>(gvec.partition(), vout.at<CPU>());
        }
    }
    mpi_comm_world().barrier();
    double tval = t1.stop();

    if (pu == GPU && gpu_ptr__) {
        vout.copy<memory_t::device, memory_t::host>();
    }

    double diff{0};
    for (int j = 0; j < gvec.partition().gvec_count_fft(); j++) {
        diff += std::abs(vin[j] - vout[j]);
    }

    if (diff != diff) {
        RTE_THROW("NaN");
    }
    mpi_comm_world().allreduce(&diff, 1);
    if (mpi_comm_world().rank() == 0) {
        printf("diff: %18.12f\n", diff);
        printf("performance: %12.6f transformations / sec.", 2 * repeat__ / tval);
    }

    fft.dismiss();
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--gpu_ptr=", "{int} 0: start from CPU, 1: start from GPU");
    args.register_key("--repeat=", "{int} number of repetitions");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto cutoff        = args.value<double>("cutoff", 2.0);
    auto use_gpu       = args.value<int>("use_gpu", 0);
    auto gpu_ptr       = args.value<int>("gpu_ptr", 0);
    auto repeat        = args.value<int>("repeat", 100);

    sirius::initialize(1);
    test_fft_perf(mpi_grid_dims, cutoff, repeat, use_gpu, gpu_ptr);
    mpi_comm_world().barrier();
    sddk::timer::print();
    // runtime::Timer::print_all();
    sirius::finalize();
}

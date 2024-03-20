/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

inline void
mul_veff(int n, double_complex* buff, double* veff)
{
    #pragma omp parallel for
    for (int j = 0; j < n; j++)
        buff[j] *= veff[j];
}

namespace test_vmul {

double
kernel1(int repeat, int n, std::vector<double_complex>& v1, std::vector<double>& v2)
{
    double tloc = 0;
    for (int i = 0; i < repeat; i++) {
        double t = omp_get_wtime();
        mul_veff(n, &v1[0], &v2[0]);
        //#pragma omp parallel for
        // for (int j = 0; j < n; j++) v1[i] *= v2[i];

        tloc += (omp_get_wtime() - t);
    }
    return tloc;
}

double
kernel2(int repeat, int n, mdarray<double_complex, 1>& v1, mdarray<double, 1>& v2)
{
    double tloc = 0;
    for (int i = 0; i < repeat; i++) {
        double t = omp_get_wtime();
        #pragma omp parallel for
        for (int j = 0; j < n; j++)
            v1[i] *= v2[i];

        tloc += (omp_get_wtime() - t);
    }
    return tloc;
}

double
kernel3(int repeat, int n, mdarray<double_complex, 1>& v1, std::vector<double>& v2)
{
    double tloc = 0;
    for (int i = 0; i < repeat; i++) {
        double t = omp_get_wtime();
        #pragma omp parallel for
        for (int j = 0; j < n; j++)
            v1[i] *= v2[i];

        tloc += (omp_get_wtime() - t);
    }
    return tloc;
}

void
test_vmul_drv()
{
    int n      = 216 * 216 * 216;
    int repeat = 20;

    std::vector<double_complex> v1(n);
    std::vector<double> v2(n);

    mdarray<double_complex, 1> f1(n);
    mdarray<double, 1> f2(n);
    for (int i = 0; i < n; i++) {
        v1[i] = 1.0 / double_complex(i + 1, i + 1);
        v2[i] = 2.0;
        f1[i] = 1.0 / double_complex(i + 1, i + 1);
        f2[i] = 2.0;
    }

    std::cout << "vector size: " << n << std::endl;

    double t = kernel1(repeat, n, v1, v2);
    std::cout << "kernel1 time: " << t
              << " speed: " << double(n * (2 * sizeof(double_complex) + sizeof(double)) * repeat) / t / (1 << 30)
              << " GBs" << std::endl;

    t = kernel2(repeat, n, f1, f2);
    std::cout << "kernel2 time: " << t
              << " speed: " << double(n * (2 * sizeof(double_complex) + sizeof(double)) * repeat) / t / (1 << 30)
              << " GBs" << std::endl;

    t = kernel3(repeat, n, f1, v2);
    std::cout << "kernel3 time: " << t
              << " speed: " << double(n * (2 * sizeof(double_complex) + sizeof(double)) * repeat) / t / (1 << 30)
              << " GBs" << std::endl;
}

} // namespace test_vmul

using namespace sirius;

void
test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int num_fft_streams__,
          int num_threads_fft__, int use_gpu, double gpu_workload__)
{
    //== MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world());
    //==
    //== matrix3d<double> M;
    //== M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    //== FFT3D_grid fft_grid(2.01 * cutoff__, M);

    //== FFT3D_context fft_ctx(mpi_grid, fft_grid, num_fft_streams__, num_threads_fft__,
    //==                       static_cast<processing_unit_t>(use_gpu), gpu_workload__);

    //== Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft_grid, mpi_grid.communicator(1 << 1),
    //==           mpi_grid.dimension_size(0), false, false);
    int n = 216 * 216 * 216; // fft_ctx.fft(0)->local_size();

    //== std::vector<double> pw_ekin(gvec.num_gvec(), 0);
    std::vector<double> veff(n, 2.0);

    //== if (mpi_comm_world().rank() == 0)
    //== {
    //==     printf("total number of G-vectors: %i\n", gvec.num_gvec());
    //==     printf("local number of G-vectors: %i\n", gvec.num_gvec(0));
    //==     printf("FFT grid size: %i %i %i\n", fft_grid.size(0), fft_grid.size(1), fft_grid.size(2));
    //==     printf("number of FFT streams: %i\n", fft_ctx.num_fft_streams());
    //==     printf("number of FFT groups: %i\n", mpi_grid.dimension_size(0));
    //== }

    mdarray<double_complex, 1> buf1(n);
    for (int j = 0; j < n; j++) {
        buf1(j) = double_complex(1, 0);
        // fft_ctx.fft(0)->buffer(j) = double_complex(1, 0);
    }

    test_vmul::test_vmul_drv();

    double tloc = 0;
    std::cout << "vector size: " << n << std::endl;
    for (int i = 0; i < num_bands__; i++) {
        double t = omp_get_wtime();
        #pragma omp parallel for
        for (int j = 0; j < n; j++)
            buf1[j] *= veff[j];

        tloc += (omp_get_wtime() - t);
    }

    std::cout << "time: " << tloc << " speed: "
              << double(n * (2 * sizeof(double_complex) + sizeof(double)) * num_bands__) / tloc / (1 << 30) << " GBs"
              << std::endl;

    test_vmul::test_vmul_drv();

    // fft_ctx.prepare();
    //
    // Hloc_operator hloc(fft_ctx, gvec, veff);

    // Wave_functions<false> phi(4 * num_bands__, gvec, mpi_grid, CPU);
    // for (int i = 0; i < 4 * num_bands__; i++)
    //{
    //     for (int j = 0; j < phi.num_gvec_loc(); j++)
    //     {
    //         phi(j, i) = type_wrapper<double_complex>::random();
    //     }
    // }
    // Wave_functions<false> hphi(4 * num_bands__, num_bands__, gvec, mpi_grid, CPU);
    //
    // runtime::Timer t1("h_loc");
    // for (int i = 0; i < 4; i++)
    //{
    //     hphi.copy_from(phi, i * num_bands__, num_bands__);
    //     hloc.apply(0, hphi, i * num_bands__, num_bands__);
    // }
    // t1.stop();

    // double diff = 0;
    // for (int i = 0; i < 4 * num_bands__; i++)
    //{
    //     for (int j = 0; j < phi.num_gvec_loc(); j++)
    //     {
    //         diff += std::abs(2.0 * phi(j, i) - hphi(j, i));
    //     }
    // }
    // mpi_comm_world().allreduce(&diff, 1);
    // if (mpi_comm_world().rank() == 0)
    //{
    //     printf("diff: %18.12f\n", diff);
    // }

    // fft_ctx.dismiss();
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--num_fft_streams=", "{int} number of independent FFT streams");
    args.register_key("--num_threads_fft=", "{int} number of threads for each FFT");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--gpu_workload=", "{double} worload of GPU");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }
    auto mpi_grid_dims   = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto num_fft_streams = args.value<int>("num_fft_streams", 1);
    auto num_threads_fft = args.value<int>("num_threads_fft", omp_get_max_threads());
    auto cutoff          = args.value<double>("cutoff", 2.0);
    auto num_bands       = args.value<int>("num_bands", 10);
    auto use_gpu         = args.value<int>("use_gpu", 0);
    auto gpu_workload    = args.value<double>("gpu_workload", 0.8);

    test_vmul::test_vmul_drv();

    sirius::initialize(1);

    test_vmul::test_vmul_drv();

    test_hloc(mpi_grid_dims, cutoff, num_bands, num_fft_streams, num_threads_fft, use_gpu, gpu_workload);
    mpi_comm_world().barrier();
    runtime::Timer::print();
    sirius::finalize();
}

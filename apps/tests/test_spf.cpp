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
test_spf(std::vector<int> mpi_grid_dims__, double cutoff__, int use_gpu__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world());

    // auto& c1 = mpi_grid.communicator(1 << 0 | 1 << 2); ////mpi_comm_world().split(mpi_comm_world().rank() /
    // mpi_grid_dims__[0]); auto c2 = mpi_comm_world().split(c1.rank());
    auto& c2 = mpi_grid.communicator(1 << 0);

    /* create FFT box */
    FFT3D_grid fft_box(Utils::find_translations(2.01 * cutoff__, M));
    /* create FFT driver */
    FFT3D fft(fft_box, c2, pu);
    /* create G-vectors */
    Gvec gvec({0, 0, 0}, M, cutoff__, fft_box, mpi_comm_world().size(), fft.comm(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("FFT grid size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
        printf("FFT comm size: %i\n", fft.comm().size());
    }

    experimental::Smooth_periodic_function<double_complex> spf(fft, gvec, mpi_comm_world());

    std::vector<double_complex> fpw(gvec.num_gvec());
    for (int i = 0; i < gvec.num_gvec(); i++) {
        fpw[i] = double_complex(i, 0); // type_wrapper<double_complex>::random();
    }

    mdarray<double_complex, 1> tmp(gvec.gvec_count(mpi_comm_world().rank()));

    for (int i = 0; i < gvec.gvec_count(mpi_comm_world().rank()); i++) {
        spf.f_pw_local(i) = tmp[i] = fpw[i + gvec.gvec_offset(mpi_comm_world().rank())];
    }
    fft.prepare(gvec.partition());
    spf.fft_transform(1);
    spf.fft_transform(-1);
    fft.dismiss();

    for (int i = 0; i < gvec.gvec_count(mpi_comm_world().rank()); i++) {
        if (std::abs(spf.f_pw_local(i) - tmp[i]) > 1e-12) {
            std::stringstream s;
            s << "large difference: " << std::abs(spf.f_pw_local(i) - tmp[i]);
            RTE_THROW(s);
        }
    }

    auto fpw1 = spf.gather_f_pw();
    if (mpi_comm_world().rank() == 0) {
        for (int i = 0; i < gvec.num_gvec(); i++) {
            std::cout << i << " " << fpw[i] << " " << fpw1[i] << std::endl;
        }
    }
    mpi_comm_world().barrier();

    for (int i = 0; i < gvec.num_gvec(); i++) {
        if (std::abs(fpw[i] - fpw1[i]) > 1e-12) {
            std::stringstream s;
            s << "large difference: " << std::abs(fpw[i] - fpw1[i]);
            RTE_THROW(s);
        }
    }
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
    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto cutoff        = args.value<double>("cutoff", 2.0);
    auto use_gpu       = args.value<int>("use_gpu", 0);

    sirius::initialize(1);
    test_spf(mpi_grid_dims, cutoff, use_gpu);

    mpi_comm_world().barrier();
    sddk::timer::print();
    // sddk::timer::print_all();
    sirius::finalize();
}

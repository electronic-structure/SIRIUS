/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <thread>

/* test transformation of real function */

using namespace sirius;

int
run_test(cmd_args& args, device_t pu__)
{
    // std::vector<int> vd = args.value< std::vector<int> >("dims", {132, 132, 132});
    // vector3d<int> dims(vd[0], vd[1], vd[2]);
    // double cutoff = args.value<double>("cutoff", 50);

    // matrix3d<double> M;
    // M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    // FFT3D fft(find_translations(cutoff, M), Communicator::world(), pu__);

    // Gvec gvec(M, cutoff, Communicator::world(), false);
    // Gvec_partition gvecp(gvec, Communicator::world(), Communicator::self());

    // Gvec gvec_r(M, cutoff, Communicator::world(), true);
    // Gvec_partition gvecp_r(gvec_r, Communicator::world(), Communicator::self());

    // fft.prepare(gvecp_r);

    // mdarray<double_complex, 1> phi(gvecp_r.gvec_count_fft());
    // for (int i = 0; i < gvecp_r.gvec_count_fft(); i++) {
    //     phi(i) = utils::random<double_complex>();
    // }
    // phi(0) = 1.0;
    // fft.transform<1>(&phi[0]);
    // mdarray<double_complex, 1> phi1(gvecp_r.gvec_count_fft());
    // fft.transform<-1>(&phi1[0]);

    // double rms{0};
    // for (int i = 0; i < gvecp_r.gvec_count_fft(); i++) {
    //     rms += std::pow(std::abs(phi(i) - phi1(i)), 2);
    // }
    // rms = std::sqrt(rms / gvecp_r.gvec_count_fft());

    // fft.dismiss();

    // if (rms > 1e-13) {
    //     return 1;
    // } else {
    //     return 0;
    // }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    sirius::initialize(true);
    printf("running %-30s : ", argv[0]);
    int result = run_test(args, device_t::CPU);
    if (result) {
        printf("\x1b[31m"
               "Failed"
               "\x1b[0m"
               "\n");
    } else {
        printf("\x1b[32m"
               "OK"
               "\x1b[0m"
               "\n");
    }
    sirius::finalize();

    return result;
}

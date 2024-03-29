/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>
#include <thread>
#include <wave_functions.h>

using namespace sirius;

void
test_fft(vector3d<int> const& dims__, double cutoff__, int num_bands__, std::vector<int> mpi_grid_dims__)
{
    Communicator comm(MPI_COMM_WORLD);
    MPI_grid mpi_grid(mpi_grid_dims__, comm);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(dims__, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), GPU);
    fft.allocate_on_device();

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), fft.comm(), mpi_grid.communicator(1 << 0).size(),
              false, false);
    // gvec.index_map().allocate_on_device();
    // gvec.index_map().copy_to_device();
    gvec.z_columns_pos().allocate_on_device();
    gvec.z_columns_pos().copy_to_device();

    Wave_functions psi_in(num_bands__, gvec, mpi_grid, true);
    Wave_functions psi_out(num_bands__, gvec, mpi_grid, true);

    if (comm.rank() == 0) {
        printf("num_gvec: %i\n", gvec.num_gvec());
    }
    printf("num_gvec_fft: %i\n", gvec.num_gvec_fft());
    printf("num_gvec_loc: %i\n", gvec.num_gvec(comm.rank()));
    printf("num_gvec_loc: %i\n", psi_in.num_gvec_loc());

    for (int i = 0; i < num_bands__; i++) {
        for (int j = 0; j < psi_in.num_gvec_loc(); j++) {
            psi_in(j, i) = type_wrapper<double_complex>::random();
        }
    }
    psi_in.swap_forward(0, num_bands__);

    // mdarray<double_complex, 1> pw_buf(gvec.num_gvec());
    // pw_buf.allocate_on_device();

    Timer t1("fft|transform");
    for (int i = 0; i < psi_in.spl_num_swapped().local_size(); i++) {
        // cuda_copy_to_device(pw_buf.at<GPU>(), psi_in[i], gvec.num_gvec() * sizeof(double_complex));

        ///* set PW coefficients into proper positions inside FFT buffer */
        // fft.input_on_device(gvec.num_gvec(), gvec.index_map().at<GPU>(), pw_buf.at<GPU>());

        // fft.transform<1>(psi_in.gvec(), pw_buf.at<GPU>());
        // fft.transform<-1>(psi_out.gvec(), pw_buf.at<GPU>());
        //
        ///* phi(G) += fft_buffer(G) */
        // fft.output_on_device(gvec.num_gvec(), gvec.index_map().at<GPU>(), pw_buf.at<GPU>(), 0.0);

        // cuda_copy_to_host(psi_out[i], pw_buf.at<GPU>(), gvec.num_gvec() * sizeof(double_complex));

        fft.transform<1>(psi_in.gvec(), psi_in[i]);
        fft.transform<-1>(psi_out.gvec(), psi_out[i]);
    }
    t1.stop();

    psi_out.swap_backward(0, num_bands__);

    double diff = 0;
    for (int i = 0; i < num_bands__; i++) {
        for (int j = 0; j < psi_in.num_gvec_loc(); j++) {
            double d = std::abs(psi_in(j, i) - psi_out(j, i));
            if (d > 1e-10) {
                std::cout << "j=" << j << " expected: " << psi_in(j, i) << " got: " << psi_out(j, i) << std::endl;
            }
            diff += d;
        }
    }
    printf("diff: %18.12f\n", diff);

    // psi_slab.zero();
    // for (int i = 0; i < num_bands__; i++)
    //{
    //     for (int j = 0; j < gvec.num_gvec(mpi_grid.communicator().rank()); j++)
    //     {
    //         int ig = gvec.offset_gvec(mpi_grid.communicator().rank()) + j;
    //         if (ig == i) psi_slab(j, i) = 1;
    //     }
    // }

    // for (int i = 0; i < (int)panel_col_distr.local_size(); i++)
    //{
    //     for (int j = 0; j < gvec.num_gvec_fft(); j++) psi_panel(j, i) = type_wrapper<double_complex>::random();
    // }

    // slab_to_panel(num_bands__, psi_slab, psi_panel, gvec, mpi_grid);

    ////pstdout pout(comm);
    ////pout.printf("rank: %i\n", comm.rank());
    ////for (int i = 0; i < gvec.num_gvec_loc(); i++)
    ////{
    ////    pout.printf("row: %4i ", i);
    ////    for (int j = 0; j < panel_col_distr.local_size(); j++)
    ////    {
    ////        pout.printf("{%12.6f %12.6f} ", psi_panel(i, j).real(), psi_panel(i, j).imag());
    ////    }
    ////    pout.printf("\n");
    ////}
    ////pout.flush();

    ////Timer::delay(1);
    ////STOP();

    // mdarray<double_complex, 1> psi_tmp(gvec.num_gvec_fft());
    // for (int igloc = 0; igloc < (int)panel_col_distr.local_size(); igloc++)
    ////for (int ig = 0; ig < gvec.num_gvec(); ig++)
    //{
    //    int ig = static_cast<int>(panel_col_distr[igloc]);
    //    auto v = gvec[ig];
    //    printf("ig: %i, gvec: %i %i %i\n", ig, v[0], v[1], v[2]);
    //    //psi.zero();
    //    //psi(ig) = 1.0;
    //    //fft.transform<1>(gvec, &psi(gvec.gvec_offset()));
    //    fft.transform<1>(gvec, &psi_panel(0, igloc));
    //    fft.transform<-1>(gvec, &psi_tmp(0));

    //    double diff = 0;
    //    for (int i = 0; i < gvec.num_gvec_fft(); i++)
    //    {
    //        diff += std::pow(std::abs(psi_panel(i, igloc) - psi_tmp(i)), 2);
    //    }
    //    diff = std::sqrt(diff / gvec.num_gvec_fft());
    //
    //    //== double diff = 0;
    //    //== /* loop over 3D array (real space) */
    //    //== for (int j0 = 0; j0 < fft.fft_grid().size(0); j0++)
    //    //== {
    //    //==     for (int j1 = 0; j1 < fft.fft_grid().size(1); j1++)
    //    //==     {
    //    //==         for (int j2 = 0; j2 < fft.local_size_z(); j2++)
    //    //==         {
    //    //==             /* get real space fractional coordinate */
    //    //==             auto rl = vector3d<double>(double(j0) / fft.fft_grid().size(0),
    //    //==                                        double(j1) / fft.fft_grid().size(1),
    //    //==                                        double(fft.offset_z() + j2) / fft.fft_grid().size(2));
    //    //==             int idx = fft.fft_grid().index_by_coord(j0, j1, j2);

    //    //==             diff += std::pow(std::abs(fft.buffer(idx) - std::exp(double_complex(0.0, twopi * (rl * v)))),
    //    2);
    //    //==         }
    //    //==     }
    //    //== }
    //    //== diff = std::sqrt(diff / fft.size());
    //    printf("RMS difference : %18.10e", diff);
    //    if (diff < 1e-10)
    //    {
    //        printf("  OK\n");
    //    }
    //    else
    //    {
    //        printf("  Fail\n");
    //    }
    //}
    comm.barrier();
}

void
test2(vector3d<int> const& dims__, double cutoff__, std::vector<int> mpi_grid_dims__)
{
    Communicator comm(MPI_COMM_WORLD);
    MPI_grid mpi_grid(mpi_grid_dims__, comm);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft1(dims__, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), CPU);
    auto dims2 = dims__;
    for (int i : {0, 1, 2})
        dims2[i] *= 2;
    FFT3D fft2(dims2, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), CPU);

    Gvec gvec1(vector3d<double>(0, 0, 0), M, cutoff__, fft1.grid(), fft1.comm(), mpi_grid.communicator(1 << 0).size(),
               false, false);

    FFT3D* fft = &fft2;

    mdarray<double_complex, 1> psi_tmp(gvec1.num_gvec());
    for (int ig = 0; ig < std::min(gvec1.num_gvec(), 100); ig++) {
        auto v = gvec1[ig];
        printf("ig: %i, gvec: %i %i %i\n", ig, v[0], v[1], v[2]);
        psi_tmp.zero();
        psi_tmp(ig) = 1.0;
        fft->transform<1>(gvec1, &psi_tmp(gvec1.offset_gvec_fft()));

        double diff = 0;
        /* loop over 3D array (real space) */
        for (int j0 = 0; j0 < fft->grid().size(0); j0++) {
            for (int j1 = 0; j1 < fft->grid().size(1); j1++) {
                for (int j2 = 0; j2 < fft->local_size_z(); j2++) {
                    /* get real space fractional coordinate */
                    auto rl = vector3d<double>(double(j0) / fft->grid().size(0), double(j1) / fft->grid().size(1),
                                               double(fft->offset_z() + j2) / fft->grid().size(2));
                    int idx = fft->grid().index_by_coord(j0, j1, j2);

                    diff += std::pow(std::abs(fft->buffer(idx) - std::exp(double_complex(0.0, twopi * (rl * v)))), 2);
                }
            }
        }
        diff = std::sqrt(diff / fft->size());
        printf("RMS difference : %18.10e", diff);
        if (diff < 1e-10) {
            printf("  OK\n");
        } else {
            printf("  Fail\n");
        }
    }
}

void
test3(vector3d<int> const& dims__, double cutoff__)
{
    Communicator comm(MPI_COMM_WORLD);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;

    FFT3D fft(dims__, Platform::max_num_threads(), comm, CPU);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), comm, 1, false, false);
    Gvec gvec_r(vector3d<double>(0, 0, 0), M, cutoff__, fft.grid(), comm, 1, false, true);

    printf("num_gvec: %i, num_gvec_reduced: %i\n", gvec.num_gvec(), gvec_r.num_gvec());
    printf("num_gvec_loc: %i %i\n", gvec.num_gvec(comm.rank()), gvec_r.num_gvec(comm.rank()));

    // mdarray<double_complex, 1> phi(gvec.num_gvec());
    // for (int i = 0; i < fft.size(); i++) fft.buffer(i) = type_wrapper<double>::random();
    // fft.transform<-1>(gvec, &phi(gvec.offset_gvec_fft()));

    // for (size_t i = 0; i < gvec.z_columns().size(); i++)
    //{
    //     auto zcol = gvec.z_columns()[i];
    //     printf("x,y: %3i %3i\n", zcol.x, zcol.y);
    //     for (size_t j = 0; j < zcol.z.size(); j++)
    //     {
    //         printf("z: %3i, val: %12.6f %12.6f\n", zcol.z[j], phi(zcol.offset + j).real(), phi(zcol.offset +
    //         j).imag());
    //     }
    // }

    mdarray<double_complex, 1> phi(gvec_r.num_gvec());
    for (int i = 0; i < gvec_r.num_gvec(); i++)
        phi(i) = type_wrapper<double_complex>::random();
    phi(0) = 1.0;
    fft.transform<1>(gvec_r, &phi(gvec.offset_gvec_fft()));

    mdarray<double_complex, 1> phi1(gvec_r.num_gvec());
    fft.transform<-1>(gvec_r, &phi1(gvec.offset_gvec_fft()));

    double diff = 0;
    for (int i = 0; i < gvec_r.num_gvec(); i++) {
        diff += std::abs(phi(i) - phi1(i));
    }
    printf("diff: %18.12f\n", diff);
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--mpi_grid=", "{vector2d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (argn == 1) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims        = args.value<vector3d<int>>("dims");
    double cutoff             = args.value<double>("cutoff", 1);
    int num_bands             = args.value<int>("num_bands", 50);
    std::vector<int> mpi_grid = args.value<std::vector<int>>("mpi_grid", {1, 1});

    Platform::initialize(1);

    test_fft(dims, cutoff, num_bands, mpi_grid);
    // test2(dims, cutoff, mpi_grid);
    // test3(dims, cutoff);

    Timer::print();

    Platform::finalize();
}

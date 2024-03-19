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
test_gvec_distr(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    // FFT3D_grid fft_box(2.01 * cutoff__, M);
    // FFT3D_grid fft_box(cutoff__, M);

    Gvec gvec;

    gvec = Gvec(vector3d<double>(0, 0, 0), M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    splindex<block> spl_num_gsh(gvec.num_shells(), mpi_comm_world().size(), mpi_comm_world().rank());

    runtime::pstdout pout(mpi_comm_world());
    pout.printf("rank: %i\n", mpi_comm_world().rank());
    pout.printf("-----------------------\n");
    // pout.printf("FFT box size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
    pout.printf("num_gvec            : %i\n", gvec.num_gvec());
    pout.printf("num_gvec_loc        : %i\n", gvec.gvec_count(mpi_comm_world().rank()));
    pout.printf("num_zcols           : %i\n", gvec.num_zcol());
    pout.printf("num_gvec_fft        : %i\n", gvec.partition().gvec_count_fft());
    pout.printf("offset_gvec_fft     : %i\n", gvec.partition().gvec_offset_fft());
    pout.printf("num_zcols_local     : %i\n", gvec.partition().zcol_distr_fft().counts[mpi_comm_world().rank()]);
    pout.printf("num_gvec_shells     : %i\n", gvec.num_shells());
    pout.printf("num_gvec_shells_loc : %i\n", spl_num_gsh.local_size());
    pout.flush();

    std::vector<int> nv(gvec.num_shells(), 0);
    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        nv[gvec.shell(ig)]++;
    }

    if (mpi_comm_world().rank() == 0) {
        FILE* fout = fopen("gshells.dat", "w+");
        for (int i = 0; i < gvec.num_shells(); i++) {
            fprintf(fout, "%i %i\n", i, nv[i]);
        }
        fclose(fout);
    }

    std::vector<double> test_data(gvec.num_gvec());
    std::iota(test_data.begin(), test_data.end(), 0.0);

    /* get number of G=vectors in the new distribution */
    std::vector<int> ngv_new(mpi_comm_world().size(), 0);
    for (int igloc = 0; igloc < gvec.count(); igloc++) {
        int ig   = gvec.offset() + igloc;
        int igsh = gvec.shell(ig);
        ngv_new[spl_num_gsh.local_rank(igsh)]++;
    }

    std::vector<int> ngv_to_send = ngv_new;

    mpi_comm_world().allreduce(&ngv_new[0], mpi_comm_world().size());
    if (mpi_comm_world().rank() == 0) {
        for (int i = 0; i < mpi_comm_world().size(); i++) {
            std::cout << i << " " << ngv_new[i] << std::endl;
        }
    }

    for (int rank = 0; rank < mpi_comm_world().size(); rank++) {
        if (mpi_comm_world().rank() == rank) {
            printf("rank: %i\n", rank);
            for (int igloc = 0; igloc < gvec.count(); igloc++) {
                int ig        = gvec.offset() + igloc;
                int igsh      = gvec.shell(ig);
                auto location = spl_num_gsh.location(igsh);
                printf("ig: %i, shell: %i -> rank: %i\n", ig, igsh, location.rank);
            }
            for (int rank1 = 0; rank1 < mpi_comm_world().size(); rank1++) {
                printf("send to %i rank %i elements\n", rank1, ngv_to_send[rank1]);
            }
        }
    }

    block_data_descriptor a2a_from_gvec(mpi_comm_world().size());
    block_data_descriptor a2a_to_gsh(mpi_comm_world().size());

    for (int igloc = 0; igloc < gvec.count(); igloc++) {
        int ig   = gvec.offset() + igloc;
        int igsh = gvec.shell(ig);
        a2a_from_gvec.counts[spl_num_gsh.local_rank(igsh)]++;
    }
    a2a_from_gvec.calc_offsets();

    if (a2a_from_gvec.size() != gvec.count()) {
        RTE_THROW("wrong number of G-vectors");
    }

    /* repack data */
    std::vector<double> tmp_buf(gvec.count());
    std::fill(a2a_from_gvec.counts.begin(), a2a_from_gvec.counts.end(), 0);
    for (int igloc = 0; igloc < gvec.count(); igloc++) {
        int ig                                                      = gvec.offset() + igloc;
        int igsh                                                    = gvec.shell(ig);
        int r                                                       = spl_num_gsh.local_rank(igsh);
        tmp_buf[a2a_from_gvec.offsets[r] + a2a_from_gvec.counts[r]] = test_data[ig];
        a2a_from_gvec.counts[r]++;
    }

    for (int r = 0; r < mpi_comm_world().size(); r++) {
        for (int igloc = 0; igloc < gvec.gvec_count(r); igloc++) {
            int ig   = gvec.gvec_offset(r) + igloc;
            int igsh = gvec.shell(ig);
            if (spl_num_gsh.local_rank(igsh) == mpi_comm_world().rank()) {
                a2a_to_gsh.counts[r]++;
            }
        }
    }
    a2a_to_gsh.calc_offsets();
    printf("a2a_to_gsh.size() = %i\n", a2a_to_gsh.size());

    std::vector<double> recv_buf(a2a_to_gsh.size());

    mpi_comm_world().alltoall(tmp_buf.data(), a2a_from_gvec.counts.data(), a2a_from_gvec.offsets.data(),
                              recv_buf.data(), a2a_to_gsh.counts.data(), a2a_to_gsh.offsets.data());

    mpi_comm_world().alltoall(recv_buf.data(), a2a_to_gsh.counts.data(), a2a_to_gsh.offsets.data(), tmp_buf.data(),
                              a2a_from_gvec.counts.data(), a2a_from_gvec.offsets.data());

    std::vector<double> test_data1(gvec.num_gvec(), 0);

    std::fill(a2a_from_gvec.counts.begin(), a2a_from_gvec.counts.end(), 0);
    for (int igloc = 0; igloc < gvec.count(); igloc++) {
        int ig         = gvec.offset() + igloc;
        int igsh       = gvec.shell(ig);
        int r          = spl_num_gsh.local_rank(igsh);
        test_data1[ig] = tmp_buf[a2a_from_gvec.offsets[r] + a2a_from_gvec.counts[r]];
        a2a_from_gvec.counts[r]++;
    }

    mpi_comm_world().allreduce(test_data1.data(), gvec.num_gvec());

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        if (std::abs(test_data[ig] - test_data1[ig]) > 1e-12) {
            RTE_THROW("wrong data was collected");
        }
    }

    // auto& gvp = gvec.partition();

    // for (int i = 0; i < gvp.num_zcol(); i++) {
    //     for (size_t j = 0; j < gvp.zcol(i).z.size(); j++) {
    //         printf("icol: %i idx: %li z: %i\n", i, j, gvp.zcol(i).z[j]);
    //     }
    // }
}

void
test_gvec(double cutoff__, bool reduce__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    // FFT3D_grid fft_box(2.01 * cutoff__, M);
    // FFT3D_grid fft_box(cutoff__, M);

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, mpi_comm_world(), mpi_comm_world(), reduce__);

    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto G = gvec.gvec(ig);
        // printf("ig: %i, G: %i %i %i\n", ig, G[0], G[1], G[2]);
        auto idx = gvec.index_by_gvec(G);
        if (idx != ig) {
            std::stringstream s;
            s << "wrong reverce index" << std::endl << "direct index: " << ig << std::endl << "reverce index: " << idx;
            RTE_THROW(s);
        }
    }
}

int
main(int argn, char** argv)
{
    cmd_args args;

    args.register_key("--cutoff=", "{double} wave-functions cutoff");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto cutoff = args.value<double>("cutoff", 2.0);

    sirius::initialize(1);
    test_gvec_distr(cutoff);
    test_gvec(cutoff, false);
    test_gvec(cutoff, true);
    sirius::finalize();
}

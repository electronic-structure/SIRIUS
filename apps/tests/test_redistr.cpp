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
test_redistr(std::vector<int> mpi_grid_dims, int M, int N)
{
    if (mpi_grid_dims.size() != 2) {
        RTE_THROW("2d MPI grid is expected");
    }

    MPI_Win win;

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims[0], mpi_grid_dims[1]);

    dmatrix<double> mtrx(M, N, blacs_grid, 16, 16);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mtrx.set(j, i, double((j + 1) * (i + 1)));
        }
    }

    splindex<block> spl_row(M, mpi_comm_world().size(), mpi_comm_world().rank());
    matrix<double> mtrx2(spl_row.local_size(), N);

    MPI_Win_create(mtrx2.at<CPU>(), mtrx2.size(), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_fence(0, win);

    runtime::Timer t1("MPI_Put");

    for (int icol = 0; icol < mtrx.num_cols_local(); icol++) {
        int icol_glob = mtrx.icol(icol);
        for (int irow = 0; irow < mtrx.num_rows_local(); irow++) {
            int irow_glob = mtrx.irow(irow);

            auto location = spl_row.location(irow_glob);
            MPI_Put(&mtrx(irow, icol), 1, mpi_type_wrapper<double>::kind(), location.second,
                    icol_glob * spl_row.local_size(location.second) + location.first, 1,
                    mpi_type_wrapper<double>::kind(), win);
        }
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    double tval = t1.stop();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < spl_row.local_size(); j++) {
            int jglob = spl_row[j];
            if (std::abs(mtrx2(j, i) - double((jglob + 1) * (i + 1))) > 1e-14) {
                RTE_THROW("error");
            }
            // pout.printf("%4i ", mtrx2(j, i));
        }
        // pout.printf("\n");
    }

    printf("time: %f\n", tval);
}

struct element_pack
{
    int data;
    int idx;
};

void
test_redistr2()
{
    int N = 12;
    splindex<block_cyclic> spl1(N, mpi_comm_world().size(), mpi_comm_world().rank(), 2);
    splindex<block> spl2(N, mpi_comm_world().size(), mpi_comm_world().rank());

    std::vector<int> data1(spl1.local_size());
    std::vector<int> data2(spl2.local_size());

    for (int i = 0; i < spl1.local_size(); i++) {
        data1[i] = spl1[i];
    }

    block_data_descriptor sd(mpi_comm_world().size());
    std::vector<std::vector<element_pack>> sbuf(mpi_comm_world().size());
    /* local set of sending elements */
    for (int i = 0; i < spl1.local_size(); i++) {
        /* location on the receiving side */
        auto loc = spl2.location(spl1[i]);
        sd.counts[loc.second] += static_cast<int>(sizeof(element_pack));
        element_pack p;
        p.data = data1[i];
        p.idx  = loc.first;
        sbuf[loc.second].push_back(p);
    }
    sd.calc_offsets();
    std::vector<element_pack> sbuf_tot;
    for (int r = 0; r < mpi_comm_world().size(); r++) {
        sbuf_tot.insert(sbuf_tot.end(), sbuf[r].begin(), sbuf[r].end());
    }

    block_data_descriptor rd(mpi_comm_world().size());
    for (int i = 0; i < spl2.local_size(); i++) {
        auto loc = spl1.location(spl2[i]);
        rd.counts[loc.second] += static_cast<int>(sizeof(element_pack));
    }
    rd.calc_offsets();

    std::vector<element_pack> rbuf_tot(spl2.local_size());

    mpi_comm_world().alltoall((char*)sbuf_tot.data(), sd.counts.data(), sd.offsets.data(), (char*)rbuf_tot.data(),
                              rd.counts.data(), rd.offsets.data());

    runtime::pstdout pout(mpi_comm_world());
    pout.printf("rank: %i\n", mpi_comm_world().rank());
    for (int i = 0; i < spl2.local_size(); i++) {
        // int j = spl2[i];
        // auto loc = spl1.location(j);
        // data2[i] = rbuf_tot[rd.offsets[loc.secod] + loc.first
        // data2[rbuf_tot[i].idx] = rbuf_tot[i].data;
        pout.printf("data: %i idx: %i\n", rbuf_tot[i].data, rbuf_tot[i].idx);
        data2[rbuf_tot[i].idx] = rbuf_tot[i].data;
        pout.printf("%i\n", data2[i]);
    }
    pout.flush();
    for (int i = 0; i < spl2.local_size(); i++) {
        pout.printf("%i\n", data2[i]);
    }
    // for (int i = 0; i < spl2.local_size(); i++) {
    //     auto loc = spl1.location(spl2[i]);
    //     data1[i] = rbuf_tot[rd.offsets[loc.secod] + loc.irst
    // }
    //
}

void
test_redistr3()
{
    int N = 12;
    splindex<block_cyclic> spl1(N, mpi_comm_world().size(), mpi_comm_world().rank(), 2);
    splindex<block> spl2(N, mpi_comm_world().size(), mpi_comm_world().rank());

    std::vector<int> data1(spl1.local_size());
    std::vector<int> data2(spl2.local_size());

    for (int i = 0; i < spl1.local_size(); i++) {
        data1[i] = spl1[i];
    }

    block_data_descriptor sd(mpi_comm_world().size());
    std::vector<std::vector<int>> sbuf(mpi_comm_world().size());
    /* local set of sending elements */
    for (int i = 0; i < spl1.local_size(); i++) {
        /* location on the receiving side */
        auto loc = spl2.location(spl1[i]);
        sd.counts[loc.second]++;
        sbuf[loc.second].push_back(data1[i]);
    }
    sd.calc_offsets();
    std::vector<int> sbuf_tot;
    for (int r = 0; r < mpi_comm_world().size(); r++) {
        sbuf_tot.insert(sbuf_tot.end(), sbuf[r].begin(), sbuf[r].end());
    }

    block_data_descriptor rd(mpi_comm_world().size());
    for (int i = 0; i < spl2.local_size(); i++) {
        auto loc = spl1.location(spl2[i]);
        rd.counts[loc.second]++;
    }
    rd.calc_offsets();

    std::vector<int> rbuf_tot(spl2.local_size());

    mpi_comm_world().alltoall(sbuf_tot.data(), sd.counts.data(), sd.offsets.data(), rbuf_tot.data(), rd.counts.data(),
                              rd.offsets.data());

    rd.counts = std::vector<int>(mpi_comm_world().size(), 0);
    runtime::pstdout pout(mpi_comm_world());
    pout.printf("rank: %i\n", mpi_comm_world().rank());
    for (int i = 0; i < spl2.local_size(); i++) {
        int j    = spl2[i];
        auto loc = spl1.location(j);
        data2[i] = rbuf_tot[rd.offsets[loc.second] + rd.counts[loc.second]];
        rd.counts[loc.second]++;
        // data2[i] = rbuf_tot[rd.offsets[loc.secod] + loc.first
        // data2[rbuf_tot[i].idx] = rbuf_tot[i].data;
        // pout.printf("data: %i idx: %i\n", rbuf_tot[i].data, rbuf_tot[i].idx);
        // data2[rbuf_tot[i].idx] = rbuf_tot[i].data;
        // pout.printf("%i\n", data2[i]);
    }
    pout.flush();
    for (int i = 0; i < spl2.local_size(); i++) {
        pout.printf("%i\n", data2[i]);
    }
    // for (int i = 0; i < spl2.local_size(); i++) {
    //     auto loc = spl1.location(spl2[i]);
    //     data1[i] = rbuf_tot[rd.offsets[loc.secod] + loc.irst
    // }
    //
}

void
test_redistr4(std::vector<int> mpi_grid_dims, int M, int N)
{
    if (mpi_grid_dims.size() != 2) {
        RTE_THROW("2d MPI grid is expected");
    }

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims[0], mpi_grid_dims[1]);

    dmatrix<double> mtrx(M, N, blacs_grid, 16, 16);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mtrx.set(j, i, double((j + 1) * (i + 1)));
        }
    }

    /* cache cartesian ranks */
    mdarray<int, 2> cart_rank(mtrx.blacs_grid().num_ranks_row(), mtrx.blacs_grid().num_ranks_col());
    for (int i = 0; i < mtrx.blacs_grid().num_ranks_col(); i++) {
        for (int j = 0; j < mtrx.blacs_grid().num_ranks_row(); j++) {
            cart_rank(j, i) = mtrx.blacs_grid().cart_rank(j, i);
        }
    }

    splindex<block> spl_row(M, mpi_comm_world().size(), mpi_comm_world().rank());
    matrix<double> mtrx2(spl_row.local_size(), N);

    block_data_descriptor sd(mpi_comm_world().size());
    block_data_descriptor rd(mpi_comm_world().size());
    std::vector<double> sbuf(mtrx.size());
    std::vector<double> rbuf(mtrx2.size());

    std::vector<int> recv_row_rank(spl_row.local_size());
    for (int irow = 0; irow < spl_row.local_size(); irow++) {
        recv_row_rank[irow] = mtrx.spl_row().local_rank(spl_row[irow]);
    }

    auto pack = [&mtrx, &cart_rank, &spl_row, &recv_row_rank, &sd, &rd, &sbuf]() {
        std::vector<int> row_rank(mtrx.num_rows_local());
        /* cache receiving ranks */
        for (int irow = 0; irow < mtrx.num_rows_local(); irow++) {
            int rank       = spl_row.local_rank(mtrx.irow(irow));
            row_rank[irow] = rank;
            sd.counts[rank] += mtrx.num_cols_local();
        }
        sd.calc_offsets();
        sd.counts = std::vector<int>(mpi_comm_world().size(), 0);
        /* pack for all rank */
        for (int icol = 0; icol < mtrx.num_cols_local(); icol++) {
            for (int irow = 0; irow < mtrx.num_rows_local(); irow++) {
                int rank                                 = row_rank[irow];
                sbuf[sd.offsets[rank] + sd.counts[rank]] = mtrx(irow, icol);
                sd.counts[rank]++;
            }
        }

        /* compute receiving counts and offsets */
        for (int icol = 0; icol < mtrx.num_cols(); icol++) {
            auto location_col = mtrx.spl_col().location(icol);
            for (int irow = 0; irow < spl_row.local_size(); irow++) {
                rd.counts[cart_rank(recv_row_rank[irow], location_col.second)]++;
            }
        }
        rd.calc_offsets();
    };

    runtime::Timer t1("pack");
    pack();
    t1.stop();

    runtime::Timer t2("a2a");
    mpi_comm_world().alltoall(sbuf.data(), sd.counts.data(), sd.offsets.data(), rbuf.data(), rd.counts.data(),
                              rd.offsets.data());
    t2.stop();

    runtime::Timer t3("unpack");
    rd.counts = std::vector<int>(mpi_comm_world().size(), 0);

    for (int icol = 0; icol < N; icol++) {
        auto location_col = mtrx.spl_col().location(icol);
        for (int irow = 0; irow < spl_row.local_size(); irow++) {
            int rank          = cart_rank(recv_row_rank[irow], location_col.second);
            mtrx2(irow, icol) = rbuf[rd.offsets[rank] + rd.counts[rank]];
            rd.counts[rank]++;
        }
    }
    t3.stop();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < spl_row.local_size(); j++) {
            int jglob = spl_row[j];
            if (std::abs(mtrx2(j, i) - double((jglob + 1) * (i + 1))) > 1e-14) {
                RTE_THROW("error");
            }
            // pout.printf("%4i ", mtrx2(j, i));
        }
        // pout.printf("\n");
    }
}

void
test_redistr5(std::vector<int> mpi_grid_dims, int M, int N)
{
    if (mpi_grid_dims.size() != 2) {
        RTE_THROW("2d MPI grid is expected");
    }

    BLACS_grid blacs_grid(mpi_comm_world(), mpi_grid_dims[0], mpi_grid_dims[1]);

    dmatrix<double> mtrx(M, N, blacs_grid, 2, 2);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mtrx.set(j, i, double((j + 1) * (i + 1)));
        }
    }

    /* cache cartesian ranks */
    mdarray<int, 2> cart_rank(mtrx.blacs_grid().num_ranks_row(), mtrx.blacs_grid().num_ranks_col());
    for (int i = 0; i < mtrx.blacs_grid().num_ranks_col(); i++) {
        for (int j = 0; j < mtrx.blacs_grid().num_ranks_row(); j++) {
            cart_rank(j, i) = mtrx.blacs_grid().cart_rank(j, i);
        }
    }

    const int BS = 4;

    mdarray<double, 1> buf(BS * BS);

    block_data_descriptor sd(mpi_comm_world().size());

    for (int icol = 0; icol < BS; icol++) {
        auto pos_icol = mtrx.spl_col().location(icol);
        for (int irow = 0; irow < BS; irow++) {
            auto pos_irow = mtrx.spl_row().location(irow);
            sd.counts[cart_rank(pos_icol.second, pos_irow.second)]++;
        }
    }
    if (mpi_comm_world().rank() == 0) {
        for (int i = 0; i < mpi_comm_world().size(); i++) {
            printf("%i\n", sd.counts[i]);
        }
    }

    sd.calc_offsets();
    buf.zero();
    int k{0};
    for (int icol = 0; icol < BS; icol++) {
        auto pos_icol = mtrx.spl_col().location(icol);
        for (int irow = 0; irow < BS; irow++) {
            auto pos_irow = mtrx.spl_row().location(irow);
            if (pos_icol.second == mtrx.rank_col() && pos_irow.second == mtrx.rank_row()) {
                buf[sd.offsets[cart_rank(pos_irow.second, pos_icol.second)] + k] = mtrx(pos_irow.first, pos_icol.first);
                k++;
            }
        }
    }

    mpi_comm_world().allgather(&buf[0], sd.counts.data(), sd.offsets.data());

    mdarray<double, 1> buf1(BS * BS);

    sd.counts = std::vector<int>(mpi_comm_world().size(), 0);

    for (int icol = 0; icol < BS; icol++) {
        auto pos_icol = mtrx.spl_col().location(icol);
        for (int irow = 0; irow < BS; irow++) {
            auto pos_irow = mtrx.spl_row().location(irow);
            int rank      = cart_rank(pos_irow.second, pos_icol.second);

            buf1[irow + icol * BS] = buf[sd.offsets[rank] + sd.counts[rank]];
            sd.counts[rank]++;
        }
    }

    if (mpi_comm_world().rank() == 0) {
        for (int i = 0; i < BS; i++) {
            for (int j = 0; j < BS; j++) {
                printf("%4.2f ", buf1[i + j * BS]);
            }
            printf("\n");
        }
    }
    // splindex<block> spl_row(M, mpi_comm_world().size(), mpi_comm_world().rank());
    // matrix<double> mtrx2(spl_row.local_size(), N);

    // block_data_descriptor sd(mpi_comm_world().size());
    // block_data_descriptor rd(mpi_comm_world().size());
    // std::vector<double> sbuf(mtrx.size());
    // std::vector<double> rbuf(mtrx2.size());

    // std::vector<int> recv_row_rank(spl_row.local_size());
    // for (int irow = 0; irow < spl_row.local_size(); irow++) {
    //     recv_row_rank[irow] = mtrx.spl_row().local_rank(spl_row[irow]);
    // }

    // auto pack = [&mtrx, &cart_rank, &spl_row, &recv_row_rank, &sd, &rd, &sbuf]()
    //{
    //     std::vector<int> row_rank(mtrx.num_rows_local());
    //     /* cache receiving ranks */
    //     for (int irow = 0; irow < mtrx.num_rows_local(); irow++) {
    //         int rank = spl_row.local_rank(mtrx.irow(irow));
    //         row_rank[irow] = rank;
    //         sd.counts[rank] += mtrx.num_cols_local();
    //     }
    //     sd.calc_offsets();
    //     sd.counts = std::vector<int>(mpi_comm_world().size(), 0);
    //     /* pack for all rank */
    //     for (int icol = 0; icol < mtrx.num_cols_local(); icol++) {
    //         for (int irow = 0; irow < mtrx.num_rows_local(); irow++) {
    //             int rank = row_rank[irow];
    //             sbuf[sd.offsets[rank] + sd.counts[rank]] = mtrx(irow, icol);
    //             sd.counts[rank]++;
    //         }
    //     }
    //
    //     /* compute receiving counts and offsets */
    //     for (int icol = 0; icol < mtrx.num_cols(); icol++) {
    //         auto location_col = mtrx.spl_col().location(icol);
    //         for (int irow = 0; irow < spl_row.local_size(); irow++) {
    //             rd.counts[cart_rank(recv_row_rank[irow], location_col.second)]++;
    //         }
    //     }
    //     rd.calc_offsets();
    // };
    //
    // runtime::Timer t1("pack");
    // pack();
    // t1.stop();

    // runtime::Timer t2("a2a");
    // mpi_comm_world().alltoall(sbuf.data(), sd.counts.data(), sd.offsets.data(),
    //                           rbuf.data(), rd.counts.data(), rd.offsets.data());
    // t2.stop();

    // runtime::Timer t3("unpack");
    // rd.counts = std::vector<int>(mpi_comm_world().size(), 0);

    // for (int icol = 0; icol < N; icol++) {
    //     auto location_col = mtrx.spl_col().location(icol);
    //     for (int irow = 0; irow < spl_row.local_size(); irow++) {
    //         int rank = cart_rank(recv_row_rank[irow], location_col.second);
    //         mtrx2(irow, icol) = rbuf[rd.offsets[rank] + rd.counts[rank]];
    //         rd.counts[rank]++;
    //     }
    // }
    // t3.stop();

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < spl_row.local_size(); j++) {
    //         int jglob = spl_row[j];
    //         if (std::abs(mtrx2(j, i) - double((jglob + 1) * (i + 1))) > 1e-14) {
    //             RTE_THROW("error");
    //         }
    //         //pout.printf("%4i ", mtrx2(j, i));
    //     }
    //     //pout.printf("\n");
    // }
}

int
main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--M=", "{int} global number of matrix rows");
    args.register_key("--N=", "{int} global number of matrix columns");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto mpi_grid_dims = args.value<std::vector<int>>("mpi_grid_dims", {1, 1});
    auto M             = args.value<int>("M", 10000);
    auto N             = args.value<int>("N", 1000);

    sirius::initialize(1);
    // test_redistr(mpi_grid_dims, M, N);
    // test_redistr3();
    test_redistr5(mpi_grid_dims, M, N);
    runtime::Timer::print();
    sirius::finalize();
}

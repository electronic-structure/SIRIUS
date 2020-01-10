// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file wf_trans.cpp
 *
 *  \brief Definitions.
 *
 */

#include "wf_trans.hpp"
#include "utils/profiler.hpp"

namespace sddk {

namespace { // local functions -> no internal linkage
template <typename T>
void transform_local(linalg_t la__, int ispn__, T* alpha__, Wave_functions* wf_in__, int i0__, int m__, T* mtrx__,
                     int ld__, Wave_functions* wf_out__, int j0__, int n__, stream_id sid__);

template <>
void transform_local<double>(linalg_t la__, int ispn__, double* alpha__, Wave_functions* wf_in__, int i0__, int m__,
                             double* mtrx__, int ld__, Wave_functions* wf_out__, int j0__, int n__, stream_id sid__)
{
    PROFILE("sddk::transform|local");

    auto spins = spin_range(ispn__);

    for (int s : spins) {
        /* input wave-functions may be scalar (this is the case of transformation of first-variational states
           into spinor wave-functions or transforamtion of scalar auxiliary wave-functions into spin-dependent
           wave-fucntions; in this case we set spin index of input wave-function to 0 */
        int in_s = (wf_in__->num_sc() == 1) ? 0 : s;
        /* transform plane-wave part */
        linalg(la__).gemm(
            'N', 'N', 2 * wf_in__->pw_coeffs(in_s).num_rows_loc(), n__, m__, alpha__,
            reinterpret_cast<double*>(wf_in__->pw_coeffs(in_s).prime().at(wf_in__->preferred_memory_t(), 0, i0__)),
            2 * wf_in__->pw_coeffs(in_s).prime().ld(), mtrx__, ld__, &linalg_const<double>::one(),
            reinterpret_cast<double*>(wf_out__->pw_coeffs(s).prime().at(wf_out__->preferred_memory_t(), 0, j0__)),
            2 * wf_out__->pw_coeffs(s).prime().ld(), sid__);
        if (wf_in__->has_mt()) {
            TERMINATE("not implemented");
        }
    }
}

template <>
void transform_local<double_complex>(linalg_t la__, int ispn__, double_complex* alpha__, Wave_functions* wf_in__,
                                     int i0__, int m__, double_complex* mtrx__, int ld__, Wave_functions* wf_out__,
                                     int j0__, int n__, stream_id sid__)
{
    PROFILE("sddk::transform|local");

    auto spins = spin_range(ispn__);

    for (int s : spins) {
        /* input wave-functions may be scalar (this is the case of transformation of first-variational states
           into spinor wave-functions or transforamtion of scalar auxiliary wave-functions into spin-dependent
           wave-fucntions; in this case we set spin index of input wave-function to 0 */
        int in_s = (wf_in__->num_sc() == 1) ? 0 : s;
        /* transform plane-wave part */
        linalg(la__).gemm('N', 'N', wf_in__->pw_coeffs(in_s).num_rows_loc(), n__, m__, alpha__,
                           wf_in__->pw_coeffs(in_s).prime().at(wf_in__->preferred_memory_t(), 0, i0__),
                           wf_in__->pw_coeffs(in_s).prime().ld(), mtrx__, ld__, &linalg_const<double_complex>::one(),
                           wf_out__->pw_coeffs(s).prime().at(wf_out__->preferred_memory_t(), 0, j0__),
                           wf_out__->pw_coeffs(s).prime().ld(), sid__);
        /* transform muffin-tin part */
        if (wf_in__->has_mt()) {
            linalg(la__).gemm('N', 'N', wf_in__->mt_coeffs(in_s).num_rows_loc(), n__, m__, alpha__,
                               wf_in__->mt_coeffs(in_s).prime().at(wf_in__->preferred_memory_t(), 0, i0__),
                               wf_in__->mt_coeffs(in_s).prime().ld(), mtrx__, ld__,
                               &linalg_const<double_complex>::one(),
                               wf_out__->mt_coeffs(s).prime().at(wf_out__->preferred_memory_t(), 0, j0__),
                               wf_out__->mt_coeffs(s).prime().ld(), sid__);
        }
    }
}
} // namespace

template <typename T>
void transform(memory_t mem__, linalg_t la__, int ispn__, double alpha__, std::vector<Wave_functions*> wf_in__,
               int i0__, int m__, dmatrix<T>& mtrx__, int irow0__, int jcol0__, double beta__,
               std::vector<Wave_functions*> wf_out__, int j0__, int n__)
{
    PROFILE("sddk::transform");

    assert(n__ != 0);
    assert(m__ != 0);
    assert(wf_in__.size() == wf_out__.size());

    int nwf    = static_cast<int>(wf_in__.size());
    auto& comm = mtrx__.comm();

    double ngop{0};
    if (std::is_same<T, double>::value) {
        ngop = 2e-9;
    }
    if (std::is_same<T, double_complex>::value) {
        ngop = 8e-9;
    }

    auto sddk_pp = utils::get_env<int>("SDDK_PRINT_PERFORMANCE");

    auto sddk_bs_raw    = utils::get_env<int>("SDDK_TRANS_BLOCK_SIZE");
    int sddk_block_size = (sddk_bs_raw == nullptr) ? sddk_trans_default_block_size : *sddk_bs_raw;

    T alpha = alpha__;

    PROFILE_START("sddk::transform|init");
    /* initial values for the resulting wave-functions */
    for (int iv = 0; iv < nwf; iv++) {
        if (beta__ == 0) {
            wf_out__[iv]->zero(get_device_t(wf_out__[iv]->preferred_memory_t()), ispn__, j0__, n__);
        } else {
            wf_out__[iv]->scale(wf_out__[iv]->preferred_memory_t(), ispn__, j0__, n__, beta__);
        }
    }
    PROFILE_STOP("sddk::transform|init");

    if (sddk_pp) {
        comm.barrier();
    }
    double time = -omp_get_wtime();

    /* trivial case */
    if (comm.size() == 1) {
        if (is_device_memory(mem__)) {
            // acc::copyin(mtrx__.at(memory_t::device, irow0__, jcol0__), mtrx__.ld(),
            //            mtrx__.at(memory_t::host, irow0__, jcol0__), mtrx__.ld(), m__, n__, stream_id(0));
            acc::copyin(mtrx__.at(memory_t::device, irow0__, jcol0__), mtrx__.ld(),
                        mtrx__.at(memory_t::host, irow0__, jcol0__), mtrx__.ld(), m__, n__);
        }
        for (int iv = 0; iv < nwf; iv++) {
            transform_local(la__, ispn__, &alpha, wf_in__[iv], i0__, m__, mtrx__.at(mem__, irow0__, jcol0__),
                            mtrx__.ld(), wf_out__[iv], j0__, n__, stream_id(iv));
        }
        if (is_device_memory(mem__)) {
            /* wait for the stream to finish zgemm */
            for (int iv = 0; iv < nwf; iv++) {
                acc::sync_stream(stream_id(iv));
            }
        }
        if (sddk_pp) {
            time += omp_get_wtime();
            int k = wf_in__[0]->gkvec().num_gvec() + wf_in__[0]->num_mt_coeffs();
            std::printf("transform() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, nvec=%i, time=%f (sec)]\n",
                   ngop * m__ * n__ * k * nwf / time, k, n__, m__, nwf, time);
        }
        return;
    }

    const int BS = sddk_block_size;

    int num_streams = std::min(4, omp_get_max_threads());

    static T* ptr_h{nullptr};
    static T* ptr_d{nullptr};
    static mdarray<T, 1> buf(BS * BS, memory_t::host, "transform::buf");
    if (is_device_memory(mem__)) {
        if (!ptr_h) {
            ptr_h = sddk::allocate<T>(BS * BS * num_streams, memory_t::host_pinned);
        }
        if (!ptr_d) {
            ptr_d = sddk::allocate<T>(BS * BS * num_streams, memory_t::device);
        }
    } else {
        if (!ptr_h) {
            ptr_h = sddk::allocate<T>(BS * BS * num_streams, memory_t::host);
        }
    }
    mdarray<T, 3> submatrix(ptr_h, ptr_d, BS, BS, num_streams, "transform::submatrix");

    /* cache cartesian ranks */
    mdarray<int, 2> cart_rank(mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().num_ranks_col());
    for (int i = 0; i < mtrx__.blacs_grid().num_ranks_col(); i++) {
        for (int j = 0; j < mtrx__.blacs_grid().num_ranks_row(); j++) {
            cart_rank(j, i) = mtrx__.blacs_grid().cart_rank(j, i);
        }
    }

    int nbr = m__ / BS + std::min(1, m__ % BS);
    int nbc = n__ / BS + std::min(1, n__ % BS);

    block_data_descriptor sd(comm.size());

    double time_mpi{0};

    for (int ibr = 0; ibr < nbr; ibr++) {
        /* global index of row */
        int i0 = ibr * BS;
        /* actual number of rows in the submatrix */
        int nrow = std::min(m__, (ibr + 1) * BS) - i0;

        assert(nrow != 0);

        splindex<splindex_t::block_cyclic> spl_row_begin(irow0__ + i0, mtrx__.num_ranks_row(), mtrx__.rank_row(),
                                                         mtrx__.bs_row());
        splindex<splindex_t::block_cyclic> spl_row_end(irow0__ + i0 + nrow, mtrx__.num_ranks_row(), mtrx__.rank_row(),
                                                       mtrx__.bs_row());

        int local_size_row = spl_row_end.local_size() - spl_row_begin.local_size();

        int s{0};
        for (int ibc = 0; ibc < nbc; ibc++) {
            /* global index of column */
            int j0 = ibc * BS;
            /* actual number of columns in the submatrix */
            int ncol = std::min(n__, (ibc + 1) * BS) - j0;

            assert(ncol != 0);

            splindex<splindex_t::block_cyclic> spl_col_begin(jcol0__ + j0, mtrx__.num_ranks_col(), mtrx__.rank_col(),
                                                             mtrx__.bs_col());
            splindex<splindex_t::block_cyclic> spl_col_end(jcol0__ + j0 + ncol, mtrx__.num_ranks_col(),
                                                           mtrx__.rank_col(), mtrx__.bs_col());

            int local_size_col = spl_col_end.local_size() - spl_col_begin.local_size();

            /* total number of elements owned by the current rank in the block */
            for (int i = 0; i < mtrx__.blacs_grid().num_ranks_col(); i++) {
                int scol = spl_col_end.local_size(i) - spl_col_begin.local_size(i);
                for (int j = 0; j < mtrx__.blacs_grid().num_ranks_row(); j++) {
                    int l        = cart_rank(j, i);
                    sd.counts[l] = (spl_row_end.local_size(j) - spl_row_begin.local_size(j)) * scol;
                }
            }

            sd.calc_offsets();

            assert(sd.offsets.back() + sd.counts.back() <= (int)buf.size());
            /* fetch elements of sub-matrix */
            if (local_size_row) {
                for (int j = 0; j < local_size_col; j++) {
                    std::memcpy(&buf[sd.offsets[comm.rank()] + local_size_row * j],
                                &mtrx__(spl_row_begin.local_size(), spl_col_begin.local_size() + j),
                                local_size_row * sizeof(T));
                }
            }
            auto t = std::chrono::high_resolution_clock::now();
            PROFILE_START("sddk::transform|mpi");
            /* collect submatrix */
            comm.allgather(&buf[0], sd.counts.data(), sd.offsets.data());
            PROFILE_STOP("sddk::transform|mpi");
            auto dt = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t).count();
            time_mpi += dt;

            if (is_device_memory(mem__)) {
                /* wait for the data copy; as soon as this is done, CPU buffer is free and can be reused */
                acc::sync_stream(stream_id(s % num_streams));
            }

            /* unpack data */
            std::vector<int> counts(comm.size(), 0);
            for (int jcol = 0; jcol < ncol; jcol++) {
                auto pos_jcol = mtrx__.spl_col().location(jcol0__ + j0 + jcol);
                for (int irow = 0; irow < nrow; irow++) {
                    auto pos_irow = mtrx__.spl_row().location(irow0__ + i0 + irow);
                    int rank      = cart_rank(pos_irow.rank, pos_jcol.rank);

                    submatrix(irow, jcol, s % num_streams) = buf[sd.offsets[rank] + counts[rank]];
                    counts[rank]++;
                }
            }
            for (int rank = 0; rank < comm.size(); rank++) {
                assert(sd.counts[rank] == counts[rank]);
            }
            if (is_device_memory(mem__)) {
                acc::copyin(submatrix.at(memory_t::device, 0, 0, s % num_streams), submatrix.ld(),
                            submatrix.at(memory_t::host, 0, 0, s % num_streams), submatrix.ld(), nrow, ncol,
                            stream_id(s % num_streams));
            }

            T* ptr = submatrix.at(mem__, 0, 0, s % num_streams);

            for (int iv = 0; iv < nwf; iv++) {
                transform_local(la__, ispn__, &alpha, wf_in__[iv], i0__ + i0, nrow, ptr, BS, wf_out__[iv], j0__ + j0,
                                ncol, stream_id(s % num_streams));
            }
            s++;
        } /* loop over ibc */
        if (is_device_memory(mem__)) {
            /* wait for the full block of columns (update of different wave-functions);
             * otherwise cuda streams can start updating the same block of output wave-functions */
            for (int s = 0; s < num_streams; s++) {
                acc::sync_stream(stream_id(s));
            }
        }
    } /* loop over ibr */

    if (sddk_pp) {
        comm.barrier();
        time += omp_get_wtime();
        int k = wf_in__[0]->gkvec().num_gvec() + wf_in__[0]->num_mt_coeffs();
        if (comm.rank() == 0) {
            std::printf("transform() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, nvec=%i, time=%f (sec), time_mpi=%f "
                   "(sec)]\n",
                   ngop * m__ * n__ * k * nwf / time / comm.size(), k, n__, m__, nwf, time, time_mpi);
        }
    }
}

// instantiate for required types
template void transform<double>(memory_t mem__, linalg_t la__, int ispn__, double alpha__,
                                std::vector<Wave_functions*> wf_in__, int i0__, int m__, dmatrix<double>& mtrx__,
                                int irow0__, int jcol0__, double beta__, std::vector<Wave_functions*> wf_out__,
                                int j0__, int n__);

template void transform<double_complex>(memory_t mem__, linalg_t la__, int ispn__, double alpha__,
                                        std::vector<Wave_functions*> wf_in__, int i0__, int m__,
                                        dmatrix<double_complex>& mtrx__, int irow0__, int jcol0__, double beta__,
                                        std::vector<Wave_functions*> wf_out__, int j0__, int n__);
} // namespace sddk

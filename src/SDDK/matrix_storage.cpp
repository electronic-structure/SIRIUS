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

/** \file matrix_storage.cpp
 *
 *  \brief Definitions.
 *
 */
#include "matrix_storage.hpp"
#include "utils/profiler.hpp"

namespace sddk {

template <typename T>
void matrix_storage<T, matrix_storage_t::slab>::set_num_extra(int n__, int idx0__, memory_pool* mp__)
{
    PROFILE("sddk::matrix_storage::set_num_extra");

    auto& comm_col = gvp_->comm_ortho_fft();

    /* this is how n columns of the matrix will be distributed between columns of the MPI grid */
    spl_num_col_ = splindex<splindex_t::block>(n__, comm_col.size(), comm_col.rank());

    T* ptr{nullptr};
    T* ptr_d{nullptr};
    int ncol{0};

    /* trivial case */
    if (!is_remapped()) {
        assert(num_rows_loc_ == gvp_->gvec_count_fft());
        ncol = n__;
        ptr  = prime_.at(memory_t::host, 0, idx0__);
        if (prime_.on_device()) {
            ptr_d = prime_.at(memory_t::device, 0, idx0__);
        }
    } else {
        /* maximum local number of matrix columns */
        ncol = splindex_base<int>::block_size(n__, comm_col.size());
        /* upper limit for the size of swapped extra matrix */
        size_t sz = gvp_->gvec_count_fft() * ncol;
        /* reallocate buffers if necessary */
        if (extra_buf_.size() < sz) {
            PROFILE("sddk::matrix_storage::set_num_extra|alloc");
            if (mp__) {
                send_recv_buf_ = mdarray<T, 1>(sz, *mp__, "matrix_storage.send_recv_buf_");
                extra_buf_     = mdarray<T, 1>(sz, *mp__, "matrix_storage.extra_buf_");
            } else {
                send_recv_buf_ = mdarray<T, 1>(sz, memory_t::host, "matrix_storage.send_recv_buf_");
                extra_buf_     = mdarray<T, 1>(sz, memory_t::host, "matrix_storage.extra_buf_");
            }
        }
        ptr = extra_buf_.at(memory_t::host);
    }
    /* create the extra storage */
    extra_ = mdarray<T, 2>(ptr, ptr_d, gvp_->gvec_count_fft(), ncol, "matrix_storage.extra_");
}

template <typename T>
void matrix_storage<T, matrix_storage_t::slab>::remap_from(const dmatrix<T>& mtrx__, int irow0__)
{
    PROFILE("sddk::matrix_storage::remap_from");

    auto& comm = mtrx__.blacs_grid().comm();

    /* cache cartesian ranks */
    mdarray<int, 2> cart_rank(mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().num_ranks_col());
    for (int i = 0; i < mtrx__.blacs_grid().num_ranks_col(); i++) {
        for (int j = 0; j < mtrx__.blacs_grid().num_ranks_row(); j++) {
            cart_rank(j, i) = mtrx__.blacs_grid().cart_rank(j, i);
        }
    }

    if (send_recv_buf_.size() < prime_.size()) {
        send_recv_buf_ = mdarray<T, 1>(prime_.size(), memory_t::host, "matrix_storage::send_recv_buf_");
    }

    block_data_descriptor rd(comm.size());
    rd.counts[comm.rank()] = num_rows_loc();
    comm.allgather(rd.counts.data(), comm.rank(), 1);
    rd.calc_offsets();

    block_data_descriptor sd(comm.size());

    /* global index of column */
    int j0 = 0;
    /* actual number of columns in the submatrix */
    int ncol = num_cols_;

    splindex<splindex_t::block_cyclic> spl_col_begin(j0, mtrx__.num_ranks_col(), mtrx__.rank_col(), mtrx__.bs_col());
    splindex<splindex_t::block_cyclic> spl_col_end(j0 + ncol, mtrx__.num_ranks_col(), mtrx__.rank_col(),
                                                   mtrx__.bs_col());

    int local_size_col = spl_col_end.local_size() - spl_col_begin.local_size();

    for (int rank_row = 0; rank_row < comm.size(); rank_row++) {
        if (!rd.counts[rank_row]) {
            continue;
        }
        /* global index of column */
        int i0 = rd.offsets[rank_row];
        /* actual number of rows in the submatrix */
        int nrow = rd.counts[rank_row];

        assert(nrow != 0);

        splindex<splindex_t::block_cyclic> spl_row_begin(irow0__ + i0, mtrx__.num_ranks_row(), mtrx__.rank_row(),
                                                         mtrx__.bs_row());
        splindex<splindex_t::block_cyclic> spl_row_end(irow0__ + i0 + nrow, mtrx__.num_ranks_row(), mtrx__.rank_row(),
                                                       mtrx__.bs_row());

        int local_size_row = spl_row_end.local_size() - spl_row_begin.local_size();

        mdarray<T, 1> buf(local_size_row * local_size_col);

        /* fetch elements of sub-matrix matrix */
        if (local_size_row) {
            for (int j = 0; j < local_size_col; j++) {
                std::memcpy(&buf[local_size_row * j],
                            &mtrx__(spl_row_begin.local_size(), spl_col_begin.local_size() + j),
                            local_size_row * sizeof(T));
            }
        }

        sd.counts[comm.rank()] = local_size_row * local_size_col;
        comm.allgather(sd.counts.data(), comm.rank(), 1);
        sd.calc_offsets();

        /* collect buffers submatrix */
        T* send_buf = (buf.size() == 0) ? nullptr : &buf[0];
        T* recv_buf = (send_recv_buf_.size() == 0) ? nullptr : &send_recv_buf_[0];
        comm.gather(send_buf, recv_buf, sd.counts.data(), sd.offsets.data(), rank_row);

        if (comm.rank() == rank_row) {
            /* unpack data */
            std::vector<int> counts(comm.size(), 0);
            for (int jcol = 0; jcol < ncol; jcol++) {
                auto pos_jcol = mtrx__.spl_col().location(j0 + jcol);
                for (int irow = 0; irow < nrow; irow++) {
                    auto pos_irow = mtrx__.spl_row().location(irow0__ + i0 + irow);
                    int rank      = cart_rank(pos_irow.rank, pos_jcol.rank);

                    prime_(irow, jcol) = send_recv_buf_[sd.offsets[rank] + counts[rank]];
                    counts[rank]++;
                }
            }
            for (int rank = 0; rank < comm.size(); rank++) {
                assert(sd.counts[rank] == counts[rank]);
            }
        }
    }
}

template <typename T>
void matrix_storage<T, matrix_storage_t::slab>::remap_backward(int n__, int idx0__)
{
    PROFILE("sddk::matrix_storage::remap_backward");

    /* trivial case when extra storage mirrors the prime storage */
    if (!is_remapped()) {
        return;
    }

    auto& comm_col = gvp_->comm_ortho_fft();

    auto& row_distr = gvp_->gvec_fft_slab();

    assert(n__ == spl_num_col_.global_index_size());

    /* local number of columns */
    int n_loc = spl_num_col_.local_size();

    /* reorder sending blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < comm_col.size(); j++) {
            int offset = row_distr.offsets[j];
            int count  = row_distr.counts[j];
            if (count) {
                std::memcpy(&send_recv_buf_[offset * n_loc + count * i], &extra_(offset, i), count * sizeof(T));
            }
        }
    }

    /* send and recieve dimensions */
    block_data_descriptor sd(comm_col.size()), rd(comm_col.size());
    for (int j = 0; j < comm_col.size(); j++) {
        sd.counts[j] = spl_num_col_.local_size(comm_col.rank()) * row_distr.counts[j];
        rd.counts[j] = spl_num_col_.local_size(j) * row_distr.counts[comm_col.rank()];
    }
    sd.calc_offsets();
    rd.calc_offsets();

    T* recv_buf = (num_rows_loc_ == 0) ? nullptr : prime_.at(memory_t::host, 0, idx0__);

    {
        PROFILE("sddk::matrix_storage::remap_backward|mpi");
        comm_col.alltoall(send_recv_buf_.at(memory_t::host), sd.counts.data(), sd.offsets.data(), recv_buf,
                          rd.counts.data(), rd.offsets.data());
    }

    /* move data back to device */
    if (prime_.on_device()) {
        prime_.copy_to(memory_t::device, idx0__ * num_rows_loc(), n__ * num_rows_loc());
    }
}

template <typename T>
void matrix_storage<T, matrix_storage_t::slab>::remap_forward(int n__, int idx0__, memory_pool* mp__)
{
    PROFILE("sddk::matrix_storage::remap_forward");

    set_num_extra(n__, idx0__, mp__);

    /* trivial case when extra storage mirrors the prime storage */
    if (!is_remapped()) {
        return;
    }

    auto& row_distr = gvp_->gvec_fft_slab();

    auto& comm_col = gvp_->comm_ortho_fft();

    /* local number of columns */
    int n_loc = spl_num_col_.local_size();

    /* send and recieve dimensions */
    block_data_descriptor sd(comm_col.size()), rd(comm_col.size());
    for (int j = 0; j < comm_col.size(); j++) {
        sd.counts[j] = spl_num_col_.local_size(j) * row_distr.counts[comm_col.rank()];
        rd.counts[j] = spl_num_col_.local_size(comm_col.rank()) * row_distr.counts[j];
    }
    sd.calc_offsets();
    rd.calc_offsets();

    T* send_buf = (num_rows_loc_ == 0) ? nullptr : prime_.at(memory_t::host, 0, idx0__);

    {
        PROFILE("sddk::matrix_storage::remap_forward|mpi");
        comm_col.alltoall(send_buf, sd.counts.data(), sd.offsets.data(), send_recv_buf_.at(memory_t::host),
                          rd.counts.data(), rd.offsets.data());
    }

    /* reorder recieved blocks */
    #pragma omp parallel for
    for (int i = 0; i < n_loc; i++) {
        for (int j = 0; j < comm_col.size(); j++) {
            int offset = row_distr.offsets[j];
            int count  = row_distr.counts[j];
            if (count) {
                std::memcpy(&extra_(offset, i), &send_recv_buf_[offset * n_loc + count * i], count * sizeof(T));
            }
        }
    }
}

template <typename T>
void matrix_storage<T, matrix_storage_t::slab>::scale(memory_t mem__, int i0__, int n__, double beta__)
{
    if (is_host_memory(mem__)) {
        for (int i = 0; i < n__; i++) {
            for (int j = 0; j < num_rows_loc(); j++) {
                prime(j, i0__ + i) *= beta__;
            }
        }
    } else {
#if defined(__GPU)
        scale_matrix_elements_gpu((acc_complex_double_t*)prime().at(mem__, 0, i0__), prime().ld(), num_rows_loc(), n__,
                                  beta__);
#endif
    }
}

template <>
double_complex matrix_storage<std::complex<double>, matrix_storage_t::slab>::checksum(device_t pu__, int i0__, int n__) const
{
    double_complex cs(0, 0);

    switch (pu__) {
        case device_t::CPU: {
            for (int i = 0; i < n__; i++) {
                for (int j = 0; j < num_rows_loc(); j++) {
                    cs += prime(j, i0__ + i);
                }
            }
            break;
        }
        case device_t::GPU: {
            mdarray<double_complex, 1> cs1(n__, memory_t::host, "checksum");
            cs1.allocate(memory_t::device).zero(memory_t::device);
#if defined(__GPU)
            add_checksum_gpu(prime().at(memory_t::device, 0, i0__), num_rows_loc(), n__, cs1.at(memory_t::device));
            cs1.copy_to(memory_t::host);
            cs = cs1.checksum();
#endif
            break;
        }
    }
    return cs;
}

template <>
double_complex matrix_storage<double, matrix_storage_t::slab>::checksum(device_t, int, int) const
{
    TERMINATE("matrix_storage<double, ..>::checksum is not implemented for double\n");
    return 0;
}

// instantiate required types
template class matrix_storage<double, matrix_storage_t::slab>;
template class matrix_storage<double_complex, matrix_storage_t::slab>;

} // namespace sddk

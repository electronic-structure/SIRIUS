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

/** \file dmatrix.cpp
 *
 *  \brief Definitions.
 *
 */
#include "dmatrix.hpp"

namespace la {

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__,
                    sddk::memory_t mem_type__)
    : sddk::matrix<T>(sddk::splindex_block_cyclic<>(num_rows__, n_blocks(blacs_grid__.num_ranks_row()),
                block_id(blacs_grid__.rank_row()), bs_row__).local_size(),
            sddk::splindex_block_cyclic<>(num_cols__, n_blocks(blacs_grid__.num_ranks_col()),
                block_id(blacs_grid__.rank_col()), bs_col__).local_size(), mem_type__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(bs_row__)
    , bs_col_(bs_col__)
    , blacs_grid_(&blacs_grid__)
    , spl_row_(num_rows_, n_blocks(blacs_grid__.num_ranks_row()), block_id(blacs_grid__.rank_row()), bs_row_)
    , spl_col_(num_cols_, n_blocks(blacs_grid__.num_ranks_col()), block_id(blacs_grid__.rank_col()), bs_col_)
    , spla_dist_(spla::MatrixDistribution::create_blacs_block_cyclic_from_mapping(
          blacs_grid__.comm().native(), blacs_grid__.rank_map().data(), blacs_grid__.num_ranks_row(),
          blacs_grid__.num_ranks_col(), bs_row__,bs_col__))
{
    init();
}

template <typename T>
dmatrix<T>::dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__,
                    int bs_col__)
    : sddk::matrix<T>(ptr__,
                sddk::splindex_block_cyclic<>(num_rows__, n_blocks(blacs_grid__.num_ranks_row()),
                    block_id(blacs_grid__.rank_row()), bs_row__).local_size(),
                sddk::splindex_block_cyclic<>(num_cols__, n_blocks(blacs_grid__.num_ranks_col()),
                    block_id(blacs_grid__.rank_col()), bs_col__).local_size())
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(bs_row__)
    , bs_col_(bs_col__)
    , blacs_grid_(&blacs_grid__)
    , spl_row_(num_rows_, n_blocks(blacs_grid__.num_ranks_row()), block_id(blacs_grid__.rank_row()), bs_row_)
    , spl_col_(num_cols_, n_blocks(blacs_grid__.num_ranks_col()), block_id(blacs_grid__.rank_col()), bs_col_)
    , spla_dist_(spla::MatrixDistribution::create_blacs_block_cyclic_from_mapping(
          blacs_grid__.comm().native(), blacs_grid__.rank_map().data(), blacs_grid__.num_ranks_row(),
          blacs_grid__.num_ranks_col(), bs_row__, bs_col__))
{
    init();
}

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, sddk::memory_t mem_type__)
    : sddk::matrix<T>(num_rows__, num_cols__, mem_type__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(1)
    , bs_col_(1)
    , spl_row_(num_rows_, n_blocks(1), block_id(0), bs_row_)
    , spl_col_(num_cols_, n_blocks(1), block_id(0), bs_col_)
{
}

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, sddk::memory_pool& mp__, std::string const& label__)
    : sddk::matrix<T>(num_rows__, num_cols__, mp__, label__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(1)
    , bs_col_(1)
    , spl_row_(num_rows_, n_blocks(1), block_id(0), bs_row_)
    , spl_col_(num_cols_, n_blocks(1), block_id(0), bs_col_)
{
}


template <typename T>
dmatrix<T>::dmatrix(T* ptr__, int num_rows__, int num_cols__)
    : sddk::matrix<T>(ptr__, num_rows__, num_cols__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(1)
    , bs_col_(1)
    , spl_row_(num_rows_, n_blocks(1), block_id(0), bs_row_)
    , spl_col_(num_cols_, n_blocks(1), block_id(0), bs_col_)
{
    init();
}

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__,
                    sddk::memory_pool& mp__)
    : sddk::matrix<T>(sddk::splindex_block_cyclic<>(num_rows__, n_blocks(blacs_grid__.num_ranks_row()),
                block_id(blacs_grid__.rank_row()), bs_row__).local_size(),
                sddk::splindex_block_cyclic<>(num_cols__, n_blocks(blacs_grid__.num_ranks_col()),
                    block_id(blacs_grid__.rank_col()), bs_col__).local_size(), mp__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(bs_row__)
    , bs_col_(bs_col__)
    , blacs_grid_(&blacs_grid__)
    , spl_row_(num_rows_, n_blocks(blacs_grid__.num_ranks_row()), block_id(blacs_grid__.rank_row()), bs_row_)
    , spl_col_(num_cols_, n_blocks(blacs_grid__.num_ranks_col()), block_id(blacs_grid__.rank_col()), bs_col_)
    , spla_dist_(spla::MatrixDistribution::create_blacs_block_cyclic_from_mapping(
          blacs_grid__.comm().native(), blacs_grid__.rank_map().data(), blacs_grid__.num_ranks_row(),
          blacs_grid__.num_ranks_col(), bs_row__, bs_col__))
{
    init();
}

template <typename T>
void dmatrix<T>::set(int ir0__, int jc0__, int mr__, int nc__, T* ptr__, int ld__)
{
    sddk::splindex_block_cyclic<> spl_r0(ir0__, n_blocks(blacs_grid().num_ranks_row()),
            block_id(blacs_grid().rank_row()), bs_row_);
    sddk::splindex_block_cyclic<> spl_r1(ir0__ + mr__, n_blocks(blacs_grid().num_ranks_row()),
            block_id(blacs_grid().rank_row()), bs_row_);

    sddk::splindex_block_cyclic<> spl_c0(jc0__, n_blocks(blacs_grid().num_ranks_col()),
            block_id(blacs_grid().rank_col()), bs_col_);
    sddk::splindex_block_cyclic<> spl_c1(jc0__ + nc__, n_blocks(blacs_grid().num_ranks_col()),
            block_id(blacs_grid().rank_col()), bs_col_);

    int m0 = spl_r0.local_size();
    int m1 = spl_r1.local_size();
    int n0 = spl_c0.local_size();
    int n1 = spl_c1.local_size();
    std::vector<int> map_row(m1 - m0);
    std::vector<int> map_col(n1 - n0);

    for (int i = 0; i < m1 - m0; i++) {
        map_row[i] = spl_r1.global_index(m0 + i) - ir0__;
    }
    for (int j = 0; j < n1 - n0; j++) {
        map_col[j] = spl_c1.global_index(n0 + j) - jc0__;
    }

    //#pragma omp parallel for
    for (int j = 0; j < n1 - n0; j++) {
        for (int i = 0; i < m1 - m0; i++) {
            (*this)(m0 + i, n0 + j) = ptr__[map_row[i] + ld__ * map_col[j]];
        }
    }
}

template <typename T>
void dmatrix<T>::set(const int irow_glob, const int icol_glob, T val)
{
    auto r = spl_row_.location(irow_glob);
    if (blacs_grid_->rank_row() == r.ib) {
        auto c = spl_col_.location(icol_glob);
        if (blacs_grid_->rank_col() == c.ib) {
            (*this)(r.index_local, c.index_local) = val;
        }
    }
}

template <typename T>
void dmatrix<T>::add(const int irow_glob, const int icol_glob, T val)
{
    auto r = spl_row_.location(irow_glob);
    if (blacs_grid_->rank_row() == r.ib) {
        auto c = spl_col_.location(icol_glob);
        if (blacs_grid_->rank_col() == c.ib) {
            (*this)(r.index_local, c.index_local) += val;
        }
    }
}

template <typename T>
void dmatrix<T>::add(real_type<T> beta__, const int irow_glob, const int icol_glob, T val)
{
    auto r = spl_row_.location(irow_glob);
    if (blacs_grid_->rank_row() == r.ib) {
        auto c = spl_col_.location(icol_glob);
        if (blacs_grid_->rank_col() == c.ib) {
            (*this)(r.index_local, c.index_local) = (*this)(r.index_local, c.index_local) * beta__ + val;
        }
    }
}

template <typename T>
void dmatrix<T>::make_real_diag(int n__)
{
    for (int i = 0; i < n__; i++) {
        auto r = spl_row_.location(i);
        if (blacs_grid_->rank_row() == r.ib) {
            auto c = spl_col_.location(i);
            if (blacs_grid_->rank_col() == c.ib) {
                T v                                   = (*this)(r.index_local, c.index_local);
                (*this)(r.index_local, c.index_local) = std::real(v);
            }
        }
    }
}

template <typename T>
sddk::mdarray<T, 1> dmatrix<T>::get_diag(int n__)
{
    sddk::mdarray<T, 1> d(n__);
    d.zero();

    for (int i = 0; i < n__; i++) {
        auto r = spl_row_.location(i);
        if (blacs_grid_->rank_row() == r.ib) {
            auto c = spl_col_.location(i);
            if (blacs_grid_->rank_col() == c.ib) {
                d[i] = (*this)(r.index_local, c.index_local);
            }
        }
    }
    blacs_grid_->comm().allreduce(d.template at(sddk::memory_t::host), n__);
    return d;
}

template <typename T>
void dmatrix<T>::save_to_hdf5(std::string name__, int m__, int n__)
{
    sddk::mdarray<T, 2> full_mtrx(m__, n__);
    full_mtrx.zero();

    for (int j = 0; j < this->num_cols_local(); j++) {
        for (int i = 0; i < this->num_rows_local(); i++) {
            if (this->irow(i) < m__ && this->icol(j) < n__) {
                full_mtrx(this->irow(i), this->icol(j)) = (*this)(i, j);
            }
        }
    }
    this->comm().allreduce(full_mtrx.template at(sddk::memory_t::host), static_cast<int>(full_mtrx.size()));

    if (this->blacs_grid().comm().rank() == 0) {
        sddk::HDF5_tree h5(name__, sddk::hdf5_access_t::truncate);
        h5.write("nrow", m__);
        h5.write("ncol", n__);
        h5.write("mtrx", full_mtrx);
    }
}

// instantiate for required types
template class dmatrix<double>;
template class dmatrix<std::complex<double>>;
#ifdef USE_FP32
template class dmatrix<float>;
template class dmatrix<std::complex<float>>;
#endif

} // namespace

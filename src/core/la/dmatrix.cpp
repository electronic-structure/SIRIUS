/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file dmatrix.cpp
 *
 *  \brief Definitions.
 *
 */
#include "dmatrix.hpp"

namespace sirius {

namespace la {

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__,
                    memory_t mem_type__)
    : matrix<T>({splindex_block_cyclic<>(num_rows__, n_blocks(blacs_grid__.num_ranks_row()),
                                         block_id(blacs_grid__.rank_row()), bs_row__)
                         .local_size(),
                 splindex_block_cyclic<>(num_cols__, n_blocks(blacs_grid__.num_ranks_col()),
                                         block_id(blacs_grid__.rank_col()), bs_col__)
                         .local_size()},
                mem_type__)
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
dmatrix<T>::dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__,
                    int bs_col__)
    : matrix<T>({splindex_block_cyclic<>(num_rows__, n_blocks(blacs_grid__.num_ranks_row()),
                                         block_id(blacs_grid__.rank_row()), bs_row__)
                         .local_size(),
                 splindex_block_cyclic<>(num_cols__, n_blocks(blacs_grid__.num_ranks_col()),
                                         block_id(blacs_grid__.rank_col()), bs_col__)
                         .local_size()},
                ptr__)
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
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, memory_t mem_type__)
    : matrix<T>({num_rows__, num_cols__}, mem_type__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(1)
    , bs_col_(1)
    , spl_row_(num_rows_, n_blocks(1), block_id(0), bs_row_)
    , spl_col_(num_cols_, n_blocks(1), block_id(0), bs_col_)
{
}

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, memory_pool& mp__, std::string const& label__)
    : matrix<T>({num_rows__, num_cols__}, mp__, label__)
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
    : matrix<T>({num_rows__, num_cols__}, ptr__)
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
                    memory_pool& mp__)
    : matrix<T>({splindex_block_cyclic<>(num_rows__, n_blocks(blacs_grid__.num_ranks_row()),
                                         block_id(blacs_grid__.rank_row()), bs_row__)
                         .local_size(),
                 splindex_block_cyclic<>(num_cols__, n_blocks(blacs_grid__.num_ranks_col()),
                                         block_id(blacs_grid__.rank_col()), bs_col__)
                         .local_size()},
                mp__)
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
void
dmatrix<T>::set(int ir0__, int jc0__, int mr__, int nc__, T* ptr__, int ld__)
{
    splindex_block_cyclic<> spl_r0(ir0__, n_blocks(blacs_grid().num_ranks_row()), block_id(blacs_grid().rank_row()),
                                   bs_row_);
    splindex_block_cyclic<> spl_r1(ir0__ + mr__, n_blocks(blacs_grid().num_ranks_row()),
                                   block_id(blacs_grid().rank_row()), bs_row_);

    splindex_block_cyclic<> spl_c0(jc0__, n_blocks(blacs_grid().num_ranks_col()), block_id(blacs_grid().rank_col()),
                                   bs_col_);
    splindex_block_cyclic<> spl_c1(jc0__ + nc__, n_blocks(blacs_grid().num_ranks_col()),
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
void
dmatrix<T>::set(const int irow_glob, const int icol_glob, T val)
{
    if (blacs_grid_) {
        auto r = spl_row_.location(irow_glob);
        if (blacs_grid_->rank_row() == r.ib) {
            auto c = spl_col_.location(icol_glob);
            if (blacs_grid_->rank_col() == c.ib) {
                (*this)(r.index_local, c.index_local) = val;
            }
        }
    } else {
        (*this)(irow_glob, icol_glob) = val;
    }
}

template <typename T>
void
dmatrix<T>::add(const int irow_glob, const int icol_glob, T val)
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
void
dmatrix<T>::add(real_type<T> beta__, const int irow_glob, const int icol_glob, T val)
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
void
dmatrix<T>::make_real_diag(int n__)
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
mdarray<T, 1>
dmatrix<T>::get_diag(int n__)
{
    mdarray<T, 1> d({n__});
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
    blacs_grid_->comm().allreduce(d.template at(memory_t::host), n__);
    return d;
}

template <typename T>
void
dmatrix<T>::save_to_hdf5(std::string name__, int m__, int n__)
{
    mdarray<T, 2> full_mtrx({m__, n__});
    full_mtrx.zero();

    for (int j = 0; j < this->num_cols_local(); j++) {
        for (int i = 0; i < this->num_rows_local(); i++) {
            if (this->irow(i) < m__ && this->icol(j) < n__) {
                full_mtrx(this->irow(i), this->icol(j)) = (*this)(i, j);
            }
        }
    }
    this->comm().allreduce(full_mtrx.template at(memory_t::host), static_cast<int>(full_mtrx.size()));

    if (this->blacs_grid().comm().rank() == 0) {
        sirius::HDF5_tree h5(name__, sirius::hdf5_access_t::truncate);
        h5.write("nrow", m__);
        h5.write("ncol", n__);
        h5.write("mtrx", full_mtrx);
    }
}

// instantiate for required types
template class dmatrix<double>;
template class dmatrix<std::complex<double>>;
#ifdef SIRIUS_USE_FP32
template class dmatrix<float>;
template class dmatrix<std::complex<float>>;
#endif

} // namespace la

} // namespace sirius

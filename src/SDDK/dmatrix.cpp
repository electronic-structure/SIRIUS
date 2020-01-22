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

namespace sddk {

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__,
                    memory_t mem_type__)
    : matrix<T>(splindex<splindex_t::block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(),
                                                   bs_row__)
                    .local_size(),
                splindex<splindex_t::block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(),
                                                   bs_col__)
                    .local_size(),
                mem_type__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(bs_row__)
    , bs_col_(bs_col__)
    , blacs_grid_(&blacs_grid__)
    , spl_row_(num_rows_, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row_)
    , spl_col_(num_cols_, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col_)
{
    init();
}

template <typename T>
dmatrix<T>::dmatrix(int num_rows__, int num_cols__, memory_t mem_type__)
    : matrix<T>(num_rows__, num_cols__, mem_type__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(1)
    , bs_col_(1)
    , spl_row_(num_rows_, 1, 0, bs_row_)
    , spl_col_(num_cols_, 1, 0, bs_col_)
{
}

template <typename T>
dmatrix<T>::dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__,
                    int bs_col__)
    : matrix<T>(ptr__,
                splindex<splindex_t::block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(),
                                                   bs_row__)
                    .local_size(),
                splindex<splindex_t::block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(),
                                                   bs_col__)
                    .local_size())
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(bs_row__)
    , bs_col_(bs_col__)
    , blacs_grid_(&blacs_grid__)
    , spl_row_(num_rows_, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row_)
    , spl_col_(num_cols_, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col_)
{
    init();
}

template <typename T>
dmatrix<T>::dmatrix(memory_pool& mp__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__,
                    int bs_col__)
    : matrix<T>(splindex<splindex_t::block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(),
                                                   bs_row__)
                    .local_size(),
                splindex<splindex_t::block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(),
                                                   bs_col__)
                    .local_size(), mp__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(bs_row__)
    , bs_col_(bs_col__)
    , blacs_grid_(&blacs_grid__)
    , spl_row_(num_rows_, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row_)
    , spl_col_(num_cols_, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col_)
{
    init();
}

template <typename T>
dmatrix<T>::dmatrix(T* ptr__, int num_rows__, int num_cols__)
    : matrix<T>(ptr__, num_rows__, num_cols__)
    , num_rows_(num_rows__)
    , num_cols_(num_cols__)
    , bs_row_(1)
    , bs_col_(1)
    , spl_row_(num_rows_, 1, 0, bs_row_)
    , spl_col_(num_cols_, 1, 0, bs_col_)
{
    init();
}

template <typename T>
void dmatrix<T>::set(int ir0__, int jc0__, int mr__, int nc__, T* ptr__, int ld__)
{
    splindex<splindex_t::block_cyclic> spl_r0(ir0__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(), bs_row_);
    splindex<splindex_t::block_cyclic> spl_r1(ir0__ + mr__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(),
                                              bs_row_);

    splindex<splindex_t::block_cyclic> spl_c0(jc0__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(), bs_col_);
    splindex<splindex_t::block_cyclic> spl_c1(jc0__ + nc__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(),
                                              bs_col_);

    int m0 = spl_r0.local_size();
    int m1 = spl_r1.local_size();
    int n0 = spl_c0.local_size();
    int n1 = spl_c1.local_size();
    std::vector<int> map_row(m1 - m0);
    std::vector<int> map_col(n1 - n0);

    for (int i = 0; i < m1 - m0; i++) {
        map_row[i] = spl_r1[m0 + i] - ir0__;
    }
    for (int j = 0; j < n1 - n0; j++) {
        map_col[j] = spl_c1[n0 + j] - jc0__;
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
    if (blacs_grid_->rank_row() == r.rank) {
        auto c = spl_col_.location(icol_glob);
        if (blacs_grid_->rank_col() == c.rank) {
            (*this)(r.local_index, c.local_index) = val;
        }
    }
}

template <typename T>
void dmatrix<T>::add(const int irow_glob, const int icol_glob, T val)
{
    auto r = spl_row_.location(irow_glob);
    if (blacs_grid_->rank_row() == r.rank) {
        auto c = spl_col_.location(icol_glob);
        if (blacs_grid_->rank_col() == c.rank) {
            (*this)(r.local_index, c.local_index) += val;
        }
    }
}

template <typename T>
void dmatrix<T>::add(double beta__, const int irow_glob, const int icol_glob, T val)
{
    auto r = spl_row_.location(irow_glob);
    if (blacs_grid_->rank_row() == r.rank) {
        auto c = spl_col_.location(icol_glob);
        if (blacs_grid_->rank_col() == c.rank) {
            (*this)(r.local_index, c.local_index) = (*this)(r.local_index, c.local_index) * beta__ + val;
        }
    }
}

template <typename T>
void dmatrix<T>::make_real_diag(int n__)
{
    for (int i = 0; i < n__; i++) {
        auto r = spl_row_.location(i);
        if (blacs_grid_->rank_row() == r.rank) {
            auto c = spl_col_.location(i);
            if (blacs_grid_->rank_col() == c.rank) {
                T v                                   = (*this)(r.local_index, c.local_index);
                (*this)(r.local_index, c.local_index) = std::real(v);
            }
        }
    }
}

template <typename T>
mdarray<T, 1> dmatrix<T>::get_diag(int n__)
{
    mdarray<T, 1> d(n__);
    d.zero();

    for (int i = 0; i < n__; i++) {
        auto r = spl_row_.location(i);
        if (blacs_grid_->rank_row() == r.rank) {
            auto c = spl_col_.location(i);
            if (blacs_grid_->rank_col() == c.rank) {
                d[i] = (*this)(r.local_index, c.local_index);
            }
        }
    }
    blacs_grid_->comm().allreduce(d.template at(memory_t::host), n__);
    return d;
}

template <typename T>
void dmatrix<T>::save_to_hdf5(std::string name__, int m__, int n__)
{
    mdarray<T, 2> full_mtrx(m__, n__);
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
        HDF5_tree h5(name__, hdf5_access_t::truncate);
        h5.write("nrow", m__);
        h5.write("ncol", n__);
        h5.write("mtrx", full_mtrx);
    }
}

template <>
void dmatrix<double_complex>::serialize(std::string name__, int n__) const
{
    mdarray<double_complex, 2> full_mtrx(num_rows(), num_cols());
    full_mtrx.zero();

    for (int j = 0; j < num_cols_local(); j++) {
        for (int i = 0; i < num_rows_local(); i++) {
            full_mtrx(irow(i), icol(j)) = (*this)(i, j);
        }
    }
    blacs_grid_->comm().allreduce(full_mtrx.at(memory_t::host), static_cast<int>(full_mtrx.size()));

    // json dict;
    // dict["mtrx_re"] = json::array();
    // for (int i = 0; i < num_rows(); i++) {
    //    dict["mtrx_re"].push_back(json::array());
    //    for (int j = 0; j < num_cols(); j++) {
    //        dict["mtrx_re"][i].push_back(full_mtrx(i, j).real());
    //    }
    //}

    if (blacs_grid_->comm().rank() == 0) {
        // std::cout << "mtrx: " << name__ << std::endl;
        // std::cout << dict.dump(4);

        std::printf("matrix label: %s\n", name__.c_str());
        std::printf("{\n");
        for (int i = 0; i < n__; i++) {
            std::printf("{");
            for (int j = 0; j < n__; j++) {
                std::printf("%18.13f + I * %18.13f", full_mtrx(i, j).real(), full_mtrx(i, j).imag());
                if (j != n__ - 1) {
                    std::printf(",");
                }
            }
            if (i != n__ - 1) {
                std::printf("},\n");
            } else {
                std::printf("}\n");
            }
        }
        std::printf("}\n");
    }

    // std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
    // ofs << dict.dump(4);
}

template <>
void dmatrix<double>::serialize(std::string name__, int n__) const
{
    mdarray<double, 2> full_mtrx(num_rows(), num_cols());
    full_mtrx.zero();

    for (int j = 0; j < num_cols_local(); j++) {
        for (int i = 0; i < num_rows_local(); i++) {
            full_mtrx(irow(i), icol(j)) = (*this)(i, j);
        }
    }
    blacs_grid_->comm().allreduce(full_mtrx.at(memory_t::host), static_cast<int>(full_mtrx.size()));

    // json dict;
    // dict["mtrx"] = json::array();
    // for (int i = 0; i < num_rows(); i++) {
    //    dict["mtrx"].push_back(json::array());
    //    for (int j = 0; j < num_cols(); j++) {
    //        dict["mtrx"][i].push_back(full_mtrx(i, j));
    //    }
    //}

    // if (blacs_grid_->comm().rank() == 0) {
    //    std::cout << "mtrx: " << name__ << std::endl;
    //    std::cout << dict.dump(4);
    //}

    // std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
    // ofs << dict.dump(4);

    if (blacs_grid_->comm().rank() == 0) {
        std::printf("matrix label: %s\n", name__.c_str());
        std::printf("{\n");
        for (int i = 0; i < n__; i++) {
            std::printf("{");
            for (int j = 0; j < n__; j++) {
                std::printf("%18.13f", full_mtrx(i, j));
                if (j != n__ - 1) {
                    std::printf(",");
                }
            }
            if (i != n__ - 1) {
                std::printf("},\n");
            } else {
                std::printf("}\n");
            }
        }
        std::printf("}\n");
    }
}

// instantiate for required types
template class dmatrix<double>;
template class dmatrix<double_complex>;

} // namespace sddk

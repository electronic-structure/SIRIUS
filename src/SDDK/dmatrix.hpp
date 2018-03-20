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

/** \file dmatrix.hpp
 *
 *  \brief Contains definition and implementaiton of dmatrix class.
 */

#ifndef __DMATRIX_HPP__
#define __DMATRIX_HPP__

#include "blacs_grid.hpp"
#include "splindex.hpp"
#include "hdf5_tree.hpp"

namespace sddk {

/// Distributed matrix.
template <typename T>
class dmatrix : public matrix<T>
{
  private:
    /// Global number of matrix rows.
    int num_rows_{0};

    /// Global number of matrix columns.
    int num_cols_{0};

    /// Row block size.
    int bs_row_{0};

    /// Column block size.
    int bs_col_{0};

    /// BLACS grid.
    BLACS_grid const* blacs_grid_{nullptr};

    /// Split index of matrix rows.
    splindex<block_cyclic> spl_row_;

    /// Split index of matrix columns.
    splindex<block_cyclic> spl_col_;

    /// ScaLAPACK matrix descriptor.
    ftn_int descriptor_[9];

    void init()
    {
        #ifdef __SCALAPACK
        if (blacs_grid_ != nullptr) {
            linalg_base::descinit(descriptor_, num_rows_, num_cols_, bs_row_, bs_col_, 0, 0, blacs_grid_->context(),
                                  spl_row_.local_size());
        }
        #endif
    }

    /* forbid copy constructor */
    dmatrix(dmatrix<T> const& src) = delete;
    /* forbid assigment operator */
    dmatrix<T>& operator=(dmatrix<T> const& src) = delete;

  public:
    // Default constructor
    dmatrix()
    {
    }

    dmatrix(int num_rows__,
            int num_cols__,
            BLACS_grid const& blacs_grid__,
            int bs_row__,
            int bs_col__,
            memory_t mem_type__ = memory_t::host)
        : matrix<T>(splindex<block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row__).local_size(),
                    splindex<block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col__).local_size(),
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

    dmatrix(int num_rows__,
            int num_cols__,
            memory_t mem_type__ = memory_t::host)
        : matrix<T>(num_rows__, num_cols__, mem_type__)
        , num_rows_(num_rows__)
        , num_cols_(num_cols__)
        , bs_row_(1)
        , bs_col_(1)
        , spl_row_(num_rows_, 1, 0, bs_row_)
        , spl_col_(num_cols_, 1, 0, bs_col_)
    {
        init();
    }

    dmatrix(T* ptr__,
            int num_rows__,
            int num_cols__,
            BLACS_grid const& blacs_grid__,
            int bs_row__,
            int bs_col__)
        : matrix<T>(ptr__,
                    splindex<block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row__).local_size(),
                    splindex<block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col__).local_size())
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

    dmatrix(T* ptr__,
            int num_rows__,
            int num_cols__)
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

    dmatrix(dmatrix<T>&& src) = default;

    dmatrix<T>& operator=(dmatrix<T>&& src) = default;

    /// Return size of the square matrix or -1 in case of rectangular matrix.
    inline int size() const
    {
        if (num_rows_ == num_cols_) {
            return num_rows_;
        }
        return -1;
    }

    inline int num_rows() const
    {
        return num_rows_;
    }

    inline int num_rows_local() const
    {
        return spl_row_.local_size();
    }

    inline int num_rows_local(int rank) const
    {
        return spl_row_.local_size(rank);
    }

    inline int irow(int irow_loc) const
    {
        return spl_row_[irow_loc];
    }

    inline int num_cols() const
    {
        return num_cols_;
    }

    /// Local number of columns.
    inline int num_cols_local() const
    {
        return spl_col_.local_size();
    }

    inline int num_cols_local(int rank) const
    {
        return spl_col_.local_size(rank);
    }

    /// Inindex of column in global matrix.
    inline int icol(int icol_loc) const
    {
        return spl_col_[icol_loc];
    }

    inline int const* descriptor() const
    {
        return descriptor_;
    }

    //void zero(int ir0__, int ic0__, int nr__, int nc__)
    //{
    //    splindex<block_cyclic> spl_r0(ir0__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(), bs_row_);
    //    splindex<block_cyclic> spl_r1(ir0__ + nr__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(), bs_row_);

    //    splindex<block_cyclic> spl_c0(ic0__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(), bs_col_);
    //    splindex<block_cyclic> spl_c1(ic0__ + nc__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(), bs_col_);

    //    int m0 = spl_r0.local_size();
    //    int m1 = spl_r1.local_size();
    //    int n0 = spl_c0.local_size();
    //    int n1 = spl_c1.local_size();
    //    for (int j = n0; j < n1; j++) {
    //        std::fill(this->template at<CPU>(m0, j), this->template at<CPU>(m1, j), 0);
    //    }

    //    if (this->on_device()) {
    //        acc::zero(this->template at<GPU>(m0, n0), this->ld(), m1 - m0, n1 - n0);
    //    }
    //}

    inline void set(const int irow_glob, const int icol_glob, T val)
    {
        auto r = spl_row_.location(irow_glob);
        if (blacs_grid_->rank_row() == r.rank) {
            auto c = spl_col_.location(icol_glob);
            if (blacs_grid_->rank_col() == c.rank) {
                (*this)(r.local_index, c.local_index) = val;
            }
        }
    }

    inline void add(const int irow_glob, const int icol_glob, T val)
    {
        auto r = spl_row_.location(irow_glob);
        if (blacs_grid_->rank_row() == r.rank) {
            auto c = spl_col_.location(icol_glob);
            if (blacs_grid_->rank_col() == c.rank) {
                (*this)(r.local_index, c.local_index) += val;
            }
        }
    }

    inline void add(double beta__, const int irow_glob, const int icol_glob, T val)
    {
        auto r = spl_row_.location(irow_glob);
        if (blacs_grid_->rank_row() == r.rank) {
            auto c = spl_col_.location(icol_glob);
            if (blacs_grid_->rank_col() == c.rank) {
                (*this)(r.local_index, c.local_index) = (*this)(r.local_index, c.local_index) * beta__ + val;
            }
        }
    }

    inline void make_real_diag(int n__)
    {
        for (int i = 0; i < n__; i++) {
            auto r = spl_row_.location(i);
            if (blacs_grid_->rank_row() == r.rank) {
                auto c = spl_col_.location(i);
                if (blacs_grid_->rank_col() == c.rank) {
                    T v = (*this)(r.local_index, c.local_index);
                    (*this)(r.local_index, c.local_index) = std::real(v);
                }
            }
        }
    }

    inline mdarray<T, 1> get_diag(int n__)
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
        blacs_grid_->comm().allreduce(d.template at<CPU>(), n__);
        return std::move(d);
    }

    inline splindex<block_cyclic> const& spl_col() const
    {
        return spl_col_;
    }

    inline splindex<block_cyclic> const& spl_row() const
    {
        return spl_row_;
    }

    inline int rank_row() const
    {
        return blacs_grid_->rank_row();
    }

    inline int num_ranks_row() const
    {
        return blacs_grid_->num_ranks_row();
    }

    inline int rank_col() const
    {
        return blacs_grid_->rank_col();
    }

    inline int num_ranks_col() const
    {
        return blacs_grid_->num_ranks_col();
    }

    inline int bs_row() const
    {
        return bs_row_;
    }

    inline int bs_col() const
    {
        return bs_col_;
    }

    inline BLACS_grid const& blacs_grid() const
    {
        assert(blacs_grid_ != nullptr);
        return *blacs_grid_;
    }

    void save_to_hdf5(std::string name__, int m__, int n__)
    {
        mdarray<T, 2> full_mtrx(m__, n__);
        full_mtrx.zero();

        for (int j = 0; j < this->num_cols_local(); j++) {
            for (int i = 0; i < this->num_rows_local(); i++) {
                if (this->irow(i) < m__ &&  this->icol(j) < n__) {
                    full_mtrx(this->irow(i), this->icol(j)) = (*this)(i, j);
                }
            }
        }
        this->comm().allreduce(full_mtrx.template at<CPU>(), static_cast<int>(full_mtrx.size()));

        if (this->blacs_grid().comm().rank() == 0) {
            HDF5_tree h5(name__, true);
            h5.write("nrow", m__);
            h5.write("ncol", n__);
            h5.write("mtrx", full_mtrx);
        }
    }

    inline void serialize(std::string name__, int n__) const;

    inline Communicator const& comm() const {
        if (blacs_grid_ != nullptr) {
            return blacs_grid().comm();
        } else {
            return mpi_comm_self();
        }
    }
};

template <>
inline void dmatrix<double_complex>::serialize(std::string name__, int n__) const
{
    mdarray<double_complex, 2> full_mtrx(num_rows(), num_cols());
    full_mtrx.zero();

    for (int j = 0; j < num_cols_local(); j++) {
        for (int i = 0; i < num_rows_local(); i++) {
            full_mtrx(irow(i), icol(j)) = (*this)(i, j);
        }
    }
    blacs_grid_->comm().allreduce(full_mtrx.at<CPU>(), static_cast<int>(full_mtrx.size()));

    //json dict;
    //dict["mtrx_re"] = json::array();
    //for (int i = 0; i < num_rows(); i++) {
    //    dict["mtrx_re"].push_back(json::array());
    //    for (int j = 0; j < num_cols(); j++) {
    //        dict["mtrx_re"][i].push_back(full_mtrx(i, j).real());
    //    }
    //}

    if (blacs_grid_->comm().rank() == 0) {
        //std::cout << "mtrx: " << name__ << std::endl;
        // std::cout << dict.dump(4);

        printf("matrix label: %s\n", name__.c_str());
        printf("{\n");
        for (int i = 0; i < n__; i++) {
            printf("{");
            for (int j = 0; j < n__; j++) {
                printf("%18.13f + I * %18.13f", full_mtrx(i, j).real(), full_mtrx(i, j).imag());
                if (j != n__ - 1) {
                    printf(",");
                }
            }
            if (i != n__ - 1) {
                printf("},\n");
            } else {
                printf("}\n");
            }
        }
        printf("}\n");
    }

    // std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
    // ofs << dict.dump(4);
}

template <>
inline void dmatrix<double>::serialize(std::string name__, int n__) const
{
    mdarray<double, 2> full_mtrx(num_rows(), num_cols());
    full_mtrx.zero();

    for (int j = 0; j < num_cols_local(); j++) {
        for (int i = 0; i < num_rows_local(); i++) {
            full_mtrx(irow(i), icol(j)) = (*this)(i, j);
        }
    }
    blacs_grid_->comm().allreduce(full_mtrx.at<CPU>(), static_cast<int>(full_mtrx.size()));

    //json dict;
    //dict["mtrx"] = json::array();
    //for (int i = 0; i < num_rows(); i++) {
    //    dict["mtrx"].push_back(json::array());
    //    for (int j = 0; j < num_cols(); j++) {
    //        dict["mtrx"][i].push_back(full_mtrx(i, j));
    //    }
    //}

    //if (blacs_grid_->comm().rank() == 0) {
    //    std::cout << "mtrx: " << name__ << std::endl;
    //    std::cout << dict.dump(4);
    //}

    // std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
    // ofs << dict.dump(4);

    if (blacs_grid_->comm().rank() == 0) {
        printf("matrix label: %s\n", name__.c_str());
        printf("{\n");
        for (int i = 0; i < n__; i++) {
            printf("{");
            for (int j = 0; j < n__; j++) {
                printf("%18.13f", full_mtrx(i, j));
                if (j != n__ - 1) {
                    printf(",");
                }
            }
            if (i != n__ - 1) {
                printf("},\n");
            } else {
                printf("}\n");
            }
        }
        printf("}\n");
    }
}

} // namespace sddk

#endif // __DMATRIX_HPP__

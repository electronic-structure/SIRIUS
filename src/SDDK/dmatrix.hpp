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
 *  \brief Contains definition and implementation of sddk::dmatrix class.
 */

#ifndef __DMATRIX_HPP__
#define __DMATRIX_HPP__

#include <iomanip>
#include <spla/spla.hpp>
#include <costa/layout.hpp>
#include <costa/grid2grid/transformer.hpp>
#include "linalg/blacs_grid.hpp"
#include "splindex.hpp"
#include "hdf5_tree.hpp"
#include "type_definition.hpp"

namespace sddk {

namespace fmt {
template <typename T>
std::ostream& operator<<(std::ostream& out, std::complex<T> z)
{
    out << z.real() << " + I*" << z.imag();
    return out;
}
}

/// Distributed matrix.
template <typename T>
class dmatrix<T, matrix_distribution_t::block_cyclic> : public matrix<T>
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
    splindex<splindex_t::block_cyclic> spl_row_;

    /// Split index of matrix columns.
    splindex<splindex_t::block_cyclic> spl_col_;

    /// ScaLAPACK matrix descriptor.
    ftn_int descriptor_[9];

    /// Matrix distribution used for SPLA library functions
    spla::MatrixDistribution spla_dist_{spla::MatrixDistribution::create_mirror(MPI_COMM_SELF)};

    costa::grid_layout<T> grid_layout_;

    void init()
    {
#ifdef SIRIUS_SCALAPACK
        if (blacs_grid_ != nullptr) {
            linalg_base::descinit(descriptor_, num_rows_, num_cols_, bs_row_, bs_col_, 0, 0, blacs_grid_->context(),
                                  spl_row_.local_size());
        }
#endif
        grid_layout_ = costa::block_cyclic_layout<T>(this->num_rows(), this->num_cols(), this->bs_row(),
                this->bs_col(), 1, 1, this->num_rows(), this->num_cols(), this->blacs_grid().num_ranks_row(),
                this->blacs_grid().num_ranks_col(), 'R', 0, 0, this->at(memory_t::host), this->ld(), 'C',
                this->blacs_grid().comm().rank());
    }

    /* forbid copy constructor */
    dmatrix(dmatrix<T> const& src) = delete;
    /* forbid assignment operator */
    dmatrix<T>& operator=(dmatrix<T> const& src) = delete;

  public:
    // Default constructor
    dmatrix()
    {
    }

    dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__,
            memory_t mem_type__ = memory_t::host);

    dmatrix(int num_rows__, int num_cols__, memory_t mem_type__ = memory_t::host);

    dmatrix(int num_rows__, int num_cols__, memory_pool& mp__, std::string const& label__ = "");

    dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__);

    dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__,
            memory_pool& mp__);

    dmatrix(T* ptr__, int num_rows__, int num_cols__);

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

    inline int size_local() const
    {
      return this->num_rows_local() * this->num_cols_local();
    }

    /// Return number of rows in the global matrix.
    inline int num_rows() const
    {
        return num_rows_;
    }

    /// Return local number of rows for this MPI rank.
    inline int num_rows_local() const
    {
        return spl_row_.local_size();
    }

    /// Return local number of rows for a given MPI rank.
    inline int num_rows_local(int rank) const
    {
        return spl_row_.local_size(rank);
    }

    /// Return global row index in the range [0, num_rows) by the local index in the range [0, num_rows_local).
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

    inline spla::MatrixDistribution& spla_distribution()
    {
        return spla_dist_;
    }

    //void zero(int ir0__, int ic0__, int nr__, int nc__)
    //{
    //    splindex<splindex_t::block_cyclic> spl_r0(ir0__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(), bs_row_);
    //    splindex<splindex_t::block_cyclic> spl_r1(ir0__ + nr__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(), bs_row_);

    //    splindex<splindex_t::block_cyclic> spl_c0(ic0__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(), bs_col_);
    //    splindex<splindex_t::block_cyclic> spl_c1(ic0__ + nc__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(), bs_col_);

    //    int m0 = spl_r0.local_size();
    //    int m1 = spl_r1.local_size();
    //    int n0 = spl_c0.local_size();
    //    int n1 = spl_c1.local_size();
    //    for (int j = n0; j < n1; j++) {
    //        std::fill(this->template at<device_t::CPU>(m0, j), this->template at<CPU>(m1, j), 0);
    //    }

    //    if (this->on_device()) {
    //        acc::zero(this->template at<device_t::GPU>(m0, n0), this->ld(), m1 - m0, n1 - n0);
    //    }
    //}

    void set(int ir0__, int jc0__, int mr__, int nc__, T* ptr__, int ld__);

    void set(const int irow_glob, const int icol_glob, T val);

    void add(const int irow_glob, const int icol_glob, T val);

    void add(real_type<T> beta__, const int irow_glob, const int icol_glob, T val);

    void make_real_diag(int n__);

    sddk::mdarray<T, 1> get_diag(int n__);

    inline splindex<splindex_t::block_cyclic> const& spl_col() const
    {
        return spl_col_;
    }

    inline splindex<splindex_t::block_cyclic> const& spl_row() const
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

    void save_to_hdf5(std::string name__, int m__, int n__);

    sddk::mdarray<T, 2> get_full_matrix() const
    {
        mdarray<T, 2> full_mtrx(num_rows(), num_cols());
        full_mtrx.zero();

        for (int j = 0; j < num_cols_local(); j++) {
            for (int i = 0; i < num_rows_local(); i++) {
                full_mtrx(irow(i), icol(j)) = (*this)(i, j);
            }
        }
        if (blacs_grid_) {
            blacs_grid_->comm().allreduce(full_mtrx.at(memory_t::host), static_cast<int>(full_mtrx.size()));
        }
        return full_mtrx;
    }

    nlohmann::json serialize_to_json(int m__, int n__) const
    {
        auto full_mtrx = get_full_matrix();

        nlohmann::json dict;
        dict["mtrx_re"] = nlohmann::json::array();
        for (int i = 0; i < num_rows(); i++) {
            dict["mtrx_re"].push_back(nlohmann::json::array());
            for (int j = 0; j < num_cols(); j++) {
                dict["mtrx_re"][i].push_back(std::real(full_mtrx(i, j)));
            }
        }
        if (!std::is_scalar<T>::value) {
            dict["mtrx_im"] = nlohmann::json::array();
            for (int i = 0; i < num_rows(); i++) {
                dict["mtrx_im"].push_back(nlohmann::json::array());
                for (int j = 0; j < num_cols(); j++) {
                    dict["mtrx_im"][i].push_back(std::imag(full_mtrx(i, j)));
                }
            }
        }
        return dict;
        // std::ofstream ofs(aiida_output_file, std::ofstream::out | std::ofstream::trunc);
        // ofs << dict.dump(4);
    }

    std::stringstream serialize(std::string name__, int m__, int n__) const
    {
        auto full_mtrx = get_full_matrix();

        std::stringstream out;
        using namespace fmt;
        out << std::setprecision(12) << std::setw(24) << std::fixed;

        out << "matrix label : " << name__ << std::endl;
        out << "{" << std::endl;
        for (int i = 0; i < m__; i++) {
            out << "{";
            for (int j = 0; j < n__; j++) {
                out << full_mtrx(i, j);
                if (j != n__ - 1) {
                    out << ",";
                }
            }
            if (i != n__ - 1) {
                out << "}," << std::endl;
            } else {
                out << "}" << std::endl;
            }
        }
        out << "}";

        return out;
    }

    inline T checksum(int m__, int n__) const
    {
        T cs{0};

        if (blacs_grid_ != nullptr) {
            splindex<splindex_t::block_cyclic> spl_row(m__, this->blacs_grid().num_ranks_row(),
                                                       this->blacs_grid().rank_row(), this->bs_row());
            splindex<splindex_t::block_cyclic> spl_col(n__, this->blacs_grid().num_ranks_col(),
                                                       this->blacs_grid().rank_col(), this->bs_col());
            for (int i = 0; i < spl_col.local_size(); i++) {
                for (int j = 0; j < spl_row.local_size(); j++) {
                    cs += (*this)(j, i);
                }
            }
            this->blacs_grid().comm().allreduce(&cs, 1);
        } else {
            for (int i = 0; i < n__; i++) {
                for (int j = 0; j < m__; j++) {
                    cs += (*this)(j, i);
                }
            }
        }
        return cs;
    }

    inline Communicator const& comm() const
    {
        if (blacs_grid_ != nullptr) {
            return blacs_grid().comm();
        } else {
            return Communicator::self();
        }
    }

    costa::grid_layout<T>& grid_layout()
    {
        return grid_layout_;
    }

    costa::grid_layout<T> grid_layout(int irow0__, int jcol0__, int mrow__, int ncol__)
    {
        return costa::block_cyclic_layout<T>(this->num_rows(), this->num_cols(), this->bs_row(),
                this->bs_col(), irow0__ + 1, jcol0__ + 1, mrow__, ncol__, this->blacs_grid().num_ranks_row(),
                this->blacs_grid().num_ranks_col(), 'R', 0, 0, this->at(memory_t::host), this->ld(), 'C',
                this->blacs_grid().comm().rank());
    }

};

/// Distributed matrix.
template <typename T>
class dmatrix<T, matrix_distribution_t::slab> : public matrix<T>
{
  private:
    int num_rows_;
    int num_cols_;
    splindex<splindex_t::chunk, int> spl_row_;
    Communicator comm_;
    costa::grid_layout<T> grid_layout_;
  public:
    dmatrix(int num_rows__, int num_cols__, std::vector<int> counts__, Communicator const& comm__)
        : matrix<T>(counts__[comm__.rank()], num_cols__)
        , num_rows_(num_rows__)
        , num_cols_(num_cols__)
        , comm_(comm__)
    {
        spl_row_ = splindex<splindex_t::chunk, int>(num_rows__, comm_.size(), comm_.rank(), counts__);

        std::vector<int> rowsplit(comm_.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm_.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + spl_row_.local_size(i);
        }
        std::vector<int> colsplit({0, num_cols_});
        std::vector<int> owners(comm_.size());
        for (int i = 0; i < comm_.size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->at(memory_t::host);
        localblock.ld = this->ld();
        localblock.row = comm_.rank();
        localblock.col = 0;

        grid_layout_ = costa::custom_layout<T>(comm_.size(), 1, rowsplit.data(), colsplit.data(),
                owners.data(), 1, &localblock, 'C');
    }

    costa::grid_layout<T>& grid_layout()
    {
        return grid_layout_;
    }

};

} // namespace sddk

#endif // __DMATRIX_HPP__

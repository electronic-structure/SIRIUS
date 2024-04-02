/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file dmatrix.hpp
 *
 *  \brief Contains definition and implementation of distributed matrix class.
 */

#ifndef __DMATRIX_HPP__
#define __DMATRIX_HPP__

#include <iomanip>
#include <spla/spla.hpp>
#include <costa/layout.hpp>
#include <costa/grid2grid/transformer.hpp>
#include "core/la/blacs_grid.hpp"
#include "core/splindex.hpp"
#include "core/hdf5_tree.hpp"
#include "core/typedefs.hpp"
#include "core/rte/rte.hpp"
#include "core/json.hpp"
#include "core/memory.hpp"

namespace sirius {

namespace la {

namespace fmt {
template <typename T>
std::ostream&
operator<<(std::ostream& out, std::complex<T> z)
{
    out << z.real() << " + I*" << z.imag();
    return out;
}
} // namespace fmt

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
    sirius::splindex_block_cyclic<> spl_row_;

    /// Split index of matrix columns.
    sirius::splindex_block_cyclic<> spl_col_;

    /// ScaLAPACK matrix descriptor.
    ftn_int descriptor_[9];

    /// Matrix distribution used for SPLA library functions
    spla::MatrixDistribution spla_dist_{spla::MatrixDistribution::create_mirror(MPI_COMM_SELF)};

    costa::grid_layout<T> grid_layout_;

    void
    init()
    {
        if (blacs_grid_ != nullptr) {
#ifdef SIRIUS_SCALAPACK
            linalg_base::descinit(descriptor_, num_rows_, num_cols_, bs_row_, bs_col_, 0, 0, blacs_grid_->context(),
                                  spl_row_.local_size());
#endif
            grid_layout_ = costa::block_cyclic_layout<T>(
                    this->num_rows(), this->num_cols(), this->bs_row(), this->bs_col(), 1, 1, this->num_rows(),
                    this->num_cols(), this->blacs_grid().num_ranks_row(), this->blacs_grid().num_ranks_col(), 'R', 0, 0,
                    this->at(memory_t::host), this->ld(), 'C', this->blacs_grid().comm().rank());
        }
    }

    /* forbid copy constructor */
    dmatrix(dmatrix<T> const& src) = delete;
    /* forbid assignment operator */
    dmatrix<T>&
    operator=(dmatrix<T> const& src) = delete;

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

    dmatrix<T>&
    operator=(dmatrix<T>&& src) = default;

    /// Return size of the square matrix or -1 in case of rectangular matrix.
    inline int
    size() const
    {
        if (num_rows_ == num_cols_) {
            return num_rows_;
        }
        return -1;
    }

    inline int
    size_local() const
    {
        return this->num_rows_local() * this->num_cols_local();
    }

    /// Return number of rows in the global matrix.
    inline int
    num_rows() const
    {
        return num_rows_;
    }

    /// Return local number of rows for this MPI rank.
    inline int
    num_rows_local() const
    {
        return spl_row_.local_size();
    }

    /// Return local number of rows for a given MPI rank.
    inline int
    num_rows_local(int rank) const
    {
        return spl_row_.local_size(block_id(rank));
    }

    /// Return global row index in the range [0, num_rows) by the local index in the range [0, num_rows_local).
    inline int
    irow(int irow_loc) const
    {
        return spl_row_.global_index(irow_loc);
    }

    inline int
    num_cols() const
    {
        return num_cols_;
    }

    /// Local number of columns.
    inline int
    num_cols_local() const
    {
        return spl_col_.local_size();
    }

    inline int
    num_cols_local(int rank) const
    {
        return spl_col_.local_size(block_id(rank));
    }

    /// Inindex of column in global matrix.
    inline int
    icol(int icol_loc) const
    {
        return spl_col_.global_index(icol_loc);
    }

    inline int const*
    descriptor() const
    {
        return descriptor_;
    }

    inline spla::MatrixDistribution&
    spla_distribution()
    {
        return spla_dist_;
    }

    // void zero(int ir0__, int ic0__, int nr__, int nc__)
    //{
    //     splindex<splindex_t::block_cyclic> spl_r0(ir0__, blacs_grid().num_ranks_row(), blacs_grid().rank_row(),
    //     bs_row_); splindex<splindex_t::block_cyclic> spl_r1(ir0__ + nr__, blacs_grid().num_ranks_row(),
    //     blacs_grid().rank_row(), bs_row_);

    //    splindex<splindex_t::block_cyclic> spl_c0(ic0__, blacs_grid().num_ranks_col(), blacs_grid().rank_col(),
    //    bs_col_); splindex<splindex_t::block_cyclic> spl_c1(ic0__ + nc__, blacs_grid().num_ranks_col(),
    //    blacs_grid().rank_col(), bs_col_);

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

    using matrix<T>::copy_to;

    void
    copy_to(memory_t mem__, int ir0__, int ic0__, int nr__, int nc__)
    {
        int m0, m1, n0, n1;
        if (blacs_grid_ != nullptr) {
            splindex_block_cyclic<> spl_r0(ir0__, n_blocks(blacs_grid().num_ranks_row()),
                                           block_id(blacs_grid().rank_row()), bs_row_);
            splindex_block_cyclic<> spl_r1(ir0__ + nr__, n_blocks(blacs_grid().num_ranks_row()),
                                           block_id(blacs_grid().rank_row()), bs_row_);

            splindex_block_cyclic<> spl_c0(ic0__, n_blocks(blacs_grid().num_ranks_col()),
                                           block_id(blacs_grid().rank_col()), bs_col_);
            splindex_block_cyclic<> spl_c1(ic0__ + nc__, n_blocks(blacs_grid().num_ranks_col()),
                                           block_id(blacs_grid().rank_col()), bs_col_);

            m0 = spl_r0.local_size();
            m1 = spl_r1.local_size();
            n0 = spl_c0.local_size();
            n1 = spl_c1.local_size();
        } else {
            m0 = ir0__;
            m1 = ir0__ + nr__;
            n0 = ic0__;
            n1 = ic0__ + nc__;
        }

        if (is_host_memory(mem__)) {
            acc::copyout(this->at(memory_t::host, m0, n0), this->ld(), this->at(memory_t::device, m0, n0), this->ld(),
                         m1 - m0, n1 - n0);
        }
        if (is_device_memory(mem__)) {
            acc::copyin(this->at(memory_t::device, m0, n0), this->ld(), this->at(memory_t::host, m0, n0), this->ld(),
                        m1 - m0, n1 - n0);
        }
    }

    void
    set(int ir0__, int jc0__, int mr__, int nc__, T* ptr__, int ld__);

    void
    set(const int irow_glob, const int icol_glob, T val);

    void
    add(const int irow_glob, const int icol_glob, T val);

    void
    add(real_type<T> beta__, const int irow_glob, const int icol_glob, T val);

    void
    make_real_diag(int n__);

    mdarray<T, 1>
    get_diag(int n__);

    inline auto const&
    spl_col() const
    {
        return spl_col_;
    }

    inline auto const&
    spl_row() const
    {
        return spl_row_;
    }

    inline int
    rank_row() const
    {
        return blacs_grid_->rank_row();
    }

    inline int
    num_ranks_row() const
    {
        return blacs_grid_->num_ranks_row();
    }

    inline int
    rank_col() const
    {
        return blacs_grid_->rank_col();
    }

    inline int
    num_ranks_col() const
    {
        return blacs_grid_->num_ranks_col();
    }

    /// Row blocking factor
    inline int
    bs_row() const
    {
        return bs_row_;
    }

    /// Column blocking factor
    inline int
    bs_col() const
    {
        return bs_col_;
    }

    inline auto const&
    blacs_grid() const
    {
        RTE_ASSERT(blacs_grid_ != nullptr);
        return *blacs_grid_;
    }

    void
    save_to_hdf5(std::string name__, int m__, int n__);

    auto
    get_full_matrix() const
    {
        mdarray<T, 2> full_mtrx({num_rows(), num_cols()});
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

    nlohmann::json
    serialize_to_json(int m__, int n__) const
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
    }

    std::stringstream
    serialize(std::string name__, int m__, int n__) const
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

    inline T
    checksum(int m__, int n__) const
    {
        T cs{0};

        if (blacs_grid_ != nullptr) {
            splindex_block_cyclic<> spl_row(m__, n_blocks(this->blacs_grid().num_ranks_row()),
                                            block_id(this->blacs_grid().rank_row()), this->bs_row());
            splindex_block_cyclic<> spl_col(n__, n_blocks(this->blacs_grid().num_ranks_col()),
                                            block_id(this->blacs_grid().rank_col()), this->bs_col());
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

    inline auto const&
    comm() const
    {
        if (blacs_grid_ != nullptr) {
            return blacs_grid().comm();
        } else {
            return mpi::Communicator::self();
        }
    }

    costa::grid_layout<T>&
    grid_layout()
    {
        return grid_layout_;
    }

    costa::grid_layout<T>
    grid_layout(int irow0__, int jcol0__, int mrow__, int ncol__)
    {
        return costa::block_cyclic_layout<T>(
                this->num_rows(), this->num_cols(), this->bs_row(), this->bs_col(), irow0__ + 1, jcol0__ + 1, mrow__,
                ncol__, this->blacs_grid().num_ranks_row(), this->blacs_grid().num_ranks_col(), 'R', 0, 0,
                this->at(memory_t::host), this->ld(), 'C', this->blacs_grid().comm().rank());
    }
};

} // namespace la

} // namespace sirius

#endif // __DMATRIX_HPP__

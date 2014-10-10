// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file dmatrix.h
 *
 *  \brief Contains definition and implementaiton of dmatrix class.
 */

#ifndef __DMATRIX_H__
#define __DMATRIX_H__

#include "splindex.h"
#include "mdarray.h"
#include "communicator.h"
#include "blacs_grid.h"
#include "timer.h"

const int _panel_to_slice_ = 0;
const int _slice_to_panel_ = 1;

/// Distribued matrix.
template <typename T>
class dmatrix
{
    private:

        /// Global number of matrix rows.
        int num_rows_;

        /// Global number of matrix columns.
        int num_cols_;

        int num_ranks_row_;

        int rank_row_;

        int num_ranks_col_;

        int rank_col_;

        int bs_;

        BLACS_grid const* blacs_grid_;

        splindex<block_cyclic> spl_row_;

        splindex<block_cyclic> spl_col_;

        /// Local part of the distributed matrix.
        matrix<T> matrix_local_;

        matrix<T> matrix_slice_;

        mdarray<T, 1> ata_buffer_;

        /// Matrix descriptor.
        ftn_int descriptor_[9];

        void init()
        {
            #ifdef _SCALAPACK_
            bs_ = lin_alg<scalapack>::cyclic_block_size();
            #endif

            spl_row_ = splindex<block_cyclic>(num_rows_, num_ranks_row_, rank_row_, bs_);
            spl_col_ = splindex<block_cyclic>(num_cols_, num_ranks_col_, rank_col_, bs_);

            matrix_local_ = matrix<T>(nullptr, spl_row_.local_size(), spl_col_.local_size());

            #ifdef _SCALAPACK_
            lin_alg<scalapack>::descinit(descriptor_, num_rows_, num_cols_, bs_, bs_, 0, 0, blacs_grid_->context(), matrix_local_.ld());
            #endif
        }

    public:
        
        // Default constructor
        dmatrix()
            : num_rows_(0),
              num_cols_(0),
              num_ranks_row_(1), 
              rank_row_(0), 
              num_ranks_col_(1), 
              rank_col_(0),
              bs_(1),
              blacs_grid_(nullptr)
        {
        }
        
        dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__) 
            : num_rows_(num_rows__),
              num_cols_(num_cols__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()), 
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              bs_(1),
              blacs_grid_(&blacs_grid__)
        {
            init();
            matrix_local_.allocate();
        }

        dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__) 
            : num_rows_(num_rows__),
              num_cols_(num_cols__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()), 
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              bs_(1),
              blacs_grid_(&blacs_grid__)
        {
            init();
            matrix_local_.set_ptr(ptr__);
        }

        // forbid copy constructor
        dmatrix(dmatrix<T> const& src) = delete;
        // forbid move constructor
        dmatrix(dmatrix<T>&& src) = delete;
        // forbid assigment operator
        dmatrix<T>& operator=(dmatrix<T> const& src) = delete;

        inline dmatrix<T>& operator=(dmatrix<T>&& src)
        {
            if (this != &src)
            {
                num_rows_      = src.num_rows_;
                num_cols_      = src.num_cols_;
                num_ranks_row_ = src.num_ranks_row_;
                rank_row_      = src.rank_row_;
                num_ranks_col_ = src.num_ranks_col_;
                rank_col_      = src.rank_col_;
                bs_            = src.bs_;
                blacs_grid_    = src.blacs_grid_;
                spl_row_       = src.spl_row_;
                spl_col_       = src.spl_col_;
                matrix_local_  = std::move(src.matrix_local_);
                for (int i = 0; i < 9; i++) descriptor_[i] = src.descriptor_[i];
            }
            return *this;
        }

        inline void allocate(int mode__ = 0)
        {
            matrix_local_.allocate(mode__);
        }

        inline void deallocate()
        {
            matrix_local_.deallocate();
        }

        inline void allocate_ata_buffer(int max_num_cols_local)
        {
            int n = max_num_cols_local / num_ranks_row_ + std::min(1, max_num_cols_local % num_ranks_row_);
            ata_buffer_ = mdarray<T, 1>(num_rows() * n);
            matrix_slice_ = matrix<T>(num_rows(), n);
        }

        inline int num_rows() const
        {
            return num_rows_;
        }

        inline int num_rows_local() const
        {
            return static_cast<int>(spl_row_.local_size());
        }

        inline int num_rows_local(int rank) const
        {
            return static_cast<int>(spl_row_.local_size(rank));
        }

        inline int irow(int irow_loc) const
        {
            return static_cast<int>(spl_row_[irow_loc]);
        }

        inline int num_cols() const
        {
            return num_cols_;
        }
        
        /// Local number of columns.
        inline int num_cols_local() const
        {
            return static_cast<int>(spl_col_.local_size());
        }
        
        /// Inindex of column in global matrix.
        inline int icol(int icol_loc) const
        {
            return static_cast<int>(spl_col_[icol_loc]);
        }

        inline int* descriptor()
        {
            return descriptor_;
        }

        T* ptr()
        {
            return matrix_local_.ptr();
        }

        int ld() const
        {
            return matrix_local_.ld();
        }

        #ifdef _GPU_
        inline void allocate_on_device()
        {
            matrix_local_.allocate_on_device();
        }

        inline void deallocate_on_device()
        {
            matrix_local_.deallocate_on_device();
        }

        inline void pin_memory()
        {
            matrix_local_.pin_memory();
        }

        inline void unpin_memory()
        {
            matrix_local_.unpin_memory();
        }

        inline void zero_on_device()
        {
            matrix_local_.zero_on_device();
        }

        inline void copy_cols_to_device(int icol_fisrt, int icol_last)
        {
            splindex<block_cyclic> s0(icol_fisrt, num_ranks_col_, rank_col_, bs_);
            splindex<block_cyclic> s1(icol_last,  num_ranks_col_, rank_col_, bs_);
            int nloc = static_cast<int>(s1.local_size() - s0.local_size());
            if (nloc)
            {
                cuda_copy_to_device(at<GPU>(0, s0.local_size()), at<CPU>(0, s0.local_size()),
                                    num_rows_local() * nloc * sizeof(double_complex));
            }
        }

        inline void copy_cols_to_host(int icol_fisrt, int icol_last)
        {
            splindex<block_cyclic> s0(icol_fisrt, num_ranks_col_, rank_col_, bs_);
            splindex<block_cyclic> s1(icol_last,  num_ranks_col_, rank_col_, bs_);
            int nloc = static_cast<int>(s1.local_size() - s0.local_size());
            if (nloc)
            {
                cuda_copy_to_host(at<CPU>(0, s0.local_size()), at<GPU>(0, s0.local_size()),
                                    num_rows_local() * nloc * sizeof(double_complex));
            }
        }
        #endif

        inline T& operator()(const int64_t irow_loc, const int64_t icol_loc) 
        {
            return matrix_local_(irow_loc, icol_loc);
        }
        
        template <processing_unit_t pu>
        inline T* at() 
        {
            return matrix_local_.at<pu>();
        }

        template <processing_unit_t pu>
        inline T* at(int64_t const irow_loc, int64_t const icol_loc) 
        {
            return matrix_local_.at<pu>(irow_loc, icol_loc);
        }

        inline void set(const int irow_glob, const int icol_glob, T val)
        {
            auto r = spl_row_.location(irow_glob);
            if (rank_row_ == r.second)
            {
                auto c = spl_col_.location(icol_glob);
                if (rank_col_ == c.second)
                {
                    matrix_local_(r.first, c.first) = val;
                }
            }
        }

        inline void add(const int irow_glob, const int icol_glob, T val)
        {
            auto r = spl_row_.location(irow_glob);
            if (rank_row_ == r.second)
            {
                auto c = spl_col_.location(icol_glob);
                if (rank_col_ == c.second)
                {
                    matrix_local_(r.first, c.first) += val;
                }
            }
        }

        void zero()
        {
            matrix_local_.zero();
        }

        inline matrix<T>& panel()
        {
            return matrix_local_;
        }

        inline matrix<T>& slice()
        {
            return matrix_slice_;
        }

        //== template<int direction__>
        //== void shuffle_vertical(int offset__, int size__, mdarray<T, 2>& matrix_slice__)
        //== {
        //==     sirius::Timer t("dmatrix::shuffle");

        //==     /* trivial case */
        //==     if (num_ranks_row_ * num_ranks_col_ == 1 && offset__ == 0 && size__ == num_cols_)
        //==     {
        //==         if (direction__ == _slice_to_panel_) matrix_slice__ >> matrix_local_;
        //==         if (direction__ == _panel_to_slice_) matrix_local_ >> matrix_slice__;
        //==         return;
        //==     }

        //==     /* get local size of slice */
        //==     splindex<block_cyclic> s0(offset__,          num_ranks_col_, rank_col_, bs_);
        //==     splindex<block_cyclic> s1(offset__ + size__, num_ranks_col_, rank_col_, bs_);

        //==     int nloc = static_cast<int>(s1.local_size() - s0.local_size());
        //==     splindex<block> sub_spl_col(nloc, num_ranks_row_, rank_row_);

        //==     assert(num_rows() == (int)matrix_slice__.size(0));
        //==     assert(sub_spl_col.local_size() == (int)matrix_slice__.size(1));

        //==     std::vector<int> offsets(num_ranks_row_);
        //==     std::vector<int> counts(num_ranks_row_);

        //==     T* ptr = (nloc == 0) ? nullptr : &matrix_local_(0, s0.local_size());

        //==     for (int rank = 0; rank < num_ranks_row_; rank++)
        //==     {
        //==         /* each rank allocates a subpanel */
        //==         mdarray<double_complex, 2> sub_panel(spl_row_.local_size(rank), sub_spl_col.local_size());

        //==         /* make a table of sizes and offsets */
        //==         for (int i = 0; i < num_ranks_row_; i++)
        //==         {
        //==             /* offset of a sub-panel */
        //==             offsets[i] = static_cast<int>(spl_row_.local_size(rank) * sub_spl_col.global_offset(i));
        //==             /* size of a sub-panel */
        //==             counts[i] = static_cast<int>(spl_row_.local_size(rank) * sub_spl_col.local_size(i));
        //==         }
        //==         
        //==         if (direction__ == _slice_to_panel_)
        //==         {
        //==             /* fill the sub-panel */
        //==             for (int i = 0; i < sub_spl_col.local_size(); i++)
        //==             {
        //==                 /* loop over local fraction of rows */
        //==                 for (int j = 0; j < spl_row_.local_size(rank); j++)
        //==                 {
        //==                     sub_panel(j, i) = matrix_slice__(spl_row_.global_index(j, rank), i);
        //==                 }
        //==             }

        //==             /* gather local panel */
        //==             blacs_grid_->comm_row().gather(sub_panel.ptr(), ptr, &counts[0], &offsets[0], rank);
        //==         }

        //==         if (direction__ == _panel_to_slice_)
        //==         {
        //==             /* scatter local matrix between ranks */
        //==             blacs_grid_->comm_row().scatter(ptr, sub_panel.ptr(), &counts[0], &offsets[0], rank);
        //==             
        //==             /* loop over local fraction of columns */
        //==             for (int i = 0; i < sub_spl_col.local_size(); i++)
        //==             {
        //==                 /* loop over local fraction of rows */
        //==                 for (int j = 0; j < spl_row_.local_size(rank); j++)
        //==                 {
        //==                     /* copy necessary parts of panel to the full vector */
        //==                     matrix_slice__(spl_row_.global_index(j, rank), i) = sub_panel(j, i);
        //==                 }
        //==             }
        //==         }
        //==     }
        //== }

        template<int direction__>
        void shuffle_vertical_ata(int offset__, int size__, mdarray<T, 2>& matrix_slice__)
        {
            sirius::Timer t("dmatrix::shuffle_ata");

            /* trivial case */
            if (num_ranks_row_ * num_ranks_col_ == 1 && offset__ == 0 && size__ == num_cols_)
            {
                if (direction__ == _slice_to_panel_) matrix_slice__ >> matrix_local_;
                if (direction__ == _panel_to_slice_) matrix_local_ >> matrix_slice__;
                return;
            }

            /* get local size of slice */
            splindex<block_cyclic> s0(offset__,          num_ranks_col_, rank_col_, bs_);
            splindex<block_cyclic> s1(offset__ + size__, num_ranks_col_, rank_col_, bs_);

            int nloc = static_cast<int>(s1.local_size() - s0.local_size());
            splindex<block> sub_spl_col(nloc, num_ranks_row_, rank_row_);

            assert(num_rows() == (int)matrix_slice__.size(0));
            assert(sub_spl_col.local_size() <= (int)matrix_slice__.size(1));

            std::vector<int> sendcounts(num_ranks_row_);
            std::vector<int> recvcounts(num_ranks_row_);
            std::vector<int> sdispls(num_ranks_row_);
            std::vector<int> rdispls(num_ranks_row_);

            T* ptr = (nloc == 0) ? nullptr : &matrix_local_(0, s0.local_size());

            mdarray<T, 1> tmp;
            if (ata_buffer_.size() == 0)
            {
                tmp = mdarray<T, 1>(num_rows() * sub_spl_col.local_size());
            }
            else
            {
                tmp = mdarray<T, 1>(&ata_buffer_(0), num_rows() * sub_spl_col.local_size());
            }

            if (direction__ == _panel_to_slice_)
            {
                rdispls[0] = 0;
                for (int rank = 0; rank < num_ranks_row_; rank++)
                {
                    /* size of the sub-panel that is sent to each rank */
                    sendcounts[rank] = (int)sub_spl_col.local_size(rank) * num_rows_local();
                    /* offset in the send buffer */
                    sdispls[rank] = (int)sub_spl_col.global_offset(rank) * num_rows_local();
    
                    /* size of each recieved sub-panel */
                    recvcounts[rank] = (int)sub_spl_col.local_size() * num_rows_local(rank);
                    /* offset in the recieved buffer */
                    if (rank) rdispls[rank] = rdispls[rank - 1] + recvcounts[rank - 1];
                }
    
                sirius::Timer t1("dmatrix::shuffle_ata|comm");
                blacs_grid_->comm_row().alltoall(ptr, &sendcounts[0], &sdispls[0], &tmp(0), &recvcounts[0], &rdispls[0]);
                t1.stop();
            
                #pragma omp parallel for
                for (int rank = 0; rank < num_ranks_row_; rank++)
                {
                     mdarray<double_complex, 2> sub_panel(&tmp(rdispls[rank]), num_rows_local(rank), sub_spl_col.local_size());
    
                     /* loop over local fraction of columns */
                     for (int i = 0; i < sub_spl_col.local_size(); i++)
                     {
                         /* loop over local fraction of rows */
                         for (int j = 0; j < num_rows_local(rank); j++)
                         {
                             /* copy necessary parts of panel to the full vector */
                             matrix_slice__(spl_row_.global_index(j, rank), i) = sub_panel(j, i);
                         }
                     }
                }
            }

            if (direction__ ==_slice_to_panel_)
            {
                sdispls[0] = 0;
                for (int rank = 0; rank < num_ranks_row_; rank++)
                {
                    sendcounts[rank] = (int)sub_spl_col.local_size() * num_rows_local(rank);
                    if (rank) sdispls[rank] = sdispls[rank - 1] + sendcounts[rank - 1];
                    
                    recvcounts[rank] = (int)sub_spl_col.local_size(rank) * num_rows_local();
                    rdispls[rank] = (int)sub_spl_col.global_offset(rank) * num_rows_local();
                }

                #pragma omp parallel for
                for (int rank = 0; rank < num_ranks_row_; rank++)
                {
                     mdarray<double_complex, 2> sub_panel(&tmp(sdispls[rank]), num_rows_local(rank), sub_spl_col.local_size());

                    /* fill the sub-panel */
                    for (int i = 0; i < sub_spl_col.local_size(); i++)
                    {
                        /* loop over local fraction of rows */
                        for (int j = 0; j < spl_row_.local_size(rank); j++)
                        {
                            sub_panel(j, i) = matrix_slice__(spl_row_.global_index(j, rank), i);
                        }
                    }
                }

                sirius::Timer t1("dmatrix::shuffle_ata|comm");
                blacs_grid_->comm_row().alltoall(&tmp(0), &sendcounts[0], &sdispls[0], ptr, &recvcounts[0], &rdispls[0]);
                t1.stop();
            }
        }

        //== template<int direction__>
        //== void shuffle_horizontal(int N__, mdarray<T, 2>& matrix_slice__)
        //== {
        //==     // TODO: fix comments

        //==     sirius::Timer t("dmatrix::shuffle_horizontal");

        //==     ///* trivial case */
        //==     //if (num_ranks_row_ * num_ranks_col_ == 1 && offset__ == 0 && size__ == num_cols_)
        //==     //{
        //==     //    if (direction__ == _slice_to_panel_) matrix_slice__ >> matrix_local_;
        //==     //    if (direction__ == _panel_to_slice_) matrix_local_ >> matrix_slice__;
        //==     //    return;
        //==     //}

        //==     splindex<block_cyclic> spl_N(N__, num_ranks_col_, rank_col_, bs_);

        //==     mdarray<T, 2> tmp(spl_N.local_size(), spl_row_.local_size());

        //==     /* transpose local matrix */
        //==     if (direction__ == _panel_to_slice_)
        //==     {
        //==         for (int i = 0; i < spl_row_.local_size(); i++)
        //==         {
        //==             for (int j = 0; j < spl_N.local_size(); j++) tmp(j, i) = matrix_local_(i, j);
        //==         }
        //==     }

        //==     T* ptr = (tmp.size() == 0) ? nullptr : &tmp(0, 0);

        //==     splindex<block> sub_spl_row(spl_row_.local_size(), num_ranks_col_, rank_col_);

        //==     assert(N__ == (int)matrix_slice__.size(0));
        //==     assert(sub_spl_row.local_size() == (int)matrix_slice__.size(1));
        //==     
        //==     #pragma omp parallel num_threads(2)
        //==     {
        //==         std::vector<int> offsets(num_ranks_col_);
        //==         std::vector<int> counts(num_ranks_col_);

        //==         std::vector<double_complex> sub_panel_tmp(spl_N.local_size(0) * sub_spl_row.local_size());
        //==         
        //==         #pragma omp for
        //==         for (int rank = 0; rank < num_ranks_col_; rank++)
        //==         {
        //==             /* each rank allocates a subpanel */
        //==             //mdarray<double_complex, 2> sub_panel(spl_N.local_size(rank), sub_spl_row.local_size());
        //==             mdarray<double_complex, 2> sub_panel(&sub_panel_tmp[0], spl_N.local_size(rank), sub_spl_row.local_size());

        //==             /* make a table of sizes and offsets */
        //==             for (int i = 0; i < num_ranks_col_; i++)
        //==             {
        //==                 /* offset of a sub-panel */
        //==                 offsets[i] = static_cast<int>(spl_N.local_size(rank) * sub_spl_row.global_offset(i));
        //==                 /* size of a sub-panel */
        //==                 counts[i] = static_cast<int>(spl_N.local_size(rank) * sub_spl_row.local_size(i));
        //==             }
        //==             
        //==             if (direction__ == _slice_to_panel_)
        //==             {
        //==                 /* fill the sub-panel */
        //==                 for (int i = 0; i < sub_spl_row.local_size(); i++)
        //==                 {
        //==                     /* loop over local fraction of rows */
        //==                     for (int j = 0; j < spl_N.local_size(rank); j++)
        //==                     {
        //==                         sub_panel(j, i) = matrix_slice__(spl_N.global_index(j, rank), i);
        //==                     }
        //==                 }

        //==                 /* gather local matrix */
        //==                 blacs_grid_->comm_col().gather(sub_panel.ptr(), ptr, &counts[0], &offsets[0], rank);
        //==             }

        //==             if (direction__ == _panel_to_slice_)
        //==             {
        //==                 /* scatter local matrix between ranks */
        //==                 blacs_grid_->comm_col().scatter(ptr, sub_panel.ptr(), &counts[0], &offsets[0], rank);
        //==                 
        //==                 /* loop over local fraction of columns */
        //==                 for (int i = 0; i < sub_spl_row.local_size(); i++)
        //==                 {
        //==                     /* loop over local fraction of rows */
        //==                     for (int j = 0; j < spl_N.local_size(rank); j++)
        //==                     {
        //==                         /* copy necessary parts of panel to the full vector */
        //==                         matrix_slice__(spl_N.global_index(j, rank), i) = sub_panel(j, i);
        //==                     }
        //==                 }
        //==             }
        //==         }
        //==     }

        //==     /* transpose back to local matrix */
        //==     if (direction__ == _slice_to_panel_)
        //==     {
        //==         for (int i = 0; i < spl_row_.local_size(); i++)
        //==         {
        //==             for (int j = 0; j < spl_N.local_size(); j++) matrix_local_(i, j) = tmp(j, i);
        //==         }
        //==     }
        //== }

        /// Gather full vectors from the panels
        /** 
         * Communication happens between rows of the MPI grid 
         */
        void gather(mdarray<T, 2>& full_vectors__)
        {
            shuffle_vertical_ata<_panel_to_slice_>(0, num_cols_, full_vectors__);
        }

        void gather(int n__, int offs__, mdarray<T, 2>& matrix_slice__)
        {
            shuffle_vertical_ata<_panel_to_slice_>(offs__, n__, matrix_slice__);
        }

        void gather(int n__, int offs__)
        {
            shuffle_vertical_ata<_panel_to_slice_>(offs__, n__, matrix_slice_);
        }

        /// Scatter full vectors to the panels
        /** 
         * Communication happens between rows of the MPI grid 
         */
        void scatter(mdarray<double_complex, 2>& full_vectors__)
        {
            shuffle_vertical_ata<_slice_to_panel_>(0, num_cols_, full_vectors__);
        }

        void scatter(int n__, int offs__, mdarray<T, 2>& matrix_slice__)
        {
            shuffle_vertical_ata<_slice_to_panel_>(offs__, n__, matrix_slice__);
        }

        void scatter(int n__, int offs__)
        {
            shuffle_vertical_ata<_slice_to_panel_>(offs__, n__, matrix_slice_);
        }

        inline splindex<block_cyclic> const& spl_col() const
        {
            return spl_col_;
        }

        inline int rank_row() const
        {
            return rank_row_;
        }

        inline int num_ranks_row() const
        {
            return num_ranks_row_;
        }

        inline int rank_col() const
        {
            return rank_col_;
        }

        inline int num_ranks_col() const
        {
            return num_ranks_col_;
        }
        
        template <processing_unit_t pu>
        static void copy_col(dmatrix<T> const& src__, int icol_src__, dmatrix<T>& dest__, int icol_dest__)
        {
            assert(src__.num_rows_local() == dest__.num_rows_local());
            assert(src__.blacs_grid_ == dest__.blacs_grid_);

            auto src_location = src__.spl_col().location(icol_src__);
            auto dest_location = dest__.spl_col().location(icol_dest__);

            /* non-blocking send */
            if (src_location.second == src__.rank_col()) 
            {
                int tag = icol_src__;
                src__.blacs_grid_->comm_col().isend(src__.matrix_local_.at<pu>(0, src_location.first), src__.num_rows_local(), 
                                                    dest_location.second, tag);
            }

            /* blocking recieve */
            if (dest_location.second == dest__.rank_col())
            {
                int tag = icol_src__;
                src__.blacs_grid_->comm_col().recv(dest__.matrix_local_.at<pu>(0, dest_location.first), dest__.num_rows_local(),
                                                   src_location.second, tag);
            }
        }

        /// Conjugate transponse of the sub-matrix.
        /** \param [in] m Number of rows of the target sub-matrix.
         *  \param [in] n Number of columns of the target sub-matrix.
         */
        static void tranc(int32_t m, int32_t n, dmatrix<double_complex>& a, int ia, int ja, dmatrix<double_complex>& c, int ic, int jc)
        {
            ia++; ja++;
            ic++; jc++;

            lin_alg<scalapack>::pztranc(m, n, double_complex(1, 0), a.ptr(), ia, ja, a.descriptor(), double_complex(0, 0), 
                                       c.ptr(), ic, jc, c.descriptor());
        }

        static void tranu(int32_t m, int32_t n, dmatrix<double_complex>& a, int ia, int ja, dmatrix<double_complex>& c, int ic, int jc)
        {
            ia++; ja++;
            ic++; jc++;

            lin_alg<scalapack>::pztranu(m, n, double_complex(1, 0), a.ptr(), ia, ja, a.descriptor(), double_complex(0, 0), 
                                       c.ptr(), ic, jc, c.descriptor());
        }

        inline int bs() const
        {
            return bs_;
        }

};

#endif // __DMATRIX_H__


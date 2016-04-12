// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

        int bs_row_;

        int bs_col_;

        BLACS_grid const* blacs_grid_;

        splindex<block_cyclic> spl_row_;

        splindex<block_cyclic> spl_col_;

        /// Local part of the distributed matrix.
        matrix<T> matrix_local_;

        /// Matrix descriptor.
        ftn_int descriptor_[9];

        void init()
        {
            spl_row_ = splindex<block_cyclic>(num_rows_, num_ranks_row_, rank_row_, bs_row_);
            spl_col_ = splindex<block_cyclic>(num_cols_, num_ranks_col_, rank_col_, bs_col_);

            matrix_local_ = matrix<T>(nullptr, spl_row_.local_size(), spl_col_.local_size());

            #ifdef __SCALAPACK
            linalg_base::descinit(descriptor_, num_rows_, num_cols_, bs_row_, bs_col_, 0, 0,
                                  blacs_grid_->context(), matrix_local_.ld());
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
              bs_row_(1),
              bs_col_(1),
              blacs_grid_(nullptr)
        {
        }
        
        dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__)
            : num_rows_(num_rows__),
              num_cols_(num_cols__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()), 
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              bs_row_(bs_row__),
              bs_col_(bs_col__),
              blacs_grid_(&blacs_grid__)
        {
            init();
            matrix_local_.allocate();
        }

        dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__) 
            : num_rows_(num_rows__),
              num_cols_(num_cols__),
              num_ranks_row_(blacs_grid__.num_ranks_row()), 
              rank_row_(blacs_grid__.rank_row()), 
              num_ranks_col_(blacs_grid__.num_ranks_col()), 
              rank_col_(blacs_grid__.rank_col()),
              bs_row_(bs_row__),
              bs_col_(bs_col__),
              blacs_grid_(&blacs_grid__)
        {
            init();
            matrix_local_ = matrix<T>(ptr__, spl_row_.local_size(), spl_col_.local_size());
        }

        /* forbid copy constructor */
        dmatrix(dmatrix<T> const& src) = delete;
        /* forbid move constructor */
        dmatrix(dmatrix<T>&& src) = delete;
        /* forbid assigment operator */
        dmatrix<T>& operator=(dmatrix<T> const& src) = delete;

        /* This is a correct declaration of default move assigment operator:        

        dmatrix<T>& operator=(dmatrix<T>&& src) = default;

           however Intel compiler refuses to recognize it and explicit definition must be introduced.
           There is no such problem with GCC.
        */
        dmatrix<T>& operator=(dmatrix<T>&& src)
        {
            if (this != &src)
            {
                num_rows_      = src.num_rows_;
                num_cols_      = src.num_cols_;
                num_ranks_row_ = src.num_ranks_row_;
                rank_row_      = src.rank_row_;
                num_ranks_col_ = src.num_ranks_col_;
                rank_col_      = src.rank_col_;
                bs_row_        = src.bs_row_;
                bs_col_        = src.bs_col_;
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

        T* ptr()
        {
            return matrix_local_.ptr();
        }

        int ld() const
        {
            return matrix_local_.ld();
        }

        #ifdef __GPU
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

        inline void copy_to_device()
        {
            matrix_local_.copy_to_device();
        }

        inline void copy_to_host()
        {
            matrix_local_.copy_to_host();
        }

        //== inline void copy_cols_to_device(int icol_fisrt, int icol_last)
        //== {
        //==     splindex<block_cyclic> s0(icol_fisrt, num_ranks_col_, rank_col_, bs_col_);
        //==     splindex<block_cyclic> s1(icol_last,  num_ranks_col_, rank_col_, bs_col_);
        //==     int nloc = static_cast<int>(s1.local_size() - s0.local_size());
        //==     if (nloc)
        //==     {
        //==         cuda_copy_to_device(at<GPU>(0, s0.local_size()), at<CPU>(0, s0.local_size()),
        //==                             num_rows_local() * nloc * sizeof(double_complex));
        //==     }
        //== }

        //== inline void copy_cols_to_host(int icol_fisrt, int icol_last)
        //== {
        //==     splindex<block_cyclic> s0(icol_fisrt, num_ranks_col_, rank_col_, bs_col_);
        //==     splindex<block_cyclic> s1(icol_last,  num_ranks_col_, rank_col_, bs_col_);
        //==     int nloc = static_cast<int>(s1.local_size() - s0.local_size());
        //==     if (nloc)
        //==     {
        //==         cuda_copy_to_host(at<CPU>(0, s0.local_size()), at<GPU>(0, s0.local_size()),
        //==                           num_rows_local() * nloc * sizeof(double_complex));
        //==     }
        //== }
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
        inline T const* at() const
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

            /* blocking receive */
            if (dest_location.second == dest__.rank_col())
            {
                int tag = icol_src__;
                src__.blacs_grid_->comm_col().recv(dest__.matrix_local_.at<pu>(0, dest_location.first), dest__.num_rows_local(),
                                                   src_location.second, tag);
            }
        }

        inline int bs_row() const
        {
            return bs_row_;
        }

        inline int bs_col() const
        {
            return bs_col_;
        }

        inline BLACS_grid const* blacs_grid() const
        {
            return blacs_grid_;
        }
};

#endif // __DMATRIX_H__


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

/// Distributed matrix.
template <typename T>
class dmatrix: public matrix<T>
{
    private:

        /// Global number of matrix rows.
        int num_rows_{0};

        /// Global number of matrix columns.
        int num_cols_{0};

        int bs_row_{0};

        int bs_col_{0};

        BLACS_grid const* blacs_grid_{nullptr};

        splindex<block_cyclic> spl_row_;

        splindex<block_cyclic> spl_col_;

        /// Matrix descriptor.
        ftn_int descriptor_[9];

        void init()
        {
            #ifdef __SCALAPACK
            linalg_base::descinit(descriptor_, num_rows_, num_cols_, bs_row_, bs_col_, 0, 0,
                                  blacs_grid_->context(), spl_row_.local_size());
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
        
        dmatrix(int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__)
            : matrix<T>(splindex<block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row__).local_size(),
                        splindex<block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col__).local_size()),
              num_rows_(num_rows__),
              num_cols_(num_cols__),
              bs_row_(bs_row__),
              bs_col_(bs_col__),
              blacs_grid_(&blacs_grid__),
              spl_row_(num_rows_, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row_),
              spl_col_(num_cols_, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col_)
        {
            init();
        }

        dmatrix(T* ptr__, int num_rows__, int num_cols__, BLACS_grid const& blacs_grid__, int bs_row__, int bs_col__) 
            : matrix<T>(ptr__,
                        splindex<block_cyclic>(num_rows__, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row__).local_size(),
                        splindex<block_cyclic>(num_cols__, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col__).local_size()),
              num_rows_(num_rows__),
              num_cols_(num_cols__),
              bs_row_(bs_row__),
              bs_col_(bs_col__),
              blacs_grid_(&blacs_grid__),
              spl_row_(num_rows_, blacs_grid__.num_ranks_row(), blacs_grid__.rank_row(), bs_row_),
              spl_col_(num_cols_, blacs_grid__.num_ranks_col(), blacs_grid__.rank_col(), bs_col_)
        {
            init();
        }

        dmatrix(dmatrix<T>&& src) = default;

        dmatrix<T>& operator=(dmatrix<T>&& src) = default;

        /* This is a correct declaration of default move assigment operator:        

        dmatrix<T>& operator=(dmatrix<T>&& src) = default;

           however Intel compiler refuses to recognize it and explicit definition must be introduced.
           There is no such problem with GCC.
        */
        //== dmatrix<T>& operator=(dmatrix<T>&& src)
        //== {
        //==     if (this != &src)
        //==     {
        //==         num_rows_      = src.num_rows_;
        //==         num_cols_      = src.num_cols_;
        //==         //num_ranks_row_ = src.num_ranks_row_;
        //==         //rank_row_      = src.rank_row_;
        //==         //num_ranks_col_ = src.num_ranks_col_;
        //==         //rank_col_      = src.rank_col_;
        //==         bs_row_        = src.bs_row_;
        //==         bs_col_        = src.bs_col_;
        //==         blacs_grid_    = src.blacs_grid_;
        //==         spl_row_       = src.spl_row_;
        //==         spl_col_       = src.spl_col_;
        //==         //matrix_local_  = std::move(src.matrix_local_);
        //==         for (int i = 0; i < 9; i++) {
        //==             descriptor_[i] = src.descriptor_[i];
        //==         }
        //==     }
        //==     return *this;
        //== }

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

        #ifdef __GPU
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
};

#endif // __DMATRIX_H__


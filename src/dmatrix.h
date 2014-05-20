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
 *
 */

#ifndef __DMATRIX_H__
#define __DMATRIX_H__

#include "splindex.h"
#include "mdarray.h"

/// Distribued matrix.
template <typename T>
class dmatrix
{
    private:

        /// global number of matrix rows
        int num_rows_;

        /// global number of matrix columns
        int num_cols_;

        int num_ranks_row_;

        int rank_row_;

        int num_ranks_col_;

        int rank_col_;

        splindex<block_cyclic> spl_row_;

        splindex<block_cyclic> spl_col_;

        /// local part of the distributed matrix
        mdarray<T, 2> matrix_local_;

        /// matrix descriptor
        int descriptor_[9];

    public:
        
        // Default constructor assumes a 1x1 MPI grid
        dmatrix() 
            : num_ranks_row_(1), 
              rank_row_(0), 
              num_ranks_col_(1), 
              rank_col_(0)
        {
        }
        
        dmatrix(int num_rows__, int num_cols__, int blacs_context__) 
            : num_ranks_row_(1), 
              rank_row_(0), 
              num_ranks_col_(1), 
              rank_col_(0)

        {
            set_dimensions(num_rows__, num_cols__, blacs_context__);
            matrix_local_.allocate();
        }

        ~dmatrix()
        {
            matrix_local_.deallocate();
        }

        void set_dimensions(int num_rows__, int num_cols__, int blacs_context__)
        {
            num_rows_ = num_rows__;
            num_cols_ = num_cols__;

            #ifdef _SCALAPACK_
            int bs = linalg<scalapack>::cyclic_block_size();
            linalg<scalapack>::gridinfo(blacs_context__, &num_ranks_row_, &num_ranks_col_, &rank_row_, &rank_col_);
            #endif

            spl_row_ = splindex<block_cyclic>(num_rows_, num_ranks_row_, rank_row_);
            spl_col_ = splindex<block_cyclic>(num_cols_, num_ranks_col_, rank_col_);

            matrix_local_.set_dimensions(spl_row_.local_size(), spl_col_.local_size());

            #ifdef _SCALAPACK_
            linalg<scalapack>::descinit(descriptor_, num_rows_, num_cols_, bs, bs, 0, 0, blacs_context__, matrix_local_.ld());
            #endif
        }

        inline void allocate()
        {
            matrix_local_.allocate();
        }

        inline void deallocate()
        {
            matrix_local_.deallocate();
        }

        #ifdef _GPU_
        inline void allocate_page_locked()
        {
            matrix_local_.allocate_page_locked();
        }

        inline void deallocate_page_locked()
        {
            matrix_local_.deallocate_page_locked();
        }
        #endif

        inline int num_rows()
        {
            return num_rows_;
        }

        inline int num_rows_local()
        {
            return spl_row_.local_size();
        }

        inline int irow(int irow_loc)
        {
            return spl_row_[irow_loc];
        }

        inline int num_cols()
        {
            return num_cols_;
        }

        inline int num_cols_local()
        {
            return spl_col_.local_size();
        }

        inline int icol(int icol_loc)
        {
            return spl_col_[icol_loc];
        }

        inline int* descriptor()
        {
            return descriptor_;
        }

        T* ptr()
        {
            return matrix_local_.ptr();
        }

        int ld()
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

        inline T* ptr_device()
        {
            return matrix_local_.ptr_device();
        }
        #endif

        inline T& operator()(const int irow_loc, const int icol_loc) 
        {
            return matrix_local_(irow_loc, icol_loc);
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

        inline mdarray<double_complex, 2>& data()
        {
            return matrix_local_;
        }

        void zero()
        {
            matrix_local_.zero();
        }

        /// Gather full vectors from the panels
        /** 
         * Communication happens between rows of the MPI grid 
         */
        void gather(mdarray<T, 2>& full_vectors, MPI_Comm comm_row)
        {
            sirius::Timer t("dmatrix::gather");

            // trivial case
            if (num_ranks_row_ * num_ranks_col_ == 1)
            {
                matrix_local_ >> full_vectors;
                return;
            }

            splindex<block> spl_col(num_cols_local(), num_ranks_row_, rank_row_);
            
            assert((int)full_vectors.size(0) == num_rows());
            assert((int)full_vectors.size(1) == spl_col.local_size());

            std::vector<int> offsets(num_ranks_row_);
            std::vector<int> counts(num_ranks_row_);

            // each rank tosses it's local matrix to other ranks in the column
            for (int rank = 0; rank < num_ranks_row_; rank++)
            {
                // each rank allocates a subpanel
                mdarray<double_complex, 2> sub_panel(spl_row_.local_size(rank), spl_col.local_size());

                // make a table of sizes and offsets
                for (int i = 0; i < num_ranks_row_; i++)
                {
                    // offset of a sub-panel
                    offsets[i] = spl_row_.local_size(rank) * spl_col.global_offset(i);
                    // size of a sub-panel
                    counts[i] = spl_row_.local_size(rank) * spl_col.local_size(i);
                }

                // scatter local matrix between ranks
                Platform::scatter(matrix_local_.ptr(), sub_panel.ptr(), &counts[0], &offsets[0], rank, comm_row);
                
                // loop over local fraction of columns
                for (int i = 0; i < spl_col.local_size(); i++)
                {
                    // loop over local fraction of rows
                    for (int j = 0; j < spl_row_.local_size(rank); j++)
                    {
                        // copy necessary parts of panel to the full vector
                        full_vectors(spl_row_.global_index(j, rank), i) = sub_panel(j, i);
                    }
                }
            }
        }

        /// Scatter full vectors to the panels
        /** 
         * Communication happens between rows of the MPI grid 
         */
        void scatter(mdarray<double_complex, 2>& full_vectors, MPI_Comm comm_row)
        {
            sirius::Timer t("dmatrix::scatter");

            // trivial case
            if (num_ranks_row_ * num_ranks_col_ == 1)
            {
                full_vectors >> matrix_local_;
                return;
            }
            
            splindex<block> spl_col(num_cols_local(), num_ranks_row_, rank_row_);
        
            assert((int)full_vectors.size(0) == num_rows());
            assert((int)full_vectors.size(1) == spl_col.local_size());

            std::vector<int> offsets(num_ranks_row_);
            std::vector<int> counts(num_ranks_row_);
        
            for (int rank = 0; rank < num_ranks_row_; rank++)
            {
                // each rank allocates a subpanel
                mdarray<double_complex, 2> sub_panel(spl_row_.local_size(rank), spl_col.local_size());

                // fill the sub-panel
                for (int i = 0; i < spl_col.local_size(); i++)
                {
                    // loop over local fraction of rows
                    for (int j = 0; j < spl_row_.local_size(rank); j++)
                    {
                        sub_panel(j, i) = full_vectors(spl_row_.global_index(j, rank), i);
                    }
                }

                // make a table of sizes and offsets
                for (int i = 0; i < num_ranks_row_; i++)
                {
                    // offset of a sub-panel
                    offsets[i] = spl_row_.local_size(rank) * spl_col.global_offset(i);
                    // size of a sub-panel
                    counts[i] = spl_row_.local_size(rank) * spl_col.local_size(i);
                }

                // gather local matrix
                Platform::gather(sub_panel.ptr(), matrix_local_.ptr(), &counts[0], &offsets[0], rank, comm_row);
            }
        }
};

#endif // __DMATRIX_H__


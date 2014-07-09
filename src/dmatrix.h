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

        splindex<block_cyclic> spl_row_;

        splindex<block_cyclic> spl_col_;

        /// Local part of the distributed matrix.
        mdarray<T, 2> matrix_local_;

        /// Matrix descriptor.
        int descriptor_[9];

    public:
        
        // Default constructor assumes a 1x1 MPI grid.
        dmatrix() 
            : num_ranks_row_(1), 
              rank_row_(0), 
              num_ranks_col_(1), 
              rank_col_(0),
              bs_(1)
        {
        }
        
        dmatrix(int num_rows__, int num_cols__, int blacs_context__) 
            : num_ranks_row_(1), 
              rank_row_(0), 
              num_ranks_col_(1), 
              rank_col_(0),
              bs_(1)
        {
            set_dimensions(num_rows__, num_cols__, blacs_context__);
            matrix_local_.allocate();
        }

        dmatrix(T* ptr__, int num_rows__, int num_cols__, int blacs_context__) 
            : num_ranks_row_(1), 
              rank_row_(0), 
              num_ranks_col_(1), 
              rank_col_(0),
              bs_(1)
        {
            set_dimensions(num_rows__, num_cols__, blacs_context__);
            matrix_local_.set_ptr(ptr__);
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
            bs_ = linalg<scalapack>::cyclic_block_size();
            linalg<scalapack>::gridinfo(blacs_context__, &num_ranks_row_, &num_ranks_col_, &rank_row_, &rank_col_);
            #endif

            spl_row_ = splindex<block_cyclic>(num_rows_, num_ranks_row_, rank_row_, bs_);
            spl_col_ = splindex<block_cyclic>(num_cols_, num_ranks_col_, rank_col_, bs_);

            matrix_local_.set_dimensions(spl_row_.local_size(), spl_col_.local_size());

            #ifdef _SCALAPACK_
            linalg<scalapack>::descinit(descriptor_, num_rows_, num_cols_, bs_, bs_, 0, 0, blacs_context__, matrix_local_.ld());
            #endif
        }

        inline void allocate(int mode__ = 0)
        {
            matrix_local_.allocate(mode__);
        }

        inline void deallocate()
        {
            matrix_local_.deallocate();
        }

        inline int num_rows()
        {
            return num_rows_;
        }

        inline int num_rows_local()
        {
            return static_cast<int>(spl_row_.local_size());
        }

        inline int irow(int irow_loc)
        {
            return static_cast<int>(spl_row_[irow_loc]);
        }

        inline int num_cols()
        {
            return num_cols_;
        }
        
        /// Local number of columns.
        inline int num_cols_local()
        {
            return static_cast<int>(spl_col_.local_size());
        }
        
        /// Inindex of column in global matrix.
        inline int icol(int icol_loc)
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

        inline T& operator()(const int64_t irow_loc, const int64_t icol_loc) 
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

        template<int direction__>
        void shuffle(int offset__, int size__, mdarray<T, 2>& matrix_slice__, MPI_Comm comm_row__)
        {
            sirius::Timer t("dmatrix::shuffle");

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
            assert(sub_spl_col.local_size() == (int)matrix_slice__.size(1));

            std::vector<int> offsets(num_ranks_row_);
            std::vector<int> counts(num_ranks_row_);

            //T* ptr = (matrix_local_.size() == 0) ? nullptr : &matrix_local_(0, s0.local_size());
            T* ptr = (nloc == 0) ? nullptr : &matrix_local_(0, s0.local_size());

            for (int rank = 0; rank < num_ranks_row_; rank++)
            {
                /* each rank allocates a subpanel */
                mdarray<double_complex, 2> sub_panel(spl_row_.local_size(rank), sub_spl_col.local_size());

                /* make a table of sizes and offsets */
                for (int i = 0; i < num_ranks_row_; i++)
                {
                    /* offset of a sub-panel */
                    offsets[i] = static_cast<int>(spl_row_.local_size(rank) * sub_spl_col.global_offset(i));
                    /* size of a sub-panel */
                    counts[i] = static_cast<int>(spl_row_.local_size(rank) * sub_spl_col.local_size(i));
                }
                
                if (direction__ == _slice_to_panel_)
                {
                    /* fill the sub-panel */
                    for (int i = 0; i < sub_spl_col.local_size(); i++)
                    {
                        /* loop over local fraction of rows */
                        for (int j = 0; j < spl_row_.local_size(rank); j++)
                        {
                            sub_panel(j, i) = matrix_slice__(spl_row_.global_index(j, rank), i);
                        }
                    }

                    /* gather local matrix */
                    Platform::gather(sub_panel.ptr(), ptr, &counts[0], &offsets[0], rank, comm_row__);
                }

                if (direction__ == _panel_to_slice_)
                {
                    /* scatter local matrix between ranks */
                    Platform::scatter(ptr, sub_panel.ptr(), &counts[0], &offsets[0], rank, comm_row__);
                    
                    /* loop over local fraction of columns */
                    for (int i = 0; i < sub_spl_col.local_size(); i++)
                    {
                        /* loop over local fraction of rows */
                        for (int j = 0; j < spl_row_.local_size(rank); j++)
                        {
                            /* copy necessary parts of panel to the full vector */
                            matrix_slice__(spl_row_.global_index(j, rank), i) = sub_panel(j, i);
                        }
                    }
                }
            }
        }

        template<int direction__>
        void shuffle_horizontal(int N__, mdarray<T, 2>& matrix_slice__, MPI_Comm comm_col__)
        {
            // TODO: fix comments

            sirius::Timer t("dmatrix::shuffle_horizontal");

            ///* trivial case */
            //if (num_ranks_row_ * num_ranks_col_ == 1 && offset__ == 0 && size__ == num_cols_)
            //{
            //    if (direction__ == _slice_to_panel_) matrix_slice__ >> matrix_local_;
            //    if (direction__ == _panel_to_slice_) matrix_local_ >> matrix_slice__;
            //    return;
            //}

            splindex<block_cyclic> spl_N(N__, num_ranks_col_, rank_col_, bs_);

            mdarray<T, 2> tmp(spl_N.local_size(), spl_row_.local_size());

            /* transpose local matrix */
            if (direction__ == _panel_to_slice_)
            {
                for (int i = 0; i < spl_row_.local_size(); i++)
                {
                    for (int j = 0; j < spl_N.local_size(); j++) tmp(j, i) = matrix_local_(i, j);
                }
            }

            T* ptr = (tmp.size() == 0) ? nullptr : &tmp(0, 0);

            splindex<block> sub_spl_row(spl_row_.local_size(), num_ranks_col_, rank_col_);

            assert(N__ == (int)matrix_slice__.size(0));
            assert(sub_spl_row.local_size() == (int)matrix_slice__.size(1));
            
            #pragma omp parallel num_threads(2)
            {
                std::vector<int> offsets(num_ranks_col_);
                std::vector<int> counts(num_ranks_col_);

                std::vector<double_complex> sub_panel_tmp(spl_N.local_size(0) * sub_spl_row.local_size());
                
                #pragma omp for
                for (int rank = 0; rank < num_ranks_col_; rank++)
                {
                    /* each rank allocates a subpanel */
                    //mdarray<double_complex, 2> sub_panel(spl_N.local_size(rank), sub_spl_row.local_size());
                    mdarray<double_complex, 2> sub_panel(&sub_panel_tmp[0], spl_N.local_size(rank), sub_spl_row.local_size());

                    /* make a table of sizes and offsets */
                    for (int i = 0; i < num_ranks_col_; i++)
                    {
                        /* offset of a sub-panel */
                        offsets[i] = static_cast<int>(spl_N.local_size(rank) * sub_spl_row.global_offset(i));
                        /* size of a sub-panel */
                        counts[i] = static_cast<int>(spl_N.local_size(rank) * sub_spl_row.local_size(i));
                    }
                    
                    if (direction__ == _slice_to_panel_)
                    {
                        /* fill the sub-panel */
                        for (int i = 0; i < sub_spl_row.local_size(); i++)
                        {
                            /* loop over local fraction of rows */
                            for (int j = 0; j < spl_N.local_size(rank); j++)
                            {
                                sub_panel(j, i) = matrix_slice__(spl_N.global_index(j, rank), i);
                            }
                        }

                        /* gather local matrix */
                        Platform::gather(sub_panel.ptr(), ptr, &counts[0], &offsets[0], rank, comm_col__);
                    }

                    if (direction__ == _panel_to_slice_)
                    {
                        /* scatter local matrix between ranks */
                        Platform::scatter(ptr, sub_panel.ptr(), &counts[0], &offsets[0], rank, comm_col__);
                        
                        /* loop over local fraction of columns */
                        for (int i = 0; i < sub_spl_row.local_size(); i++)
                        {
                            /* loop over local fraction of rows */
                            for (int j = 0; j < spl_N.local_size(rank); j++)
                            {
                                /* copy necessary parts of panel to the full vector */
                                matrix_slice__(spl_N.global_index(j, rank), i) = sub_panel(j, i);
                            }
                        }
                    }
                }
            }

            /* transpose back to local matrix */
            if (direction__ == _slice_to_panel_)
            {
                for (int i = 0; i < spl_row_.local_size(); i++)
                {
                    for (int j = 0; j < spl_N.local_size(); j++) matrix_local_(i, j) = tmp(j, i);
                }
            }
        }

        /// Gather full vectors from the panels
        /** 
         * Communication happens between rows of the MPI grid 
         */
        void gather(mdarray<T, 2>& full_vectors__, MPI_Comm comm_row__)
        {
            shuffle<_panel_to_slice_>(0, num_cols_, full_vectors__, comm_row__);
        }

        void gather(int n__, int offs__, mdarray<T, 2>& matrix_slice__, MPI_Comm comm_row__)
        {
            shuffle<_panel_to_slice_>(offs__, n__, matrix_slice__, comm_row__);
        }

        /// Scatter full vectors to the panels
        /** 
         * Communication happens between rows of the MPI grid 
         */
        void scatter(mdarray<double_complex, 2>& full_vectors__, MPI_Comm comm_row__)
        {
            shuffle<_slice_to_panel_>(0, num_cols_, full_vectors__, comm_row__);
        }

        void scatter(int n__, int offs__, mdarray<T, 2>& matrix_slice__, MPI_Comm comm_row__)
        {
            shuffle<_slice_to_panel_>(offs__, n__, matrix_slice__, comm_row__);
        }

        inline splindex<block_cyclic>& spl_col()
        {
            return spl_col_;
        }

        inline int rank_col()
        {
            return rank_col_;
        }

        static void copy_col(dmatrix<T>& src__, int icol_src__, dmatrix<T>& dest__, int icol_dest__, MPI_Comm comm_col__)
        {
            assert(src__.num_rows_local() == dest__.num_rows_local());

            auto src_location = src__.spl_col().location(icol_src__);
            auto dest_location = dest__.spl_col().location(icol_dest__);

            /* non-blocking send */
            if (src_location.second == src__.rank_col()) 
            {
                int tag = icol_src__;
                Platform::isend(&src__.matrix_local_(0, src_location.first), src__.num_rows_local(), dest_location.second, 
                                tag, comm_col__);
            }

            /* blocking recieve */
            if (dest_location.second == dest__.rank_col())
            {
                int tag = icol_src__;
                Platform::recv(&dest__.matrix_local_(0, dest_location.first), dest__.num_rows_local(), src_location.second, 
                               tag, comm_col__);
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

            linalg<scalapack>::pztranc(m, n, double_complex(1, 0), a.ptr(), ia, ja, a.descriptor(), double_complex(0, 0), 
                                       c.ptr(), ic, jc, c.descriptor());
        }
};

#endif // __DMATRIX_H__


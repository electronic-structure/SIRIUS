// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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
#include "utils.h"

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

        dmatrix<T>& operator=(dmatrix<T>&& src) = default;
        //{
        //    if (this != &src)
        //    {
        //        num_rows_      = src.num_rows_;
        //        num_cols_      = src.num_cols_;
        //        num_ranks_row_ = src.num_ranks_row_;
        //        rank_row_      = src.rank_row_;
        //        num_ranks_col_ = src.num_ranks_col_;
        //        rank_col_      = src.rank_col_;
        //        bs_row_        = src.bs_row_;
        //        bs_col_        = src.bs_col_;
        //        blacs_grid_    = src.blacs_grid_;
        //        spl_row_       = src.spl_row_;
        //        spl_col_       = src.spl_col_;
        //        matrix_local_  = std::move(src.matrix_local_);
        //        for (int i = 0; i < 9; i++) descriptor_[i] = src.descriptor_[i];
        //    }
        //    return *this;
        //}

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

//== class redist
//== {
//==     private:
//== 
//==         static inline std::vector< std::vector<double_complex> >& send_buf()
//==         {
//==             static std::vector< std::vector<double_complex> > send_buf_;
//==             return send_buf_;
//==         }
//== 
//==         static inline std::vector< std::vector<double_complex> >& recv_buf()
//==         {
//==             static std::vector< std::vector<double_complex> > recv_buf_;
//==             return recv_buf_;
//==         }
//== 
//==         struct a2a_buffer
//==         {
//==             int num_ranks;
//==             int size;
//==             int* counts;
//==             int* offsets;
//==             double_complex* buff;
//== 
//==             a2a_buffer() : num_ranks(0), counts(NULL), offsets(NULL), buff(NULL)
//==             {
//==             }
//== 
//==             ~a2a_buffer()
//==             {
//==                 if (counts) free(counts);
//==                 if (offsets) free(offsets);
//==                 if (buff) free(buff);
//==             }
//== 
//==             void realloc(int num_ranks__, int size__)
//==             {
//==                 num_ranks = num_ranks__;
//==                 size = size__;
//==                 
//==                 if (counts) free(counts);
//==                 counts = (int*)malloc(num_ranks * sizeof(int));
//==                 
//==                 if (offsets) free(offsets);
//==                 offsets = (int*)malloc(num_ranks * sizeof(int));
//==                 
//==                 if (buff) free(buff);
//==                 buff = (double_complex*)malloc(size * sizeof(double_complex));
//==             }
//==         };
//== 
//==         static inline a2a_buffer& sbuf()
//==         {
//==             static a2a_buffer sbuf_;
//==             return sbuf_;
//==         }
//== 
//==         static inline a2a_buffer& rbuf()
//==         {
//==             static a2a_buffer rbuf_;
//==             return rbuf_;
//==         }
//==             
//==     public:
//== 
//==         static void gemr2d(int M__, int N__,
//==                            dmatrix<double_complex>& A__, int ia__, int ja__,
//==                            dmatrix<double_complex>& B__, int ib__, int jb__)
//==         {
//==             if (A__.blacs_grid()->comm().mpi_comm() != B__.blacs_grid()->comm().mpi_comm())
//==             {
//==                 TERMINATE("source and destination communicators don't match");
//==             }
//==             auto& comm = A__.blacs_grid()->comm();
//==             int num_ranks = comm.size();
//== 
//==             if (ia__ != 0 || ib__ != 0)
//==             {
//==                 TERMINATE_NOT_IMPLEMENTED
//==             }
//== 
//==             if (A__.num_rows() != M__ || B__.num_rows() != M__)
//==             {
//==                 TERMINATE_NOT_IMPLEMENTED
//==             }
//== 
//== 
//==             /* slab to slice transformation */
//==             if (A__.num_ranks_row() == num_ranks && 
//==                 A__.bs_row() == splindex_base<int>::block_size(A__.num_rows(), num_ranks) &&
//==                 B__.num_ranks_col() == num_ranks &&
//==                 B__.bs_col() == 1)
//==             {
//==                 if (jb__ != 0)
//==                 {
//==                     TERMINATE_NOT_IMPLEMENTED
//==                 }
//== 
//==                 splindex<block_cyclic> sb0(jb__,       num_ranks, B__.rank_col(), B__.bs_col());
//==                 splindex<block_cyclic> sb1(jb__ + N__, num_ranks, B__.rank_col(), B__.bs_col());
//== 
//==                 int sz = N__ * A__.num_rows_local();
//==                 if (sbuf().num_ranks < num_ranks || sbuf().size < sz) sbuf().realloc(num_ranks, sz);
//==                 
//==                 int i = 0;
//==                 for (int rank = 0; rank < num_ranks; rank++)
//==                 {
//==                     sbuf().counts[rank] = (int)(sb1.local_size(rank) - sb0.local_size(rank)) * A__.num_rows_local();
//==                     sbuf().offsets[rank] = i;
//==                     i += sbuf().counts[rank];
//==                 }
//==                 if (i != sz) TERMINATE("wrong size");
//== 
//==                 /* pack */
//==                 #pragma omp parallel for
//==                 for (int icol = 0; icol < N__; icol++)
//==                 {
//==                     auto loc = B__.spl_col().location(jb__ + icol);
//==                     int rank = loc.second;
//==                     int j = (int)(loc.first - sb0.local_size());
//==                     memcpy(&sbuf().buff[sbuf().offsets[rank] + j * A__.num_rows_local()],
//==                            &A__(0, icol + ja__), A__.num_rows_local() * sizeof(double_complex));
//==                 }
//== 
//==                 sz = M__ * (int)(sb1.local_size() - sb0.local_size());
//==                 if (rbuf().num_ranks < num_ranks || rbuf().size < sz) rbuf().realloc(num_ranks, sz);
//==                 i = 0;
//==                 for (int rank = 0; rank < num_ranks; rank++)
//==                 {
//==                     rbuf().counts[rank] = (int)((sb1.local_size() - sb0.local_size()) * A__.spl_row().local_size(rank));
//==                     rbuf().offsets[rank] = i;
//==                     i += rbuf().counts[rank];
//==                 }
//==                 if (i != sz) TERMINATE("wrong size");
//== 
//==                 comm.alltoall(sbuf().buff, sbuf().counts, sbuf().offsets, rbuf().buff, rbuf().counts, rbuf().offsets);
//==                 
//==                 /* unpack */
//==                 for (int rank = 0; rank < num_ranks; rank++)
//==                 {
//==                     for (int j = 0; j < (int)(sb1.local_size() - sb0.local_size()); j++)
//==                     {
//==                         memcpy(&B__(A__.spl_row().global_index(0, rank), sb0.local_size() + j),
//==                                &rbuf().buff[rbuf().offsets[rank] + j * A__.spl_row().local_size(rank)],
//==                                A__.spl_row().local_size(rank) * sizeof(double_complex));
//==                     }
//==                 }
//==             }
//==             else
//==             if (B__.num_ranks_row() == num_ranks && 
//==                 B__.bs_row() == splindex_base<int>::block_size(B__.num_rows(), num_ranks) &&
//==                 A__.num_ranks_col() == num_ranks &&
//==                 A__.bs_col() == 1)
//==             {
//==                 if (ja__ != 0)
//==                 {
//==                     TERMINATE_NOT_IMPLEMENTED
//==                 }
//== 
//==                 splindex<block_cyclic> sa0(ja__,       num_ranks, A__.rank_col(), A__.bs_col());
//==                 splindex<block_cyclic> sa1(ja__ + N__, num_ranks, A__.rank_col(), A__.bs_col());
//== 
//==                 int sz = M__ * (int)(sa1.local_size() - sa0.local_size());
//==                 if (sbuf().num_ranks < num_ranks || sbuf().size < sz) sbuf().realloc(num_ranks, sz);
//==                 int i = 0;
//==                 for (int rank = 0; rank < num_ranks; rank++)
//==                 {
//==                     sbuf().counts[rank] = (int)((sa1.local_size() - sa0.local_size()) * B__.spl_row().local_size(rank));
//==                     sbuf().offsets[rank] = i;
//==                     i += sbuf().counts[rank];
//==                 }
//==                 if (i != sz) TERMINATE("wrong size");
//==                 /* pack */
//==                 for (int rank = 0; rank < num_ranks; rank++)
//==                 {
//==                     for (int j = 0; j < (int)(sa1.local_size() - sa0.local_size()); j++)
//==                     {
//==                         memcpy(&sbuf().buff[sbuf().offsets[rank] + j * B__.spl_row().local_size(rank)],
//==                                &A__(B__.spl_row().global_index(0, rank), sa0.local_size() + j),
//==                                B__.spl_row().local_size(rank) * sizeof(double_complex));
//==                     }
//==                 }
//== 
//==                 sz = N__ * B__.num_rows_local();
//==                 if (rbuf().num_ranks < num_ranks || rbuf().size < sz) rbuf().realloc(num_ranks, sz);
//==                 
//==                 i = 0;
//==                 for (int rank = 0; rank < num_ranks; rank++)
//==                 {
//==                     rbuf().counts[rank] = (int)(sa1.local_size(rank) - sa0.local_size(rank)) * B__.num_rows_local();
//==                     rbuf().offsets[rank] = i;
//==                     i += rbuf().counts[rank];
//==                 }
//==                 if (i != sz) TERMINATE("wrong size");
//== 
//==                 comm.alltoall(sbuf().buff, sbuf().counts, sbuf().offsets, rbuf().buff, rbuf().counts, rbuf().offsets);
//== 
//==                 /*unpack */
//==                 #pragma omp parallel for
//==                 for (int icol = 0; icol < N__; icol++)
//==                 {
//==                     auto loc = A__.spl_col().location(ja__ + icol);
//==                     int rank = loc.second;
//==                     int j = (int)(loc.first - sa0.local_size());
//==                     memcpy(&B__(0, icol + jb__), &rbuf().buff[rbuf().offsets[rank] + j * B__.num_rows_local()],
//==                            B__.num_rows_local() * sizeof(double_complex));
//==                 }
//==             }
//==             else
//==             {
//==                 TERMINATE_NOT_IMPLEMENTED
//==             }
//== 
//==                     
//== 
//== 
//== 
//== 
//== 
//==             //if (sbuf().num_ranks < num_ranks || sbuf().size < src__.panel().size()) sbuf().realloc(num_ranks, 
//== 
//== 
//== 
//==             //== double t0 = -Utils::current_time();
//==             //== std::vector< std::pair<size_t, int> > dest_location_row(src__.num_rows_local());
//==             //== std::vector< std::pair<size_t, int> > dest_location_col(src__.num_cols_local());
//== 
//==             //== // TODO: investigate this
//==             //== /* for unknown reason using std::vector<T> buf() is extremaly slow here
//==             //==  * and I can't reproduce this behaviour with a stand-alone test;
//==             //==  * current solution is to use malloc/free for all buffers */
//==             //== int* count = (int*)malloc(num_ranks * sizeof(int));
//==             //== for (int i = 0; i < num_ranks; i++) count[i] = 0;
//== 
//==             //== int special_case = 0;
//== 
//==             //== if (src__.num_ranks_row() == num_ranks && 
//==             //==     src__.bs_row() == splindex_base::block_size(src__.num_rows(), num_ranks) &&
//==             //==     dest__.num_ranks_col() == num_ranks &&
//==             //==     dest__.bs_col() == 1)
//==             //== {
//==             //==     special_case = 1;
//==             //== }
//== 
//==             //== if (dest__.num_ranks_row() == num_ranks && 
//==             //==     dest__.bs_row() == splindex_base::block_size(dest__.num_rows(), num_ranks) &&
//==             //==     src__.num_ranks_col() == num_ranks &&
//==             //==     src__.bs_col() == 1)
//==             //== {
//==             //==     special_case = 2;
//==             //== }
//== 
//==             //== /* get sending buffers sizes */
//==             //== if (special_case == 0) /* generic case */
//==             //== {
//==             //==     #pragma omp parallel
//==             //==     {
//==             //==         #pragma omp for
//==             //==         for (int iloc = 0; iloc < src__.num_rows_local(); iloc++)
//==             //==         {
//==             //==             int i = src__.irow(iloc);
//==             //==             dest_location_row[iloc] = dest__.spl_row().location(i);
//==             //==         }
//== 
//==             //==         #pragma omp for
//==             //==         for (int jloc = 0; jloc < src__.num_cols_local(); jloc++)
//==             //==         {
//==             //==             int j = src__.icol(jloc);
//==             //==             dest_location_col[jloc] = dest__.spl_col().location(j);
//==             //==         }
//==             //==     }
//==             //==     for (int jloc = 0; jloc < src__.num_cols_local(); jloc++)
//==             //==     {
//==             //==         auto& lj = dest_location_col[jloc];
//==             //==         for (int iloc = 0; iloc < src__.num_rows_local(); iloc++)
//==             //==         {
//==             //==             auto& li = dest_location_row[iloc];
//==             //==             int rank = dest__.blacs_grid()->cart_rank(li.second, lj.second);
//==             //==             count[rank]++;
//==             //==         }
//==             //==     }
//==             //== }
//==             //== if (special_case == 1) /* special case: slabs to slices */
//==             //== {
//==             //==     for (int i = 0; i < num_ranks; i++) count[i] = src__.num_rows_local() * dest__.spl_col().local_size(i);
//==             //== }
//==             //== if (special_case == 2) /* special case: slices to slabs */
//==             //== {
//==             //==     for (int i = 0; i < num_ranks; i++) count[i] = src__.num_cols_local() * dest__.spl_row().local_size(i);
//==             //== }
//== 
//==             //== /* allocate sending buffers */
//==             //== double_complex** send_buf = (double_complex**)malloc(num_ranks * sizeof(double_complex*));
//==             //== for (int i = 0; i < num_ranks; i++)
//==             //== {
//==             //==     send_buf[i] = (double_complex*)malloc(count[i] * sizeof(double_complex));
//==             //== }
//== 
//==             //== /* fill sending buffers */
//==             //== if (special_case == 0)
//==             //== {
//==             //==     for (int i = 0; i < num_ranks; i++) count[i] = 0;
//==             //==     for (int jloc = 0; jloc < src__.num_cols_local(); jloc++)
//==             //==     {
//==             //==         auto& lj = dest_location_col[jloc];
//==             //==         for (int iloc = 0; iloc < src__.num_rows_local(); iloc++)
//==             //==         {
//==             //==             auto& li = dest_location_row[iloc];
//==             //==             int rank = dest__.blacs_grid()->cart_rank(li.second, lj.second);
//==             //==             send_buf[rank][count[rank]++] = src__(iloc, jloc);
//==             //==         }
//==             //==     }
//==             //== }
//==             //== if (special_case == 1)
//==             //== {
//==             //==     /* number of elements to copy */
//==             //==     int len = src__.num_rows_local();
//==             //==     #pragma omp parallel for
//==             //==     for (int i = 0; i < src__.num_cols(); i++)
//==             //==     {
//==             //==         auto loc = dest__.spl_col().location(i);
//==             //==         /* copy elements from source to temporary buffer */
//==             //==         std::copy(&src__(0, i), &src__(0, i) + len, &send_buf[loc.second][loc.first * len]);
//==             //==     }
//== 
//==             //==     //for (int rank = 0; rank < num_ranks; rank++)
//==             //==     //{
//==             //==     //    #pragma omp for
//==             //==     //    for (int iloc = 0; iloc < dest__.spl_col().local_size(rank); iloc++)
//==             //==     //    {
//==             //==     //        /* global index of the column */
//==             //==     //        int i = dest__.spl_col().global_index(iloc, rank);
//==             //==     //        /* copy elements from source to temporary buffer */
//==             //==     //        std::copy(&src__(0, i), &src__(0, i) + len, &send_buf[rank][iloc * len]);
//==             //==     //    }
//==             //==     //}
//==             //== }
//==             //== if (special_case == 2)
//==             //== {
//==             //==     #pragma omp parallel
//==             //==     for (int rank = 0; rank < num_ranks; rank++)
//==             //==     {
//==             //==         /* number of elements to copy */
//==             //==         int len = dest__.spl_row().local_size(rank);
//==             //==         /* offset in the full row (src holds the whole rows for fraction of the columns) */
//==             //==         int i = dest__.spl_row().global_index(0, rank);
//==             //==         #pragma omp for
//==             //==         for (int iloc = 0; iloc < src__.num_cols_local(); iloc++)
//==             //==         {
//==             //==             /* copy elements from source to temporary buffer */
//==             //==             std::copy(&src__(i, iloc), &src__(i, iloc) + len, &send_buf[rank][iloc * len]);
//==             //==         }
//==             //==     }
//==             //== }
//==             //== t0 += Utils::current_time();
//==             //== printf("pack: %f, speed %f GB/s\n", t0, sizeof(double_complex) * src__.panel().size() / t0 / double(1<<30));
//== 
//==             //== /* get the number of recieved elements from each rank */
//==             //== std::vector<int> recv_counts(num_ranks);
//==             //== src__.blacs_grid()->comm().alltoall(count, 1, &recv_counts[0], 1); 
//== 
//==             //== /* allocate recieve buffers */
//==             //== double_complex** recv_buf = (double_complex**)malloc(num_ranks * sizeof(double_complex*));
//==             //== for (int i = 0; i < num_ranks; i++)
//==             //== {
//==             //==     recv_buf[i] = (double_complex*)malloc(recv_counts[i] * sizeof(double_complex));
//==             //== }
//==             //== 
//==             //== /* perform the data exchange */
//==             //== double t1 = -Utils::current_time();
//==             //== for (int i = 0; i < num_ranks; i++)
//==             //== {
//==             //==     int tag = comm.rank() * num_ranks + i;
//==             //==     comm.isend(send_buf[i], count[i], i, tag);
//==             //== }
//== 
//==             //== for (int i = 0; i < num_ranks; i++)
//==             //== {
//==             //==     int tag = i * num_ranks + comm.rank();
//==             //==     comm.recv(recv_buf[i], recv_counts[i], i, tag);
//==             //== }
//==             //== comm.barrier();
//==             //== t1 += Utils::current_time();
//==             //== printf("exchange: %f\n", t1);
//== 
//==             //== double t2 = -Utils::current_time();
//==             //== if (special_case == 0)
//==             //== {
//==             //==     std::vector< std::pair<size_t, int> > src_location_row(dest__.num_rows_local());
//==             //==     std::vector< std::pair<size_t, int> > src_location_col(dest__.num_cols_local());
//==             //==     #pragma omp parallel
//==             //==     {
//==             //==         #pragma omp for
//==             //==         for (int iloc = 0; iloc < dest__.num_rows_local(); iloc++)
//==             //==         {
//==             //==             int i = dest__.irow(iloc);
//==             //==             src_location_row[iloc] = src__.spl_row().location(i);
//==             //==         }
//== 
//==             //==         #pragma omp for
//==             //==         for (int jloc = 0; jloc < dest__.num_cols_local(); jloc++)
//==             //==         {
//==             //==             int j = dest__.icol(jloc);
//==             //==             src_location_col[jloc] = src__.spl_col().location(j);
//==             //==         }
//==             //==     }
//==             //==     for (int i = 0; i < num_ranks; i++) count[i] = 0;
//== 
//==             //==     for (int jloc = 0; jloc < dest__.num_cols_local(); jloc++)
//==             //==     {
//==             //==         auto& lj = src_location_col[jloc];
//==             //==         for (int iloc = 0; iloc < dest__.num_rows_local(); iloc++)
//==             //==         {
//==             //==             auto& li = src_location_row[iloc];
//==             //==             int rank = src__.blacs_grid()->cart_rank(li.second, lj.second);
//==             //==             dest__(iloc, jloc) = recv_buf[rank][count[rank]++];
//==             //==         }
//==             //==     }
//== 
//==             //==     for (int i = 0; i < num_ranks; i++)
//==             //==     {
//==             //==         if (count[i] != recv_counts[i]) TERMINATE("wrong counts");
//==             //==     }
//==             //== }
//==             //== if (special_case == 1)
//==             //== {
//==             //==     #pragma omp parallel
//==             //==     for (int rank = 0; rank < num_ranks; rank++)
//==             //==     {
//==             //==         /* offset in the full row (dest holds the whole rows for fraction of the columns) */
//==             //==         int i = src__.spl_row().global_index(0, rank);
//==             //==         /* number of elements to copy */
//==             //==         int len = src__.spl_row().local_size(rank);
//==             //==         #pragma omp for
//==             //==         for (int iloc = 0; iloc < dest__.num_cols_local(); iloc++)
//==             //==         {
//==             //==             /* copy elements from source to temporary buffer */
//==             //==             std::copy(&recv_buf[rank][iloc * len], &recv_buf[rank][iloc * len] + len, &dest__(i, iloc));
//==             //==         }
//==             //==     }
//==             //== }
//==             //== if (special_case == 2)
//==             //== {
//==             //==     /* number of elements to copy */
//==             //==     int len = dest__.num_rows_local();
//==             //==     #pragma omp parallel for
//==             //==     for (int i = 0; i < dest__.num_cols(); i++)
//==             //==     {
//==             //==         auto loc = src__.spl_col().location(i);
//==             //==         /* copy elements from source to temporary buffer */
//==             //==         std::copy(&recv_buf[loc.second][loc.first * len], &recv_buf[loc.second][loc.first * len] + len, &dest__(0, i));
//==             //==     }
//==             //== }
//==             //== t2 += Utils::current_time();
//==             //== printf("unpack: %f, speed %f GB/s\n", t2, sizeof(double_complex) * dest__.panel().size() / t2 / double(1<<30));
//== 
//==             //== for (int i = 0; i < num_ranks; i++)
//==             //== {
//==             //==     free(send_buf[i]);
//==             //==     free(recv_buf[i]);
//==             //== }
//==             //== free(send_buf);
//==             //== free(recv_buf);
//==             //== free(count);
//==         }
//== 
//== };

#endif // __DMATRIX_H__


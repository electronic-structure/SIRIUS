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

/** \file gvec.h
 *
 *  \brief Declaration and implementation of sirius::Gvec class.
 */

#ifndef __GVEC_H__
#define __GVEC_H__

#include <numeric>

#include "sirius_internal.h"
#include "descriptors.h"
#include "fft3d_grid.h"
#include "mpi_grid.h"

namespace sirius {

class Gvec_FFT_distribution;

/// Store list of G-vectors for FFTs and G+k basis functions.
class Gvec
{
    friend class Gvec_FFT_distribution;

    private:

        vector3d<double> q_;
        
        /// FFT box for which G-vectors are generated.
        FFT3D_grid fft_box_;

        /// Reciprocal lattice vectors.
        matrix3d<double> lattice_vectors_;
        
        /// Indicates of G-vectors are reduced by inversion symmetry.
        bool reduce_gvec_;

        int num_ranks_;

        /// Total number of G-vectors.
        int num_gvec_;

        /// Mapping between G-vector index [0:num_gvec_) and a full index.
        /** Full index is used to store x,y,z coordinates in a packed form in a single integer number. */
        mdarray<int, 1> gvec_full_index_;
    
        /// Index of the shell to which the given G-vector belongs.
        mdarray<int, 1> gvec_shell_;
        
        /// List of {x,y} positions of z-columns. 
        mdarray<int, 2> z_columns_pos_;

        int num_gvec_shells_;

        mdarray<double, 1> gvec_shell_len_;

        mdarray<int, 3> index_by_gvec_;

        /// Global list of non-zero z-columns.
        std::vector<z_column_descriptor> z_columns_;

        /// Fine-grained distribution of G-vectors.
        block_data_descriptor gvec_distr_;
        block_data_descriptor zcol_distr_;

        Gvec(Gvec const& src__) = delete;

        Gvec& operator=(Gvec const& src__) = delete;

    public:

        Gvec()
        {
        }

        Gvec(vector3d<double>        q__,
             matrix3d<double> const& M__,
             double                  Gmax__,
             FFT3D_grid const&       fft_box__,
             int                     num_ranks__,
             bool                    build_reverse_mapping__,
             bool                    reduce_gvec__)
            : q_(q__),
              fft_box_(fft_box__),
              lattice_vectors_(M__),
              reduce_gvec_(reduce_gvec__),
              num_ranks_(num_ranks__)
        {
            mdarray<int, 2> non_zero_columns(fft_box_.limits(0), fft_box_.limits(1));
            non_zero_columns.zero();

            num_gvec_ = 0;
            for (int i = fft_box_.limits(0).first; i <= fft_box_.limits(0).second; i++)
            {
                for (int j = fft_box_.limits(1).first; j <= fft_box_.limits(1).second; j++)
                {
                    std::vector<int> zcol;
                    
                    /* in general case take z in [0, Nz) */ 
                    int zmax = fft_box_.size(2) - 1;
                    /* in case of G-vector reduction take z in [0, Nz/2] for {x=0,y=0} stick */
                    if (reduce_gvec_ && !i && !j) zmax = fft_box_.limits(2).second;
                    /* loop over z-coordinates of FFT grid */ 
                    for (int iz = 0; iz <= zmax; iz++)
                    {
                        /* get z-coordinate of G-vector */
                        int k = (iz > fft_box_.limits(2).second) ? iz - fft_box_.size(2) : iz;
                        /* take G+q */
                        auto gq = lattice_vectors_ * (vector3d<double>(i, j, k) + q__);
                        /* add z-coordinate of G-vector to the list */
                        if (gq.length() <= Gmax__) zcol.push_back(k);
                    }
                    
                    if (zcol.size() && !non_zero_columns(i, j))
                    {
                        z_columns_.push_back(z_column_descriptor(i, j, zcol));
                        num_gvec_ += static_cast<int>(zcol.size());

                        non_zero_columns(i, j) = 1;
                        if (reduce_gvec__) non_zero_columns(-i, -j) = 1;
                    }
                }
            }
            
            /* put column with {x, y} = {0, 0} to the beginning */
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                if (z_columns_[i].x == 0 && z_columns_[i].y == 0)
                {
                    std::swap(z_columns_[i], z_columns_[0]);
                    break;
                }
            }
            
            /* sort z-columns starting from the second */
            std::sort(z_columns_.begin() + 1, z_columns_.end(),
                      [](z_column_descriptor const& a, z_column_descriptor const& b)
                      {
                          return a.z.size() > b.z.size();
                      });
            
            /* distribute z-columns between N ranks */
            gvec_distr_ = block_data_descriptor(num_ranks__);
            zcol_distr_ = block_data_descriptor(num_ranks__);
            /* local number of z-columns for each rank */
            std::vector< std::vector<z_column_descriptor> > zcols_local(num_ranks__);

            std::vector<int> ranks;
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                /* initialize the list of ranks to 0,1,2,... */
                if (ranks.empty())
                {
                    ranks.resize(num_ranks__);
                    std::iota(ranks.begin(), ranks.end(), 0);
                }
                /* find rank with minimum number of G-vectors */
                auto rank_with_min_gvec = std::min_element(ranks.begin(), ranks.end(), 
                                                           [this](const int& a, const int& b)
                                                           {
                                                               return gvec_distr_.counts[a] < gvec_distr_.counts[b];
                                                           });

                /* assign column to the found rank */
                zcols_local[*rank_with_min_gvec].push_back(z_columns_[i]);
                zcol_distr_.counts[*rank_with_min_gvec] += 1;
                /* count local number of G-vectors */
                gvec_distr_.counts[*rank_with_min_gvec] += static_cast<int>(z_columns_[i].z.size());
                /* exclude this rank from the search */
                ranks.erase(rank_with_min_gvec);
            }
            gvec_distr_.calc_offsets();
            zcol_distr_.calc_offsets();

            /* save new ordering of z-columns */
            z_columns_.clear();
            for (int rank = 0; rank < num_ranks__; rank++)
            {
                z_columns_.insert(z_columns_.end(), zcols_local[rank].begin(), zcols_local[rank].end());
            }
            
            /* build simple array of {x,y} coordinates for GPU kernel */
            #ifdef __GPU
            z_columns_pos_ = mdarray<int, 2>(2, z_columns_.size());
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                z_columns_pos_(0, i) = z_columns_[i].x;
                z_columns_pos_(1, i) = z_columns_[i].y;
            }
            z_columns_pos_.allocate_on_device();
            z_columns_pos_.copy_to_device();
            #endif

            /* build the full G-vector index */
            gvec_full_index_ = mdarray<int, 1>(num_gvec_);
            int ig = 0;
            for (size_t i = 0; i < z_columns_.size(); i++)
            {
                for (size_t j = 0; j < z_columns_[i].z.size(); j++)
                    gvec_full_index_(ig++) = static_cast<int>((i << 12) + j);
            }
            
            /* first G-vector must be (0, 0, 0); never reomove this check!!! */
            auto g0 = gvec_by_full_index(gvec_full_index_(0));
            if (g0[0] || g0[1] || g0[2]) TERMINATE("first G-vector is not zero");
        
            /* find G-shells */
            std::map<size_t, std::vector<int> > gsh;
            for (int ig = 0; ig < num_gvec_; ig++)
            {
                /* take G+q */
                auto gq = cart_shifted(ig);
                /* make some reasonable roundoff */
                size_t len = size_t(gq.length() * 1e10);

                if (!gsh.count(len)) gsh[len] = std::vector<int>();
                gsh[len].push_back(ig);
            }
            num_gvec_shells_ = static_cast<int>(gsh.size());
            gvec_shell_ = mdarray<int, 1>(num_gvec_);
            gvec_shell_len_ = mdarray<double, 1>(num_gvec_shells_);
            
            int n = 0;
            for (auto it = gsh.begin(); it != gsh.end(); it++)
            {
                gvec_shell_len_(n) = static_cast<double>(it->first) * 1e-10;
                for (int ig: it->second) gvec_shell_(ig) = n;
                n++;
            }
            
            /* build a mapping between G-vector and it's index */
            if (build_reverse_mapping__)
            {
                index_by_gvec_ = mdarray<int, 3>(fft_box_.limits(0), fft_box_.limits(1), fft_box_.limits(2));

                std::fill(index_by_gvec_.at<CPU>(), index_by_gvec_.at<CPU>() + fft_box_.size(), -1);

                for (int ig = 0; ig < num_gvec_; ig++)
                {
                    auto G = (*this)[ig];
                    index_by_gvec_(G[0], G[1], G[2]) = ig;
                }
            }
        }

        /// Default move assigment operator.
        Gvec& operator=(Gvec&& src__) = default;

        /// Return the total number of G-vectors within the cutoff.
        inline int num_gvec() const
        {
            return num_gvec_;
        }

        /// Number of G-vectors for a fine-grained distribution.
        inline int num_gvec(int rank__) const
        {
            assert((size_t)rank__ < gvec_distr_.counts.size());
            return gvec_distr_.counts[rank__];
        }

        /// Offset (in the global index) of G-vectors for a fine-grained distribution.
        inline int offset_gvec(int rank__) const
        {
            return gvec_distr_.offsets[rank__];
        }
        
        /// Return number of G-vector shells.
        inline int num_shells() const
        {
            return num_gvec_shells_;
        }
        
        /// Return corresponding G-vector for an index in the range [0, num_gvec).
        inline vector3d<int> gvec_by_full_index(int idx__) const
        {
            int j = idx__ & 0xFFF;
            int i = idx__ >> 12;
            assert(i < (int)z_columns_.size());
            assert(j < (int)z_columns_[i].z.size());
            int x = z_columns_[i].x;
            int y = z_columns_[i].y;
            int z = z_columns_[i].z[j];
            return fft_box_.gvec_by_coord(x, y, z);
        }

        // TODO: better names for the 3 functions and operator[] below

        /// Return corresponding G-vector for an index in the range [0, num_gvec).
        inline vector3d<int> operator[](int ig__) const
        {
            assert(ig__ >= 0 && ig__ < num_gvec_);
            return gvec_by_full_index(gvec_full_index_(ig__));
        }

        inline vector3d<double> gvec_shifted(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return (vector3d<double>(G[0], G[1], G[2]) + q_);
        }

        /// Return G-vector in Cartesian coordinates.
        inline vector3d<double> cart(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * vector3d<double>(G[0], G[1], G[2]);
        }

        /// Return G+q-vector in Cartesian coordinates.
        inline vector3d<double> cart_shifted(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + q_);
        }

        inline int shell(int ig__) const
        {
            return gvec_shell_(ig__);
        }

        inline double shell_len(int igs__) const
        {
            return gvec_shell_len_(igs__);
        }

        inline double gvec_len(int ig__) const
        {
            return gvec_shell_len_(gvec_shell_(ig__));
        }

        inline int index_g12(vector3d<int> const& g1__, vector3d<int> const& g2__) const
        {
            auto v = g1__ - g2__;
            int idx = index_by_gvec(v);
            assert(idx >= 0 && idx < num_gvec());
            return idx;
        }

        inline int index_g12_safe(int ig1__, int ig2__) const
        {
            STOP();
            return 0;
        }

        inline int index_by_gvec(vector3d<int> const& G__) const
        {
            return index_by_gvec_(G__[0], G__[1], G__[2]);
        }

        inline int num_z_cols() const
        {
            return static_cast<int>(z_columns_.size());
        }

        inline z_column_descriptor const& z_column(size_t idx__) const
        {
            return z_columns_[idx__];
        }

        inline bool reduced() const
        {
            return reduce_gvec_;
        }

        inline mdarray<int, 2>& z_columns_pos()
        {
            return z_columns_pos_;
        }

        inline mdarray<int, 2> const& z_columns_pos() const
        {
            return z_columns_pos_;
        }

        inline FFT3D_grid const& fft_box() const
        {
            return fft_box_;
        }
};

class Gvec_FFT_distribution
{
    private:

        Gvec const& gvec_;

        MPI_grid const& mpi_grid_fft_;

        block_data_descriptor zcol_fft_distr_;

        block_data_descriptor gvec_fft_distr_;

        block_data_descriptor gvec_slab_pile_;

        int num_gvec_fft_;

        int offset_gvec_fft_;

        mdarray<int, 1> zcol_offsets_;

        void build_fft_distr()
        {
            auto& fft_comm = mpi_grid_fft_.communicator(1 << 0);
            /* calculate distribution of G-vectors and z-columns for the FFT communicator */
            gvec_fft_distr_ = block_data_descriptor(fft_comm.size());
            zcol_fft_distr_ = block_data_descriptor(fft_comm.size());
            for (int rank = 0; rank < fft_comm.size(); rank++)
            {
                for (int i = 0; i < mpi_grid_fft_.dimension_size(1); i++)
                {
                    int r = rank * mpi_grid_fft_.dimension_size(1) + i;
                    gvec_fft_distr_.counts[rank] += gvec_.gvec_distr_.counts[r];
                    zcol_fft_distr_.counts[rank] += gvec_.zcol_distr_.counts[r];
                }
            }
            /* get offsets of z-columns */
            zcol_fft_distr_.calc_offsets();
            /* get offsets of G-vectors */
            gvec_fft_distr_.calc_offsets();
            /* get local number of G-vectors for a given rank */
            num_gvec_fft_ = gvec_fft_distr_.counts[fft_comm.rank()];
            /* get offset of G-vectors for a given rank */
            offset_gvec_fft_ = gvec_fft_distr_.offsets[fft_comm.rank()];
        }

        void calc_offsets()
        {
            auto& fft_comm = mpi_grid_fft_.communicator(1 << 0);

            /* calculate offsets of z-columns inside each local buffer of PW coefficients */
            int num_zcol_local = zcol_fft_distr_.counts[fft_comm.rank()];
            zcol_offsets_ = mdarray<int, 1>(num_zcol_local);
            int offs = 0;
            for (int i = 0; i < num_zcol_local; i++)
            {
                /* global index of z-column */
                int icol = zcol_fft_distr_.offsets[fft_comm.rank()] + i;
                zcol_offsets_(i) = offs;
                offs += static_cast<int>(gvec_.z_column(icol).z.size());
            }
            assert(offs == gvec_fft_distr_.counts[fft_comm.rank()]);
        }

        void pile_gvec()
        {
            /* build a table of {offset, count} values for G-vectors in the swapped wfs;
             * we are preparing to swap wave-functions from a default slab distribution to a FFT-friendly distribution 
             * +==============+      +----+----+----+
             * |    :    :    |      I    I    I    I
             * +==============+      I....I....I....I
             * |    :    :    |  ->  I    I    I    I
             * +==============+      I....I....I....I
             * |    :    :    |      I    I    I    I
             * +==============+      +----+----+----+
             *
             * i.e. we will make G-vector slabs more fat (pile-of-slabs) and at the same time reshulffle wave-functions
             * between columns of the 2D MPI grid */
            auto& fft_comm = mpi_grid_fft_.communicator(1 << 0);
            auto& comm_col = mpi_grid_fft_.communicator(1 << 1);
            int rank_row = fft_comm.rank();
            gvec_slab_pile_ = block_data_descriptor(comm_col.size());
            for (int i = 0; i < comm_col.size(); i++)
                gvec_slab_pile_.counts[i] = gvec_.num_gvec(rank_row * comm_col.size() + i);
            gvec_slab_pile_.calc_offsets();

            assert(gvec_slab_pile_.offsets.back() + gvec_slab_pile_.counts.back() == gvec_fft_distr_.counts[rank_row]);
        }
        
    public:

        Gvec_FFT_distribution(Gvec const& gvec__, MPI_grid const& mpi_grid_fft__)
            : gvec_(gvec__),
              mpi_grid_fft_(mpi_grid_fft__)
        {
            if (mpi_grid_fft__.size() != gvec__.num_ranks_)
            {
                TERMINATE("inconsistent number of ranks");
            }

            build_fft_distr();

            calc_offsets();

            pile_gvec();
        }

        Gvec_FFT_distribution(Gvec const& gvec__, Communicator const& comm__)
            : gvec_(gvec__),
              mpi_grid_fft_(comm__)
        {
            if (comm__.size() != gvec__.num_ranks_)
            {
                TERMINATE("inconsistent number of ranks");
            }

            build_fft_distr();

            calc_offsets();
        }

        inline int num_gvec_fft() const
        {
            return num_gvec_fft_;
        }

        inline int offset_gvec_fft() const
        {
            return offset_gvec_fft_;
        }

        inline block_data_descriptor const& zcol_fft_distr() const
        {
            return zcol_fft_distr_;
        }

        inline block_data_descriptor const& gvec_slab_pile() const
        {
            assert(gvec_slab_pile_.num_ranks > 0);
            return gvec_slab_pile_;
        }

        inline int zcol_offset(int icol_loc) const
        {
            return zcol_offsets_(icol_loc);
        }

        Gvec const& gvec() const
        {
            return gvec_;
        }

        MPI_grid const& mpi_grid_fft() const
        {
            return mpi_grid_fft_;
        }
};

};

#endif

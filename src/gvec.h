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

namespace sirius {

/* forward declaration */
class Gvec;

class Gvec_partition
{
    friend class Gvec;

    private:
        Gvec const* gvec_{nullptr};

        /// Communicator for FFT.
        Communicator const* fft_comm_{nullptr};

        /// Distribution of G-vectors for FFT.
        block_data_descriptor gvec_distr_fft_;

        /// Distribution of z-columns for FFT.
        block_data_descriptor zcol_distr_fft_;

        /// Distribution of G-vectors inside FFT slab.
        block_data_descriptor gvec_fft_slab_;

        mdarray<int, 1> zcol_offs_;
        
        inline void build_fft_distr();

        /// Calculate offsets of z-columns inside each local buffer of PW coefficients.
        inline void calc_offsets();
        
        inline void pile_gvec();

    public:
        Gvec_partition(Gvec const& gvec__, Communicator const& comm__)
            : gvec_(&gvec__),
              fft_comm_(&comm__)
        {
            build_fft_distr();
            calc_offsets();
            pile_gvec();
        }

        inline int gvec_count_fft() const
        {
            return gvec_distr_fft_.counts[fft_comm_->rank()];
        }

        inline int gvec_offset_fft() const
        {
            return gvec_distr_fft_.offsets[fft_comm_->rank()];
        }

        inline block_data_descriptor const& zcol_distr_fft() const
        {
            return zcol_distr_fft_;
        }

        inline block_data_descriptor const& gvec_fft_slab() const
        {
            return gvec_fft_slab_;
        }

        inline int zcol_offs(int icol__) const
        {
            return zcol_offs_(icol__);
        }

        inline Gvec const& gvec() const
        {
            return *gvec_;
        }

        inline int num_gvec() const;

        inline int num_zcol() const;

        inline z_column_descriptor const& zcol(size_t idx__) const;

        inline bool reduced() const;
};

/// Store list of G-vectors for FFTs and G+k basis functions.
class Gvec
{
    friend class Gvec_partition;

    private:

        /// k-vector of G+k.
        vector3d<double> vk_;
        
        /// Reciprocal lattice vectors.
        matrix3d<double> lattice_vectors_;
        
        /// Indicates that G-vectors are reduced by inversion symmetry.
        bool reduce_gvec_;
        
        /// Number of ranks for fine-grained distribution.
        int num_ranks_;
        
        /// Total number of G-vectors.
        int num_gvec_;

        /// Mapping between G-vector index [0:num_gvec_) and a full index.
        /** Full index is used to store x,y,z coordinates in a packed form in a single integer number. */
        mdarray<int, 1> gvec_full_index_;
    
        /// Index of the shell to which the given G-vector belongs.
        mdarray<int, 1> gvec_shell_;
        
        int num_gvec_shells_;

        mdarray<double, 1> gvec_shell_len_;

        mdarray<int, 3> gvec_index_by_xy_;

        /// Global list of non-zero z-columns.
        std::vector<z_column_descriptor> z_columns_;

        /// Fine-grained distribution of G-vectors.
        block_data_descriptor gvec_distr_;

        /// Fine-grained distribution of z-columns.
        block_data_descriptor zcol_distr_;
        
        /// Default G-vector partitioning.
        std::unique_ptr<Gvec_partition> gvec_partition_;

        /* copy constructor is forbidden */
        Gvec(Gvec const& src__) = delete;
        /* copy assigment operator is forbidden */
        Gvec& operator=(Gvec const& src__) = delete;
        /* move constructor is forbidden */
        Gvec(Gvec&& src__) = delete;

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
            return vector3d<int>(x, y, z);
        }

    public:
        
        /// Default constructor.
        Gvec()
        {
        }
        
        /// Constructor.
        Gvec(vector3d<double>        vk__,
             matrix3d<double>        M__,
             double                  Gmax__,
             FFT3D_grid const&       fft_box__,
             int                     num_ranks__,
             Communicator const&     fft_comm__,
             bool                    reduce_gvec__)
            : vk_(vk__),
              lattice_vectors_(M__),
              reduce_gvec_(reduce_gvec__),
              num_ranks_(num_ranks__)
        {
            mdarray<int, 2> non_zero_columns(fft_box__.limits(0), fft_box__.limits(1));
            non_zero_columns.zero();

            num_gvec_ = 0;
            for (int i = fft_box__.limits(0).first; i <= fft_box__.limits(0).second; i++) {
                for (int j = fft_box__.limits(1).first; j <= fft_box__.limits(1).second; j++) {
                    std::vector<int> zcol;
                    
                    /* in general case take z in [0, Nz) */ 
                    int zmax = fft_box__.size(2) - 1;
                    /* in case of G-vector reduction take z in [0, Nz/2] for {x=0,y=0} stick */
                    if (reduce_gvec_ && !i && !j) {
                        zmax = fft_box__.limits(2).second;
                    }
                    /* loop over z-coordinates of FFT grid */ 
                    for (int iz = 0; iz <= zmax; iz++) {
                        /* get z-coordinate of G-vector */
                        int k = fft_box__.gvec_by_coord(iz, 2);
                        /* take G+k */
                        auto vgk = lattice_vectors_ * (vector3d<double>(i, j, k) + vk__);
                        /* add z-coordinate of G-vector to the list */
                        if (vgk.length() <= Gmax__) {
                            zcol.push_back(k);
                        }
                    }
                    
                    if (zcol.size() && !non_zero_columns(i, j)) {
                        z_columns_.push_back(z_column_descriptor(i, j, zcol));
                        num_gvec_ += static_cast<int>(zcol.size());

                        non_zero_columns(i, j) = 1;
                        if (reduce_gvec__) {
                            non_zero_columns(-i, -j) = 1;
                        }
                    }
                }
            }
            
            /* put column with {x, y} = {0, 0} to the beginning */
            for (size_t i = 0; i < z_columns_.size(); i++) {
                if (z_columns_[i].x == 0 && z_columns_[i].y == 0) {
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
            for (size_t i = 0; i < z_columns_.size(); i++) {
                /* initialize the list of ranks to 0,1,2,... */
                if (ranks.empty()) {
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
            for (int rank = 0; rank < num_ranks__; rank++) {
                z_columns_.insert(z_columns_.end(), zcols_local[rank].begin(), zcols_local[rank].end());
            }

            gvec_index_by_xy_ = mdarray<int, 3>(2, fft_box__.limits(0), fft_box__.limits(1), memory_t::host, "Gvec.gvec_index_by_xy_");
            std::fill(gvec_index_by_xy_.at<CPU>(), gvec_index_by_xy_.at<CPU>() + gvec_index_by_xy_.size(), -1);
            
            /* build the full G-vector index and reverse mapping */
            gvec_full_index_ = mdarray<int, 1>(num_gvec_);
            int ig{0};
            for (size_t i = 0; i < z_columns_.size(); i++) {
                /* starting G-vector index for a z-stick */
                gvec_index_by_xy_(0, z_columns_[i].x, z_columns_[i].y) = ig;
                /* size of a z-stick */
                gvec_index_by_xy_(1, z_columns_[i].x, z_columns_[i].y) = static_cast<int>(z_columns_[i].z.size());
                for (size_t j = 0; j < z_columns_[i].z.size(); j++) {
                    gvec_full_index_[ig++] = static_cast<int>((i << 12) + j);
                }
            }
            
            /* first G-vector must be (0, 0, 0); never reomove this check!!! */
            auto g0 = gvec_by_full_index(gvec_full_index_(0));
            if (g0[0] || g0[1] || g0[2]) {
                TERMINATE("first G-vector is not zero");
            }
        
            /* find G-shells */
            std::map<size_t, std::vector<int> > gsh;
            for (int ig = 0; ig < num_gvec_; ig++) {
                /* take G+k */
                auto gk = gkvec_cart(ig);
                /* make some reasonable roundoff */
                size_t len = size_t(gk.length() * 1e10);

                if (!gsh.count(len)) {
                    gsh[len] = std::vector<int>();
                }
                gsh[len].push_back(ig);
            }
            num_gvec_shells_ = static_cast<int>(gsh.size());
            gvec_shell_ = mdarray<int, 1>(num_gvec_);
            gvec_shell_len_ = mdarray<double, 1>(num_gvec_shells_);
            
            int n{0};
            for (auto it = gsh.begin(); it != gsh.end(); it++) {
                gvec_shell_len_(n) = static_cast<double>(it->first) * 1e-10;
                for (int ig: it->second) {
                    gvec_shell_(ig) = n;
                }
                n++;
            }
            
            /* create default partition for G-vectors */
            gvec_partition_ = std::unique_ptr<Gvec_partition>(new Gvec_partition(*this, fft_comm__));
        }

        /// Move assigment operator.
        Gvec& operator=(Gvec&& src__)
        {
            if (this != &src__) {
                vk_               = src__.vk_;
                lattice_vectors_  = src__.lattice_vectors_;
                reduce_gvec_      = src__.reduce_gvec_;
                num_ranks_        = src__.num_ranks_;
                num_gvec_         = src__.num_gvec_;
                gvec_full_index_  = std::move(src__.gvec_full_index_);
                gvec_shell_       = std::move(src__.gvec_shell_);
                num_gvec_shells_  = src__.num_gvec_shells_;
                gvec_shell_len_   = std::move(src__.gvec_shell_len_);
                gvec_index_by_xy_ = std::move(src__.gvec_index_by_xy_);
                z_columns_        = std::move(src__.z_columns_);
                gvec_distr_       = std::move(src__.gvec_distr_);
                zcol_distr_       = std::move(src__.zcol_distr_);
                gvec_partition_   = std::move(src__.gvec_partition_);
                gvec_partition_->gvec_ = this;
            }
            return *this;
        }

        /// Return the total number of G-vectors within the cutoff.
        inline int num_gvec() const
        {
            return num_gvec_;
        }

        /// Number of G-vectors for a fine-grained distribution.
        inline int gvec_count(int rank__) const
        {
            assert(rank__ < num_ranks_);
            return gvec_distr_.counts[rank__];
        }

        /// Offset (in the global index) of G-vectors for a fine-grained distribution.
        inline int gvec_offset(int rank__) const
        {
            assert(rank__ < num_ranks_);
            return gvec_distr_.offsets[rank__];
        }

        /// Return number of G-vector shells.
        inline int num_shells() const
        {
            return num_gvec_shells_;
        }

        /// Return G vector in fractional coordinates.
        inline vector3d<int> gvec(int ig__) const
        {
            return gvec_by_full_index(gvec_full_index_(ig__));
        }

        /// Return G+k vector in fractional coordinates.
        inline vector3d<double> gkvec(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return (vector3d<double>(G[0], G[1], G[2]) + vk_);
        }

        /// Return G vector in Cartesian coordinates.
        inline vector3d<double> gvec_cart(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * vector3d<double>(G[0], G[1], G[2]);
        }

        /// Return G+k vector in Cartesian coordinates.
        inline vector3d<double> gkvec_cart(int ig__) const
        {
            auto G = gvec_by_full_index(gvec_full_index_(ig__));
            return lattice_vectors_ * (vector3d<double>(G[0], G[1], G[2]) + vk_);
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
            if (reduced() && G__[0] == 0 && G__[1] == 0 && G__[2] < 0) {
                return -1;
            }
            int ig0 = gvec_index_by_xy_(0, G__[0], G__[1]);
            if (ig0 == -1) {
                return -1;
            }
            int offs = (G__[2] >= 0) ? G__[2] : G__[2] + gvec_index_by_xy_(1, G__[0], G__[1]);
            int ig = ig0 + offs;
            assert(ig < num_gvec());
            return ig;
        }

        inline bool reduced() const
        {
            return reduce_gvec_;
        }

        inline int num_zcol() const
        {
            return static_cast<int>(z_columns_.size());
        }

        inline z_column_descriptor const& zcol(size_t idx__) const
        {
            return z_columns_[idx__];
        }

        inline Gvec_partition const& partition() const
        {
            return *gvec_partition_;
        }
};

inline void Gvec_partition::build_fft_distr()
{
    /* calculate distribution of G-vectors and z-columns for the FFT communicator */
    gvec_distr_fft_ = block_data_descriptor(fft_comm_->size());
    zcol_distr_fft_ = block_data_descriptor(fft_comm_->size());

    int nrc = gvec_->num_ranks_ / fft_comm_->size();

    if (gvec_->num_ranks_ != nrc * fft_comm_->size()) {
        TERMINATE("wrong number of MPI ranks");
    }

    for (int rank = 0; rank < fft_comm_->size(); rank++) {
        for (int i = 0; i < nrc; i++) {
            /* fine-grained rank */
            int r = rank * nrc + i;
            gvec_distr_fft_.counts[rank] += gvec_->gvec_distr_.counts[r];
            zcol_distr_fft_.counts[rank] += gvec_->zcol_distr_.counts[r];
        }
    }
    /* get offsets of z-columns */
    zcol_distr_fft_.calc_offsets();
    /* get offsets of G-vectors */
    gvec_distr_fft_.calc_offsets();
}

inline void Gvec_partition::calc_offsets()
{
    zcol_offs_ = mdarray<int, 1>(gvec_->num_zcol(), memory_t::host, "Gvec_partition.zcol_offs_");
    for (int rank = 0; rank < fft_comm_->size(); rank++) {
        int offs{0};
        /* loop over local number of z-columns */
        for (int i = 0; i < zcol_distr_fft_.counts[rank]; i++) {
            /* global index of z-column */
            int icol = zcol_distr_fft_.offsets[rank] + i;
            zcol_offs_[icol] = offs;
            offs += static_cast<int>(gvec_->z_columns_[icol].z.size());
        }
        assert(offs == gvec_distr_fft_.counts[rank]);
    }
}

inline void Gvec_partition::pile_gvec()
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
    int rank_row = fft_comm_->rank();

    int nrc = gvec_->num_ranks_ / fft_comm_->size();
    if (gvec_->num_ranks_ != nrc * fft_comm_->size()) {
        TERMINATE("wrong number of MPI ranks");
    }

    gvec_fft_slab_ = block_data_descriptor(nrc);
    for (int i = 0; i < nrc; i++) {
        gvec_fft_slab_.counts[i] = gvec_->gvec_count(rank_row * nrc + i);
    }
    gvec_fft_slab_.calc_offsets();

    assert(gvec_fft_slab_.offsets.back() + gvec_fft_slab_.counts.back() == gvec_distr_fft_.counts[rank_row]);
}

inline int Gvec_partition::num_gvec() const
{
    return gvec_->num_gvec();
}

inline int Gvec_partition::num_zcol() const
{
    return gvec_->num_zcol();
}

inline z_column_descriptor const& Gvec_partition::zcol(size_t idx__) const
{
    return gvec_->zcol(idx__);
}

inline bool Gvec_partition::reduced() const
{
    return gvec_->reduced();
}

};

#endif

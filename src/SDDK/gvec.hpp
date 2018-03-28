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

/** \file gvec.hpp
 *
 *  \brief Declaration and implementation of Gvec class.
 */

#ifndef __GVEC_HPP__
#define __GVEC_HPP__

#include <numeric>
#include <map>
#include <iostream>
#include "fft3d_grid.hpp"
#include "geometry3d.hpp"

using namespace geometry3d;

namespace sddk {

/// Descriptor of the z-column (x,y fixed, z varying) of the G-vectors.
/** Sphere of G-vectors within a given plane-wave cutoff is represented as a set of z-columns with different lengths. */
struct z_column_descriptor
{
    /// X-coordinate (can be negative and positive).
    int x;
    /// Y-coordinate (can be negative and positive).
    int y;
    /// List of the Z-coordinates of the column.
    std::vector<int> z;
    /// Constructor.
    z_column_descriptor(int x__, int y__, std::vector<int> z__)
        : x(x__)
        , y(y__)
        , z(z__)
    {
    }
};

/// A set of G-vectors for FFTs and G+k basis functions.
/** Current implemntation supports up to 2^12 (4096) z-dimension of the FFT grid and 2^20 (1048576) number of
 *  z-columns. */
class Gvec
{
  private:
    /// k-vector of G+k.
    vector3d<double> vk_;

    /// Cutoff for |G+k| vectors.
    double Gmax_;

    /// Reciprocal lattice vectors.
    matrix3d<double> lattice_vectors_;

    /// Total communicator which is used to distribute G or G+k vectors.
    Communicator const& comm_;

    /// Indicates that G-vectors are reduced by inversion symmetry.
    bool reduce_gvec_;

    /// True if this a list of G-vectors without k-point shift.
    bool bare_gvec_{true};

    /// Total number of G-vectors.
    int num_gvec_;

    /// Mapping between G-vector index [0:num_gvec_) and a full index.
    /** Full index is used to store x,y,z coordinates in a packed form in a single integer number.
     *  The index is equal to ((i << 12) + j) where i is the global index of z_column and j is the
     *  index of G-vector z-coordinate in the column i. This is a global array: each MPI rank stores exactly the
     *  same copy of the gvec_full_index_.
     *
     *  Limitations: size of z-dimension of FFT grid: 4096, number of z-columns: 1048576
     */
    mdarray<uint32_t, 1> gvec_full_index_;

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

    /// Set of G-vectors on which the current G-vector distribution can be based.
    /** This can be used to establish a local mapping between coarse and fine G-vector sets
     *  without MPI communication. */
    Gvec const* gvec_base_{nullptr};

    /// Mapping between current and base G-vector sets.
    /** This mapping allows for a local-to-local copy of PW coefficients without any MPI communication.

        Example:
        \code{.cpp}
        // Copy from a coarse G-vector set.
        for (int igloc = 0; igloc < ctx_.gvec_coarse().count(); igloc++) {
            rho_vec_[j]->f_pw_local(ctx_.gvec().gvec_base_mapping(igloc)) = rho_mag_coarse_[j]->f_pw_local(igloc);
        }
        \endcode
    */
    mdarray<int, 1> gvec_base_mapping_;

    /* copy constructor is forbidden */
    Gvec(Gvec const& src__) = delete;
    /* copy assigment operator is forbidden */
    Gvec& operator=(Gvec const& src__) = delete;
    /* move constructor is forbidden */
    Gvec(Gvec&& src__) = delete;

    /// Return corresponding G-vector for an index in the range [0, num_gvec).
    inline vector3d<int> gvec_by_full_index(uint32_t idx__) const
    {
        /* index of the z coordinate of G-vector: first 12 bits */
        uint32_t j = idx__ & 0xFFF;
        /* index of z-column: last 20 bits */
        uint32_t i = idx__ >> 12;
        assert(i < (uint32_t)z_columns_.size());
        assert(j < (uint32_t)z_columns_[i].z.size());
        int x = z_columns_[i].x;
        int y = z_columns_[i].y;
        int z = z_columns_[i].z[j];
        return vector3d<int>(x, y, z);
    }

    /// Find z-columns of G-vectors inside a sphere with Gmax radius.
    /** This function also computes the total number of G-vectors. */
    inline void find_z_columns(double Gmax__, FFT3D_grid const& fft_box__)
    {
        mdarray<int, 2> non_zero_columns(fft_box__.limits(0), fft_box__.limits(1));
        non_zero_columns.zero();

        num_gvec_ = 0;

        auto add_new_column = [&](int i, int j)
        {
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
                auto vgk = lattice_vectors_ * (vector3d<double>(i, j, k) + vk_);
                /* add z-coordinate of G-vector to the list */
                if (vgk.length() <= Gmax__) {
                    zcol.push_back(k);
                }
            }

            /* add column to the list */
            if (zcol.size() && !non_zero_columns(i, j)) {
                z_columns_.push_back(z_column_descriptor(i, j, zcol));
                num_gvec_ += static_cast<int>(zcol.size());

                non_zero_columns(i, j) = 1;
                if (reduce_gvec_) {
                    non_zero_columns(-i, -j) = 1;
                }
            }
        };

        /* copy column order from previous G-vector set */
        if (gvec_base_) {
            for (int icol = 0; icol < gvec_base_->num_zcol(); icol++) {
                int i = gvec_base_->zcol(icol).x;
                int j = gvec_base_->zcol(icol).y;
                add_new_column(i, j);
            }
        }

        for (int i = fft_box__.limits(0).first; i <= fft_box__.limits(0).second; i++) {
            for (int j = fft_box__.limits(1).first; j <= fft_box__.limits(1).second; j++) {
                add_new_column(i, j);
            }
        }

        if (!gvec_base_) {
            /* put column with {x, y} = {0, 0} to the beginning */
            for (size_t i = 0; i < z_columns_.size(); i++) {
                if (z_columns_[i].x == 0 && z_columns_[i].y == 0) {
                    std::swap(z_columns_[i], z_columns_[0]);
                    break;
                }
            }
        }

        /* sort z-columns starting from the second or skip num_zcol of base distribution */
        int n = (gvec_base_) ? gvec_base_->num_zcol() : 1;
        std::sort(z_columns_.begin() + n, z_columns_.end(),
                  [](z_column_descriptor const& a, z_column_descriptor const& b) { return a.z.size() > b.z.size(); });
    }

    /// Distribute z-columns between MPI ranks.
    inline void distribute_z_columns()
    {
        gvec_distr_ = block_data_descriptor(comm().size());
        zcol_distr_ = block_data_descriptor(comm().size());
        /* local number of z-columns for each rank */
        std::vector<std::vector<z_column_descriptor>> zcols_local(comm().size());

        /* use already existing distribution of base G-vector set */
        if (gvec_base_) {
            for (int rank = 0; rank < comm().size(); rank++) {
                for (int i = 0; i < gvec_base_->zcol_count(rank); i++) {
                    int icol = gvec_base_->zcol_offset(rank) + i;
                    /* assign column to the found rank */
                    zcols_local[rank].push_back(z_columns_[icol]);
                    /* count local number of z-columns */
                    zcol_distr_.counts[rank] += 1;
                    /* count local number of G-vectors */
                    gvec_distr_.counts[rank] += static_cast<int>(z_columns_[icol].z.size());
                }
            }
        }

        int n = (gvec_base_) ? gvec_base_->num_zcol() : 0;

        std::vector<int> ranks;
        for (int i = n; i < static_cast<int>(z_columns_.size()); i++) {
            /* initialize the list of ranks to 0,1,2,... */
            if (ranks.empty()) {
                ranks.resize(comm().size());
                std::iota(ranks.begin(), ranks.end(), 0);
            }
            /* find rank with minimum number of G-vectors */
            auto rank_with_min_gvec = std::min_element(ranks.begin(), ranks.end(), [this](const int& a, const int& b) {
                return gvec_distr_.counts[a] < gvec_distr_.counts[b];
            });

            /* assign column to the found rank */
            zcols_local[*rank_with_min_gvec].push_back(z_columns_[i]);
            /* count local number of z-columns */
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
        for (int rank = 0; rank < comm().size(); rank++) {
            z_columns_.insert(z_columns_.end(), zcols_local[rank].begin(), zcols_local[rank].end());
        }

        /* sanity check */
        int ng{0};
        for (int rank = 0; rank < comm().size(); rank++) {
            ng += gvec_distr_.counts[rank];
        }
        if (ng != num_gvec_) {
            TERMINATE("wrong number of G-vectors");
        }
    }

    /// Find a list of G-vector shells.
    /** G or G+k vectors belonging to the same shell have the same length */
    inline void find_gvec_shells()
    {
        PROFILE("sddk::Gvec::find_gvec_shells");

        /* list of pairs (length, index of G-vector) */
        std::vector<std::pair<size_t, int>> tmp(num_gvec_);
        #pragma omp parallel for schedule(static)
        for (int ig = 0; ig < num_gvec(); ig++) {
            /* take G+k */
            auto gk = gkvec_cart(ig);
            /* make some reasonable roundoff */
            size_t len = size_t(gk.length() * 1e8);
            tmp[ig]    = std::pair<size_t, int>(len, ig);
        }
        /* sort by first element in pair (length) */
        std::sort(tmp.begin(), tmp.end());

        gvec_shell_ = mdarray<int, 1>(num_gvec_);
        /* index of the first shell */
        gvec_shell_(tmp[0].second) = 0;
        num_gvec_shells_           = 1;
        /* temporary vector to store G-shell radius */
        std::vector<double> tmp_len;
        /* radius of the first shell */
        tmp_len.push_back(static_cast<double>(tmp[0].first) * 1e-8);
        for (int ig = 1; ig < num_gvec_; ig++) {
            /* if this G+k-vector has a different length */
            if (tmp[ig].first != tmp[ig - 1].first) {
                /* increment number of shells */
                num_gvec_shells_++;
                /* save the radius of the new shell */
                tmp_len.push_back(static_cast<double>(tmp[ig].first) * 1e-8);
            }
            /* assign the index of the current shell */
            gvec_shell_(tmp[ig].second) = num_gvec_shells_ - 1;
        }
        gvec_shell_len_ = mdarray<double, 1>(num_gvec_shells_);
        std::copy(tmp_len.begin(), tmp_len.end(), gvec_shell_len_.at<CPU>());
    }

    /// Initialize everything.
    void init()
    {
        PROFILE("sddk::Gvec::init");

        auto fft_grid = FFT3D_grid(find_translations(Gmax_, lattice_vectors_) + vector3d<int>({2, 2, 2}));

        find_z_columns(Gmax_, fft_grid);

        distribute_z_columns();

        gvec_index_by_xy_ = mdarray<int, 3>(2, fft_grid.limits(0), fft_grid.limits(1), memory_t::host, "Gvec.gvec_index_by_xy_");
        std::fill(gvec_index_by_xy_.at<CPU>(), gvec_index_by_xy_.at<CPU>() + gvec_index_by_xy_.size(), -1);

        /* build the full G-vector index and reverse mapping */
        gvec_full_index_ = mdarray<uint32_t, 1>(num_gvec_);
        int ig{0};
        for (size_t i = 0; i < z_columns_.size(); i++) {
            /* starting G-vector index for a z-stick */
            gvec_index_by_xy_(0, z_columns_[i].x, z_columns_[i].y) = ig;
            /* pack size of a z-stick and column index in one number */
            gvec_index_by_xy_(1, z_columns_[i].x, z_columns_[i].y) =
                static_cast<int>((z_columns_[i].z.size() << 20) + i);
            for (size_t j = 0; j < z_columns_[i].z.size(); j++) {
                gvec_full_index_[ig++] = static_cast<uint32_t>((i << 12) + j);
            }
        }
        if (ig != num_gvec_) {
            TERMINATE("wrong G-vector count");
        }
        for (int ig = 0; ig < num_gvec_; ig++) {
            auto gv = gvec(ig);
            if (index_by_gvec(gv) != ig) {
                std::stringstream s;
                s << "wrong G-vector index: ig=" << ig << " gv=" << gv << " index_by_gvec(gv)=" << index_by_gvec(gv);
                TERMINATE(s);
            }
        }

        /* first G-vector must be (0, 0, 0); never reomove this check!!! */
        auto g0 = gvec_by_full_index(gvec_full_index_(0));
        if (g0[0] || g0[1] || g0[2]) {
            TERMINATE("first G-vector is not zero");
        }

        find_gvec_shells();

        if (gvec_base_) {
            /* the size of the mapping is equal to the local number of G-vectors in the base set */
            gvec_base_mapping_ = mdarray<int, 1>(gvec_base_->count());
            /* loop over local G-vectors of a base set */
            for (int igloc = 0; igloc < gvec_base_->count(); igloc++) {
                /* G-vector in lattice coordinates */
                auto G = gvec_base_->gvec(gvec_base_->offset() + igloc);
                /* global index of G-vector in the current set */
                int ig = index_by_gvec(G);
                /* the same MPI rank must store this G-vector */
                ig -= offset();
                if (ig >= 0 && ig < count()) {
                    gvec_base_mapping_(igloc) = ig;
                } else {
                    std::stringstream s;
                    s << "local G-vector index is not found" << std::endl
                      << " G-vector: " << G << std::endl
                      << " G-vector index in base distribution : " << gvec_base_->offset() + igloc << std::endl
                      << " G-vector index in base distribution (by G-vector): " << gvec_base_->index_by_gvec(G) << std::endl
                      << " G-vector index in new distribution : " << index_by_gvec(G) << std::endl
                      << " offset in G-vector index for this rank: " << offset() << std::endl
                      << " local number of G-vectors for this rank: " << count();
                    TERMINATE(s);
                }
            }
        }

        // TODO: add a check for gvec_base (there is already a test for this).
    }

    Gvec& operator=(Gvec&& src__) = delete;

  public:
    /// Constructor for G+k vectors.
    /** \param [in] vk          K-point vector of G+k
     *  \param [in] M           Reciprocal lattice vecotors in comumn order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
     *  \param [in] comm_fft    FFT communicator
     *  \param [in] reduce_gvec True if G-vectors need to be reduced by inversion symmetry.
     */
    Gvec(vector3d<double> vk__, matrix3d<double> M__, double Gmax__, Communicator const& comm__, bool reduce_gvec__)
        : vk_(vk__)
        , Gmax_(Gmax__)
        , lattice_vectors_(M__)
        , comm_(comm__)
        , reduce_gvec_(reduce_gvec__)
        , bare_gvec_(false)
    {
        init();
    }

    /// Constructor for G-vectors.
    Gvec(matrix3d<double> M__, double Gmax__, Communicator const& comm__, bool reduce_gvec__)
        : vk_({0, 0, 0})
        , Gmax_(Gmax__)
        , lattice_vectors_(M__)
        , comm_(comm__)
        , reduce_gvec_(reduce_gvec__)
    {
        init();
    }

    /// Constructor for G-vector distribution based on a previous set.
    /** Previous set of G-vectors must be a subset of the current set. */
    Gvec(double Gmax__, Gvec const& gvec_base__)
        : vk_({0, 0, 0})
        , Gmax_(Gmax__)
        , lattice_vectors_(gvec_base__.lattice_vectors())
        , comm_(gvec_base__.comm())
        , reduce_gvec_(gvec_base__.reduced())
        , gvec_base_(&gvec_base__)
    {
        init();
    }

    /// Constructor for G-vectors with mpi_comm_self()
    Gvec(matrix3d<double> M__, double Gmax__, bool reduce_gvec__)
        : vk_({0, 0, 0})
        , Gmax_(Gmax__)
        , lattice_vectors_(M__)
        , comm_(mpi_comm_self())
        , reduce_gvec_(reduce_gvec__)
    {
        init();
    }


    vector3d<double> const& vk() const
    {
        return vk_;
    }

    Communicator const& comm() const
    {
        return comm_;
    }

    inline matrix3d<double> const& lattice_vectors() const
    {
        return lattice_vectors_;
    }

    /// Return the total number of G-vectors within the cutoff.
    inline int num_gvec() const
    {
        return num_gvec_;
    }

    /// Number of z-columns for a fine-grained distribution.
    inline int zcol_count(int rank__) const
    {
        assert(rank__ < comm().size());
        return zcol_distr_.counts[rank__];
    }

    inline int zcol_offset(int rank__) const
    {
        assert(rank__ < comm().size());
        return zcol_distr_.offsets[rank__];
    }

    /// Number of G-vectors for a fine-grained distribution.
    inline int gvec_count(int rank__) const
    {
        assert(rank__ < comm().size());
        return gvec_distr_.counts[rank__];
    }

    /// Number of G-vectors for a fine-grained distribution for the current MPI rank.
    /** The \em count and \em offset are borrowed from the MPI terminology for data distribution. */
    inline int count() const
    {
        return gvec_distr_.counts[comm().rank()];
    }

    /// Offset (in the global index) of G-vectors for a fine-grained distribution.
    inline int gvec_offset(int rank__) const
    {
        assert(rank__ < comm().size());
        return gvec_distr_.offsets[rank__];
    }

    /// Offset (in the global index) of G-vectors for a fine-grained distribution for a current MPI rank.
    /** The \em count and \em offset are borrowed from the MPI terminology for data distribution. */
    inline int offset() const
    {
        return gvec_distr_.offsets[comm().rank()];
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
        auto v  = g1__ - g2__;
        int idx = index_by_gvec(v);
        assert(idx >= 0);
        assert(idx < num_gvec());
        return idx;
    }

    inline std::pair<int, bool> index_g12_safe(vector3d<int> const& g1__, vector3d<int> const& g2__) const
    {
        auto v  = g1__ - g2__;
        int idx = index_by_gvec(v);
        bool conj{false};
        if (idx < 0) {
            idx = index_by_gvec(v * (-1));
            conj = true;
        }
        if (idx < 0 || idx >= num_gvec()) {
            std::stringstream s;
            s << "wrong index of G-G' vector" << std::endl
              << "  G: " << g1__ << std::endl
              << "  G': " << g2__ << std::endl
              << "  G - G': " << v << std::endl
              << " idx: " << idx;
            TERMINATE(s);
        }
        return std::make_pair(idx, conj);
    }

    inline int index_g12_safe(int ig1__, int ig2__) const
    {
        STOP();
        return 0;
    }

    /// Return a global G-vector index in the range [0, num_gvec) by the G-vector.
    /** The information about a G-vector index is encoded by two numbers: a starting index for the
     *  column of G-vectors and column's size. Depending on the geometry of the reciprocal lattice,
     *  z-columns may have only negative, only positive or both negative and positive frequencies for
     *  a given x and y. This information is used to compute the offset which is added to the starting index
     *  in order to get a full G-vector index. */
    inline int index_by_gvec(vector3d<int> const& G__) const
    {
        /* reduced G-vector set does not have negative z for x=y=0 */
        if (reduced() && G__[0] == 0 && G__[1] == 0 && G__[2] < 0) {
            return -1;
        }
        int ig0 = gvec_index_by_xy_(0, G__[0], G__[1]);
        if (ig0 == -1) {
            return -1;
        }
        /* index of the column */
        int icol = gvec_index_by_xy_(1, G__[0], G__[1]) & 0xFFFFF;
        /* size of the column */
        int col_size = gvec_index_by_xy_(1, G__[0], G__[1]) >> 20;

        /* three possible options for the z-column location

              frequency                ... -4, -3, -2, -1, 0, 1, 2, 3, 4 ...
           -----------------------------------------------------------------------------
              G-vec ordering
           #1 (all negative)           ___  0   1   2   3 __________________
           #2 (negative and positive)  ____________ 3   4  0  1  2 _________
           #3 (all positive)           _____________________  0  1  2  3 ___

           Remember how FFT frequencies are stored: firs positive frequences, then negative in the reverse order

           subtract first z-coordinate in column from the current z-coordinate of G-vector: in case #1 or #3 this
           already gives a proper offset, in case #2 storage of FFT frequencies must be taken into account
        */
        int z0 = G__[2] - z_columns_[icol].z[0];
        /* calculate proper offset */
        int offs = (z0 >= 0) ? z0 : z0 + col_size;
        /* full index */
        int ig = ig0 + offs;
        assert(ig >= 0 && ig < num_gvec());
        return ig;
    }

    inline bool reduced() const
    {
        return reduce_gvec_;
    }

    inline bool bare() const
    {
        return bare_gvec_;
    }

    inline int num_zcol() const
    {
        return static_cast<int>(z_columns_.size());
    }

    inline z_column_descriptor const& zcol(size_t idx__) const
    {
        return z_columns_[idx__];
    }

    inline int gvec_base_mapping(int igloc_base__) const
    {
        assert(gvec_base_ != nullptr);
        return gvec_base_mapping_(igloc_base__);
    }
};

/// Stores information about G-vector partitioning between MPI ranks for the FFT transformation.
/** FFT driver works with a small communicator. G-vectors are distributed over the entire communicator which is
    larger than the FFT communicator. In order to transform the functions, G-vectors must be redistributed to the
    FFT-friendly "fat" slabs based on the FFT communicator size. */
class Gvec_partition
{
  private:
    /// Pointer to the G-vector instance.
    Gvec const& gvec_;

    /// Communicator for the FFT.
    Communicator const& fft_comm_;

    Communicator const& comm_ortho_fft_;

    /// Distribution of G-vectors for FFT.
    block_data_descriptor gvec_distr_fft_;

    /// Distribution of z-columns for FFT.
    block_data_descriptor zcol_distr_fft_;

    /// Distribution of G-vectors inside FFT-friendly "fat" slab.
    block_data_descriptor gvec_fft_slab_;

    /// Offset of the z-column in the local data buffer.
    /** Global index of z-column is expected */
    mdarray<int, 1> zcol_offs_;

    mdarray<int, 2> rank_map_;

    /// Global index of z-column in new (fat-slab) distrubution.
    /** This is a mapping between new and original ordering of z-columns. */
    mdarray<int, 1> idx_zcol_;

    /// Global index of G-vector by local index inside fat-salb.
    mdarray<int, 1> idx_gvec_;

    inline void build_fft_distr()
    {
        /* calculate distribution of G-vectors and z-columns for the FFT communicator */
        gvec_distr_fft_ = block_data_descriptor(fft_comm().size());
        zcol_distr_fft_ = block_data_descriptor(fft_comm().size());

        for (int rank = 0; rank < fft_comm().size(); rank++) {
            for (int i = 0; i < comm_ortho_fft().size(); i++) {
                /* fine-grained rank */
                int r = rank_map_(rank, i);
                gvec_distr_fft_.counts[rank] += gvec().gvec_count(r);
                zcol_distr_fft_.counts[rank] += gvec().zcol_count(r);
            }
        }
        /* get offsets of z-columns */
        zcol_distr_fft_.calc_offsets();
        /* get offsets of G-vectors */
        gvec_distr_fft_.calc_offsets();
    }

    /// Calculate offsets of z-columns inside each local buffer of PW coefficients.
    inline void calc_offsets()
    {
        zcol_offs_ = mdarray<int, 1>(gvec().num_zcol(), memory_t::host, "Gvec_partition.zcol_offs_");
        for (int rank = 0; rank < fft_comm().size(); rank++) {
            int offs{0};
            /* loop over local number of z-columns */
            for (int i = 0; i < zcol_count_fft(rank); i++) {
                /* global index of z-column */
                int icol         = idx_zcol_[zcol_distr_fft_.offsets[rank] + i];
                zcol_offs_[icol] = offs;
                offs += static_cast<int>(gvec().zcol(icol).z.size());
            }
            assert(offs == gvec_distr_fft_.counts[rank]);
        }
    }

    inline void pile_gvec()
    {
        /* build a table of {offset, count} values for G-vectors in the swapped distribution;
         * we are preparing to swap plane-wave coefficients from a default slab distribution to a FFT-friendly
         * distribution
         * +--------------+      +----+----+----+
         * |    :    :    |      |    |    |    |
         * +--------------+      |....|....|....|
         * |    :    :    |  ->  |    |    |    |
         * +--------------+      |....|....|....|
         * |    :    :    |      |    |    |    |
         * +--------------+      +----+----+----+
         *
         * i.e. we will make G-vector slabs more fat (pile-of-slabs) and at the same time reshulffle wave-functions
         * between columns of the 2D MPI grid */
        gvec_fft_slab_ = block_data_descriptor(comm_ortho_fft_.size());
        for (int i = 0; i < comm_ortho_fft_.size(); i++) {
            gvec_fft_slab_.counts[i] = gvec().gvec_count(rank_map_(fft_comm_.rank(), i));
        }
        gvec_fft_slab_.calc_offsets();

        assert(gvec_fft_slab_.offsets.back() + gvec_fft_slab_.counts.back() == gvec_distr_fft_.counts[fft_comm().rank()]);
    }

  public:
    Gvec_partition(Gvec const& gvec__, Communicator const& fft_comm__, Communicator const& comm_ortho_fft__)
        : gvec_(gvec__)
        , fft_comm_(fft_comm__)
        , comm_ortho_fft_(comm_ortho_fft__)
    {
        if (fft_comm_.size() * comm_ortho_fft_.size() != gvec_.comm().size()) {
            TERMINATE("wrong size of communicators");
        }
        rank_map_ = mdarray<int, 2>(fft_comm_.size(), comm_ortho_fft_.size());
        rank_map_.zero();
        /* get a global rank */
        rank_map_(fft_comm_.rank(), comm_ortho_fft_.rank()) = gvec_.comm().rank();
        gvec_.comm().allreduce(&rank_map_(0, 0), gvec_.comm().size());

        build_fft_distr();

        idx_zcol_ = mdarray<int, 1>(gvec().num_zcol());
        int icol{0};
        for (int rank = 0; rank < fft_comm().size(); rank++) {
            for (int i = 0; i < comm_ortho_fft().size(); i++) {
                for (int k = 0; k < gvec_.zcol_count(rank_map_(rank, i)); k++) {
                    idx_zcol_(icol) = gvec_.zcol_offset(rank_map_(rank, i)) + k;
                    icol++;
                }
            }
            assert(icol == zcol_distr_fft_.counts[rank] + zcol_distr_fft_.offsets[rank]);
        }
        assert(icol == gvec().num_zcol());

        idx_gvec_ = mdarray<int, 1>(gvec_count_fft());
        int ig{0};
        for (int i = 0; i < comm_ortho_fft_.size(); i++) {
            for (int k = 0; k < gvec_.gvec_count(rank_map_(fft_comm_.rank(), i)); k++) {
                idx_gvec_(ig) = gvec_.gvec_offset(rank_map_(fft_comm_.rank(), i)) + k;
                ig++;
            }
        }

        calc_offsets();
        pile_gvec();
    }

    /// Return FFT communicator
    inline Communicator const& fft_comm() const
    {
        return fft_comm_;
    }

    inline Communicator const& comm_ortho_fft() const
    {
        return comm_ortho_fft_;
    }

    inline int gvec_count_fft(int rank__) const
    {
        return gvec_distr_fft_.counts[rank__];
    }

    /// Local number of G-vectors for FFT-friendly distibution.
    inline int gvec_count_fft() const
    {
        return gvec_count_fft(fft_comm().rank());
    }

    /// Return local number of z-columns.
    inline int zcol_count_fft(int rank__) const
    {
        return zcol_distr_fft_.counts[rank__];
    }

    inline int zcol_count_fft() const
    {
        return zcol_count_fft(fft_comm().rank());
    }

    template <index_domain_t index_domain>
    inline int idx_zcol(int idx__) const
    {
        switch (index_domain) {
            case index_domain_t::local: {
                return idx_zcol_(zcol_distr_fft_.offsets[fft_comm().rank()] + idx__);
                break;
            }
            case index_domain_t::global: {
                return idx_zcol_(idx__);
                break;
            }
        }
    }

    inline int idx_gvec(int idx_local__) const
    {
        return idx_gvec_(idx_local__);
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
        return gvec_;
    }

    void gather_pw_fft(double_complex* f_pw_local__, double_complex* f_pw_fft__) const
    {
        int rank = gvec().comm().rank();
        /* collect scattered PW coefficients */
        comm_ortho_fft().allgather(f_pw_local__, gvec().gvec_count(rank), f_pw_fft__,
                                   gvec_fft_slab().counts.data(), gvec_fft_slab().offsets.data());
    }

    void gather_pw_global(double_complex* f_pw_fft__, double_complex* f_pw_global__) const
    {
        for (int ig = 0; ig < gvec().count(); ig++) {
            /* position inside fft buffer */
            int ig1 = gvec_fft_slab().offsets[comm_ortho_fft().rank()] + ig;
            f_pw_global__[gvec().offset() + ig] = f_pw_fft__[ig1];
        }
        gvec().comm().allgather(&f_pw_global__[0], gvec().offset(), gvec().count());
    }
};

/// Helper class to redistribute G-vectors for symmetrization.
/** G-vectors are remapped from default distribution which balances both the local number
 *  of z-columns and G-vectors to the distributio of G-vector shells in which each MPI rank stores
 *  local set of complete G-vector shells such that the "rotated" G-vector remains on the same MPI rank. */
struct remap_gvec_to_shells
{
    /// Sending counts and offsets.
    block_data_descriptor a2a_send;

    /// Receiving counts and offsets.
    block_data_descriptor a2a_recv;

    /// Split global index of G-shells between MPI ranks.
    splindex<block_cyclic> spl_num_gsh;

    /// List of G-vectors in the remapped storage.
    mdarray<int, 2> gvec_remapped_;

    /// Mapping between index of local G-vector and global index of G-vector shell.
    mdarray<int, 1> gvec_shell_remapped_;

    /* mapping beween G-shell index and local G-vector index */
    // std::map<int, std::vector<int>> gvec_sh_;

    Communicator const& comm_;

    Gvec const& gvec_;

    /// A mapping between G-vector and it's local index in the new distribution.
    std::map<vector3d<int>, int> idx_gvec;

    remap_gvec_to_shells(Communicator const& comm__, Gvec const& gvec__)
        : comm_(comm__)
        , gvec_(gvec__)
    {
        PROFILE("sddk::remap_gvec_to_shells|init");

        a2a_send = block_data_descriptor(comm_.size());
        a2a_recv = block_data_descriptor(comm_.size());

        /* split G-vector shells between ranks in cyclic order */
        spl_num_gsh = splindex<block_cyclic>(gvec_.num_shells(), comm_.size(), comm_.rank(), 1);

        /* each rank sends a fraction of its local G-vectors to other ranks */
        /* count this fraction */
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig   = gvec_.offset() + igloc;
            int igsh = gvec__.shell(ig);
            a2a_send.counts[spl_num_gsh.local_rank(igsh)]++;
        }
        a2a_send.calc_offsets();
        /* sanity check: total number of elements to send is equal to the local number of G-vector */
        if (a2a_send.size() != gvec_.count()) {
            TERMINATE("wrong number of G-vectors");
        }
        /* count the number of elements to receive */
        for (int r = 0; r < comm_.size(); r++) {
            for (int igloc = 0; igloc < gvec_.gvec_count(r); igloc++) {
                int ig   = gvec_.gvec_offset(r) + igloc;
                int igsh = gvec_.shell(ig);
                if (spl_num_gsh.local_rank(igsh) == comm_.rank()) {
                    a2a_recv.counts[r]++;
                }
            }
        }
        a2a_recv.calc_offsets();

        /* local set of G-vectors in the remapped order */
        gvec_remapped_       = mdarray<int, 2>(3, a2a_recv.size());
        gvec_shell_remapped_ = mdarray<int, 1>(a2a_recv.size());
        std::vector<int> counts(comm_.size(), 0);
        for (int r = 0; r < comm_.size(); r++) {
            for (int igloc = 0; igloc < gvec_.gvec_count(r); igloc++) {
                int ig   = gvec_.gvec_offset(r) + igloc;
                int igsh = gvec_.shell(ig);
                auto G   = gvec_.gvec(ig);
                if (spl_num_gsh.local_rank(igsh) == comm_.rank()) {
                    for (int x = 0; x < 3; x++) {
                        gvec_remapped_(x, a2a_recv.offsets[r] + counts[r]) = G[x];
                    }
                    gvec_shell_remapped_(a2a_recv.offsets[r] + counts[r]) = igsh;
                    counts[r]++;
                }
            }
        }
        /* sanity check: sum of local sizes in the remapped order is equal to the total number of G-vectors */
        int ng = a2a_recv.size();
        comm_.allreduce(&ng, 1);
        if (ng != gvec_.num_gvec()) {
            TERMINATE("wrong number of G-vectors");
        }

        for (int ig = 0; ig < a2a_recv.size(); ig++) {
            vector3d<int> G(&gvec_remapped_(0, ig));
            idx_gvec[G] = ig;
            // int igsh = gvec_shell_remapped_(ig);
            // if (!gvec_sh_.count(igsh)) {
            //    gvec_sh_[igsh] = std::vector<int>();
            //}
            // gvec_sh_[igsh].push_back(ig);
        }
    }

    int index_by_gvec(vector3d<int> G__) const
    {
        if (idx_gvec.count(G__)) {
            return idx_gvec.at(G__);
        } else {
            return -1;
        }
    }

    int gvec_shell_remapped(int igloc__) const
    {
        return gvec_shell_remapped_(igloc__);
    }

    template <typename T>
    std::vector<T> remap_forward(T* data__) const
    {
        PROFILE("sddk::remap_gvec_to_shells|remap_forward");

        std::vector<T> send_buf(gvec_.count());
        std::vector<int> counts(comm_.size(), 0);
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig                                    = gvec_.offset() + igloc;
            int igsh                                  = gvec_.shell(ig);
            int r                                     = spl_num_gsh.local_rank(igsh);
            send_buf[a2a_send.offsets[r] + counts[r]] = data__[igloc];
            counts[r]++;
        }

        std::vector<T> recv_buf(a2a_recv.size());

        comm_.alltoall(send_buf.data(), a2a_send.counts.data(), a2a_send.offsets.data(), recv_buf.data(),
                       a2a_recv.counts.data(), a2a_recv.offsets.data());

        return std::move(recv_buf);
    }

    template <typename T>
    void remap_backward(std::vector<T> buf__, T* data__) const
    {
        PROFILE("sddk::remap_gvec_to_shells|remap_backward");

        std::vector<T> recv_buf(gvec_.count());

        comm_.alltoall(buf__.data(), a2a_recv.counts.data(), a2a_recv.offsets.data(), recv_buf.data(),
                       a2a_send.counts.data(), a2a_send.offsets.data());

        std::vector<int> counts(comm_.size(), 0);
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig        = gvec_.offset() + igloc;
            int igsh      = gvec_.shell(ig);
            int r         = spl_num_gsh.local_rank(igsh);
            data__[igloc] = recv_buf[a2a_send.offsets[r] + counts[r]];
            counts[r]++;
        }
    }
};

} // namespace sddk

#endif //__GVEC_HPP__

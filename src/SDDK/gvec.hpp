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
#include <type_traits>
#include "memory.hpp"
#include "fft3d_grid.hpp"
#include "geometry3d.hpp"
#include "serializer.hpp"
#include "splindex.hpp"
#include "utils/profiler.hpp"
#include "utils/rte.hpp"

using namespace geometry3d;

namespace sddk {

inline FFT3D_grid get_min_fft_grid(double cutoff__, matrix3d<double> M__)
{
    return FFT3D_grid(find_translations(cutoff__, M__) + vector3d<int>({2, 2, 2}));
}

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
    /// Default constructor.
    z_column_descriptor()
    {
    }
};

/// Serialize a single z-column descriptor.
inline void serialize(serializer& s__, z_column_descriptor const& zcol__)
{
    serialize(s__, zcol__.x);
    serialize(s__, zcol__.y);
    serialize(s__, zcol__.z);
}

/// Deserialize a single z-column descriptor.
inline void deserialize(serializer& s__, z_column_descriptor& zcol__)
{
    deserialize(s__, zcol__.x);
    deserialize(s__, zcol__.y);
    deserialize(s__, zcol__.z);
}

/// Serialize a vector of z-column descriptors.
inline void serialize(serializer& s__, std::vector<z_column_descriptor> const& zcol__)
{
    serialize(s__, zcol__.size());
    for (auto& e: zcol__) {
        serialize(s__, e);
    }
}

/// Deserialize a vector of z-column descriptors.
inline void deserialize(serializer& s__, std::vector<z_column_descriptor>& zcol__)
{
    size_t sz;
    deserialize(s__, sz);
    zcol__.resize(sz);
    for (size_t i = 0; i < sz; i++) {
        deserialize(s__, zcol__[i]);
    }
}

/* forward declarations */
class Gvec;
void serialize(serializer& s__, Gvec const& gv__);
void deserialize(serializer& s__, Gvec& gv__);
Gvec send_recv(Communicator const& comm__, Gvec const& gv_src__, int source__, int dest__);

/// A set of G-vectors for FFTs and G+k basis functions.
/** Current implemntation supports up to 2^12 (4096) z-dimension of the FFT grid and 2^20 (1048576) number of
 *  z-columns. The order of z-sticks and G-vectors is not fixed and depends on the number of MPI ranks used
 *  for the parallelization. */
class Gvec
{
  private:
    /// k-vector of G+k.
    vector3d<double> vk_{0, 0, 0};

    /// Cutoff for |G+k| vectors.
    double Gmax_{0};

    /// Reciprocal lattice vectors.
    matrix3d<double> lattice_vectors_;

    /// Total communicator which is used to distribute G or G+k vectors.
    Communicator comm_;

    /// Indicates that G-vectors are reduced by inversion symmetry.
    bool reduce_gvec_{false};

    /// True if this a list of G-vectors without k-point shift.
    bool bare_gvec_{true};

    /// Total number of G-vectors.
    int num_gvec_{0};

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

    /// Number of G-vector shells (groups of G-vectors with the same length).
    int num_gvec_shells_;

    /// Radii (or lengths) of G-vector shells in a.u.^-1.
    sddk::mdarray<double, 1> gvec_shell_len_;

    /// Local number of G-vector shells for the local number of G-vectors.
    /** G-vectors are distributed by sticks, not by G-shells. This means that each rank stores local fraction of
        G-vectors with a non-consecutive G-shell index and not all G-shells are present at a given rank. This
        variable stores the number of G-shells which this rank holds. */
    int num_gvec_shells_local_;

    /// Radii of G-vector shells in the local index counting [0, num_gvec_shells_local)
    std::vector<double> gvec_shell_len_local_;

    /// Mapping between local index of G-vector and local G-shell index.
    std::vector<int> gvec_shell_idx_local_;

    sddk::mdarray<int, 3> gvec_index_by_xy_;

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

    /// Lattice coordinates of a local set of G-vectors.
    /** This are also known as Miller indices */
    mdarray<int, 2> gvec_;

    /// Lattice coordinates of a local set of G+k-vectors.
    mdarray<double, 2> gkvec_;

    /// Cartiesian coordinaes of a local set of G-vectors.
    mdarray<double, 2> gvec_cart_;

    /// Cartesian coordinaes of a local set of G+k-vectors.
    mdarray<double, 2> gkvec_cart_;

    /// Length of the local fraction of G-vectors.
    mdarray<double, 1> gvec_len_;

    /// Offset in the global index for the local part of G-vectors.
    int offset_{-1};

    /// Local number of G-vectors.
    int count_{-1};

    /// Local number of z-columns.
    int num_zcol_local_{-1};

    /// Return corresponding G-vector for an index in the range [0, num_gvec).
    vector3d<int> gvec_by_full_index(uint32_t idx__) const;

    /// Find z-columns of G-vectors inside a sphere with Gmax radius.
    /** This function also computes the total number of G-vectors. */
    void find_z_columns(double Gmax__, FFT3D_grid const& fft_box__);

    /// Distribute z-columns between MPI ranks.
    void distribute_z_columns();

    /// Find a list of G-vector shells.
    /** G-vectors belonging to the same shell have the same length and transform to each other
        under a lattice symmetry operation. */
    void find_gvec_shells();

    /// Initialize lattice coordinates of the local fraction of G-vectors.
    void init_gvec_local();

    /// Initialize Cartesian coordinates of the local fraction of G-vectors.
    void init_gvec_cart_local();

    /// Initialize everything.
    void init(FFT3D_grid const& fft_grid);

    friend void sddk::serialize(serializer& s__, Gvec const& gv__);

    friend void sddk::deserialize(serializer& s__, Gvec& gv__);

    /* copy constructor is forbidden */
    Gvec(Gvec const& src__) = delete;

    /* copy assignment operator is forbidden */
    Gvec& operator=(Gvec const& src__) = delete;

  public:
    /// Constructor for G+k vectors.
    /** \param [in] vk          K-point vector of G+k
     *  \param [in] M           Reciprocal lattice vecotors in comumn order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
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
        init(get_min_fft_grid(Gmax__, M__));
    }

    /// Constructor for G-vectors.
    /** \param [in] M           Reciprocal lattice vecotors in comumn order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
     *  \param [in] reduce_gvec True if G-vectors need to be reduced by inversion symmetry.
     */
    Gvec(matrix3d<double> M__, double Gmax__, Communicator const& comm__, bool reduce_gvec__)
        : Gmax_(Gmax__)
        , lattice_vectors_(M__)
        , comm_(comm__)
        , reduce_gvec_(reduce_gvec__)
    {
        init(get_min_fft_grid(Gmax__, M__));
    }

    /// Constructor for G-vectors.
    /** \param [in] M           Reciprocal lattice vecotors in comumn order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] fft_grid    Provide explicit boundaries for the G-vector min and max frequencies.
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
     *  \param [in] reduce_gvec True if G-vectors need to be reduced by inversion symmetry.
     */
    Gvec(matrix3d<double> M__, double Gmax__, FFT3D_grid const& fft_grid__, Communicator const& comm__, bool reduce_gvec__)
        : Gmax_(Gmax__)
        , lattice_vectors_(M__)
        , comm_(comm__)
        , reduce_gvec_(reduce_gvec__)
    {
        init(fft_grid__);
    }

    /// Constructor for G-vector distribution based on a previous set.
    /** Previous set of G-vectors must be a subset of the current set. */
    Gvec(double Gmax__, Gvec const& gvec_base__)
        : Gmax_(Gmax__)
        , lattice_vectors_(gvec_base__.lattice_vectors())
        , comm_(gvec_base__.comm())
        , reduce_gvec_(gvec_base__.reduced())
        , gvec_base_(&gvec_base__)
    {
        init(get_min_fft_grid(Gmax__, lattice_vectors_));
    }

    /// Constructor for G-vectors with mpi_comm_self()
    Gvec(matrix3d<double> M__, double Gmax__, bool reduce_gvec__)
        : Gmax_(Gmax__)
        , lattice_vectors_(M__)
        , comm_(Communicator::self())
        , reduce_gvec_(reduce_gvec__)
    {
        init(get_min_fft_grid(Gmax__, M__));
    }

    Gvec(vector3d<double> vk__, matrix3d<double> M__, int ngv_loc__, int const* gv__, Communicator const& comm__, bool reduce_gvec__)
        : vk_(vk__)
        , lattice_vectors_(M__)
        , comm_(comm__)
        , reduce_gvec_(reduce_gvec__)
        , bare_gvec_(false)
        , count_(ngv_loc__)
    {
        sddk::mdarray<int, 2> G(const_cast<int*>(gv__), 3, ngv_loc__);

        gvec_  = mdarray<int, 2>(3, count(), memory_t::host, "gvec_");
        gkvec_ = mdarray<double, 2>(3, count(), memory_t::host, "gkvec_");

        /* do a first pass: determine boundaries of the grid */
        int xmin{0}, xmax{0};
        int ymin{0}, ymax{0};
        for (int i = 0; i < ngv_loc__; i++) {
            xmin = std::min(xmin, G(0, i));
            xmax = std::max(xmax, G(0, i));
            ymin = std::min(ymin, G(1, i));
            ymax = std::max(ymax, G(1, i));
        }
        comm_.allreduce<int, mpi_op_t::min>(&xmin, 1);
        comm_.allreduce<int, mpi_op_t::min>(&ymin, 1);
        comm_.allreduce<int, mpi_op_t::max>(&xmax, 1);
        comm_.allreduce<int, mpi_op_t::max>(&ymax, 1);

        sddk::mdarray<int, 2> zcol(mdarray_index_descriptor(xmin, xmax), mdarray_index_descriptor(ymin, ymax));
        zcol.zero();
        for (int ig = 0; ig < ngv_loc__; ig++) {
            zcol(G(0, ig), G(1, ig))++;
            for (int x : {0, 1, 2}) {
                gvec_(x, ig) = G(x, ig);
                gkvec_(x, ig) = G(x, ig) + vk_[x];
            }
        }
        num_zcol_local_ = 0;
        for (size_t i = 0; i < zcol.size(); i++) {
            if (zcol[i]) {
                num_zcol_local_++;
            }
        }

        init_gvec_cart_local();

        gvec_distr_ = block_data_descriptor(comm().size());
        comm().allgather(&count_, gvec_distr_.counts.data(), 1, comm_.rank());
        gvec_distr_.calc_offsets();
        offset_ = gvec_distr_.offsets[comm().rank()];

        zcol_distr_ = block_data_descriptor(comm().size());
        comm().allgather(&num_zcol_local_, zcol_distr_.counts.data(), 1, comm_.rank());
        zcol_distr_.calc_offsets();

        num_gvec_ = count_;
        comm().allreduce(&num_gvec_, 1);
    }

    /// Constructor for empty set of G-vectors.
    Gvec(Communicator const& comm__)
        : comm_(comm__)
    {
    }

    /// Move assignment operator.
    Gvec& operator=(Gvec&& src__) = default;

    /// Move constructor.
    Gvec(Gvec&& src__) = default;

    inline auto const& vk() const
    {
        return vk_;
    }

    inline Communicator const& comm() const
    {
        return comm_;
    }

    /// Set the new reciprocal lattice vectors.
    /** For the varibale-cell relaxation runs we need an option to preserve the number of G- and G+k vectors.
     *  Here we can set the new lattice vectors and update the relevant members of the Gvec class. */
    inline auto const& lattice_vectors(matrix3d<double> lattice_vectors__)
    {
        lattice_vectors_ = lattice_vectors__;
        find_gvec_shells();
        init_gvec_cart_local();
        return lattice_vectors_;
    }

    /// Retrn a const reference to the reciprocal lattice vectors.
    inline matrix3d<double> const& lattice_vectors() const
    {
        return lattice_vectors_;
    }

    /// Return the volume of the real space unit cell that corresponds to the reciprocal lattice of G-vectors.
    inline double omega() const
    {
        double const twopi_pow3 = 248.050213442398561403810520537;
        return twopi_pow3 / std::abs(lattice_vectors().det());
    }

    /// Return the total number of G-vectors within the cutoff.
    inline int num_gvec() const
    {
        return num_gvec_;
    }

    /// Number of z-columns for a fine-grained distribution.
    inline int zcol_count(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return zcol_distr_.counts[rank__];
    }

    /// Offset in the global index of z-columns for a given rank.
    inline int zcol_offset(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return zcol_distr_.offsets[rank__];
    }

    /// Number of G-vectors for a fine-grained distribution.
    inline int gvec_count(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return gvec_distr_.counts[rank__];
    }

    /// Number of G-vectors for a fine-grained distribution for the current MPI rank.
    /** The \em count and \em offset are borrowed from the MPI terminology for data distribution. */
    inline int count() const
    {
        return count_;
    }

    /// Offset (in the global index) of G-vectors for a fine-grained distribution.
    inline int gvec_offset(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return gvec_distr_.offsets[rank__];
    }

    /// Offset (in the global index) of G-vectors for a fine-grained distribution for a current MPI rank.
    /** The \em count and \em offset are borrowed from the MPI terminology for data distribution. */
    inline int offset() const
    {
        return offset_;
    }

    /// Local starting index of G-vectors if G=0 is not counted.
    inline int skip_g0() const
    {
        return (comm().rank() == 0) ? 1 : 0;
    }

    /// Return number of G-vector shells.
    inline int num_shells() const
    {
        return num_gvec_shells_;
    }

    /// Return G vector in fractional coordinates.
    template <index_domain_t idx_t>
    inline vector3d<int>
    gvec(int ig__) const
    {
        switch (idx_t) {
            case index_domain_t::local: {
                return vector3d<int>(gvec_(0, ig__), gvec_(1, ig__), gvec_(2, ig__));
                break;
            }
            case index_domain_t::global: {
                return gvec_by_full_index(gvec_full_index_(ig__));
                break;
            }
        }
    }

    /// Return G+k vector in fractional coordinates.
    template <index_domain_t idx_t>
    inline vector3d<double>
    gkvec(int ig__) const
    {
        switch (idx_t) {
            case index_domain_t::local: {
                return vector3d<double>(gkvec_(0, ig__), gkvec_(1, ig__), gkvec_(2, ig__));
                break;
            }
            case index_domain_t::global: {
                return this->gvec<idx_t>(ig__) + vk_;
                break;
            }
        }
    }

    /// Return G vector in Cartesian coordinates.
    template <index_domain_t idx_t>
    inline vector3d<double>
    gvec_cart(int ig__) const
    {
        switch (idx_t) {
            case index_domain_t::local: {
                return vector3d<double>(gvec_cart_(0, ig__), gvec_cart_(1, ig__), gvec_cart_(2, ig__));
                break;
            }
            case index_domain_t::global: {
                auto G = this->gvec<idx_t>(ig__);
                return dot(lattice_vectors_, G);
                break;
            }
        }
    }

    /// Return G+k vector in fractional coordinates.
    template <index_domain_t idx_t>
    inline vector3d<double>
    gkvec_cart(int ig__) const
    {
        switch (idx_t) {
            case index_domain_t::local: {
                return vector3d<double>(gkvec_cart_(0, ig__), gkvec_cart_(1, ig__), gkvec_cart_(2, ig__));
                break;
            }
            case index_domain_t::global: {
                auto Gk = this->gvec<idx_t>(ig__) + vk_;
                return dot(lattice_vectors_, Gk);
                break;
            }
        }
    }

    /// Return index of the G-vector shell by the G-vector index.
    inline int shell(int ig__) const
    {
        return gvec_shell_(ig__);
    }

    inline int shell(vector3d<int> const& G__) const
    {
        return this->shell(index_by_gvec(G__));
    }

    /// Return length of the G-vector shell.
    inline double shell_len(int igs__) const
    {
        return gvec_shell_len_(igs__);
    }

    /// Get lengths of all G-vector shells.
    std::vector<double> shells_len() const
    {
        std::vector<double> q(this->num_shells());
        for (int i = 0; i < this->num_shells(); i++) {
            q[i] = this->shell_len(i);
        }
        return q;
    }

    /// Return length of the G-vector.
    template <index_domain_t idx_t>
    inline double
    gvec_len(int ig__) const
    {
        switch (idx_t) {
            case index_domain_t::local: {
                return gvec_len_(ig__);
                break;
            }
            case index_domain_t::global: {
                return gvec_shell_len_(gvec_shell_(ig__));
                break;
            }
        }
    }

    inline int index_g12(vector3d<int> const& g1__, vector3d<int> const& g2__) const
    {
        auto v  = g1__ - g2__;
        int idx = index_by_gvec(v);
        RTE_ASSERT(idx >= 0);
        RTE_ASSERT(idx < num_gvec());
        return idx;
    }

    std::pair<int, bool> index_g12_safe(vector3d<int> const& g1__, vector3d<int> const& g2__) const;

    //inline int index_g12_safe(int ig1__, int ig2__) const
    //{
    //    STOP();
    //    return 0;
    //}

    /// Return a global G-vector index in the range [0, num_gvec) by the G-vector.
    /** The information about a G-vector index is encoded by two numbers: a starting index for the
     *  column of G-vectors and column's size. Depending on the geometry of the reciprocal lattice,
     *  z-columns may have only negative, only positive or both negative and positive frequencies for
     *  a given x and y. This information is used to compute the offset which is added to the starting index
     *  in order to get a full G-vector index. Check find_z_columns() to see how the z-columns are found and
     *  added to the list of columns. */
    int index_by_gvec(vector3d<int> const& G__) const;

    inline bool reduced() const
    {
        return reduce_gvec_;
    }

    inline bool bare() const
    {
        return bare_gvec_;
    }

    /// Return global number of z-columns.
    inline int num_zcol() const
    {
        return static_cast<int>(z_columns_.size());
    }

    /// Return local number of z-columns.
    inline int num_zcol_local() const
    {
        return num_zcol_local_;
    }

    inline z_column_descriptor const& zcol(size_t idx__) const
    {
        return z_columns_[idx__];
    }

    inline int gvec_base_mapping(int igloc_base__) const
    {
        RTE_ASSERT(gvec_base_ != nullptr);
        return gvec_base_mapping_(igloc_base__);
    }

    inline int num_gvec_shells_local() const
    {
        return num_gvec_shells_local_;
    }

    inline double gvec_shell_len_local(int idx__) const
    {
        return gvec_shell_len_local_[idx__];
    }

    inline int gvec_shell_idx_local(int igloc__) const
    {
        return gvec_shell_idx_local_[igloc__];
    }

    /// Return local list of G-vectors.
    inline auto const& gvec_local() const
    {
        return gvec_;
    }

    /// Return local list of G-vectors for a given rank.
    /** This function must be called by all MPI ranks of the G-vector communicator. */
    inline auto gvec_local(int rank__) const
    {
        int ngv = this->count();
        this->comm().bcast(&ngv, 1, rank__);
        mdarray<int, 2> result(3, ngv);
        if (this->comm().rank() == rank__) {
            RTE_ASSERT(ngv == this->count());
            copy(this->gvec_, result);
        }
        this->comm().bcast(&result(0, 0), 3 * ngv, rank__);
        return result;
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

    /// Communicator which is orthogonal to FFT communicator.
    Communicator const& comm_ortho_fft_;

    /// Distribution of G-vectors for FFT.
    block_data_descriptor gvec_distr_fft_;

    /// Distribution of z-columns for FFT.
    block_data_descriptor zcol_distr_fft_;

    /// Distribution of G-vectors inside FFT-friendly "fat" slab.
    block_data_descriptor gvec_fft_slab_;

    /// Mapping of MPI ranks used to split G-vectors to a 2D grid.
    mdarray<int, 2> rank_map_;

    /// Lattice coordinates of a local set of G-vectors.
    /** These are also known as Miller indices */
    mdarray<int, 2> gvec_array_;

    /// Cartesian coordinaes of a local set of G+k-vectors.
    mdarray<double, 2> gkvec_cart_array_;

    void build_fft_distr();

    /// Stack together the G-vector slabs to make a larger ("fat") slab for a FFT driver.
    void pile_gvec();

  public:
    Gvec_partition(Gvec const& gvec__, Communicator const& fft_comm__, Communicator const& comm_ortho_fft__);

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

    /// Local number of G-vectors for FFT-friendly distribution.
    inline int gvec_count_fft() const
    {
        return gvec_count_fft(fft_comm().rank());
    }

    /// Return local number of z-columns.
    inline int zcol_count_fft() const
    {
        return zcol_distr_fft_.counts[fft_comm().rank()];
    }

    inline block_data_descriptor const& gvec_fft_slab() const
    {
        return gvec_fft_slab_;
    }

    inline Gvec const& gvec() const
    {
        return gvec_;
    }

    inline vector3d<double> gkvec_cart(int igloc__) const
    {
        return vector3d<double>(&gkvec_cart_array_(0, igloc__));
    }

    inline mdarray<int, 2> const& gvec_array() const
    {
        return gvec_array_;
    }

    template <typename T> // TODO: document
    void gather_pw_fft(std::complex<T> const* f_pw_local__, std::complex<T>* f_pw_fft__) const
    {
        int rank = gvec().comm().rank();
        /* collect scattered PW coefficients */
        comm_ortho_fft().allgather(f_pw_local__, gvec().gvec_count(rank), f_pw_fft__, gvec_fft_slab().counts.data(),
                                   gvec_fft_slab().offsets.data());

    }

    template <typename T> // TODO: document
    void gather_pw_global(std::complex<T> const* f_pw_fft__, std::complex<T>* f_pw_global__) const
    {
        for (int ig = 0; ig < gvec().count(); ig++) {
            /* position inside fft buffer */
            int ig1                             = gvec_fft_slab().offsets[comm_ortho_fft().rank()] + ig;
            f_pw_global__[gvec().offset() + ig] = f_pw_fft__[ig1];
        }
        gvec().comm().allgather(&f_pw_global__[0], gvec().count(), gvec().offset());
    }

    template<typename T>
    void scatter_pw_global(std::complex<T> const* f_pw_global__, std::complex<T>* f_pw_fft__) const
    {
        for (int i = 0; i < comm_ortho_fft_.size(); i++) {
            /* offset in global index */
            int offset = this->gvec_.gvec_offset(rank_map_(fft_comm_.rank(), i));
            for (int ig = 0; ig < gvec_fft_slab_.counts[i]; ig++) {
                f_pw_fft__[gvec_fft_slab_.offsets[i] + ig] = f_pw_global__[offset + ig];
            }
        }
    }

    void update_gkvec_cart()
    {
        for (int ig = 0; ig < this->gvec_count_fft(); ig++) {
            auto G = vector3d<int>(&gvec_array_(0, ig));
            auto Gkc = dot(this->gvec_.lattice_vectors(), G + this->gvec_.vk());
            for (int x : {0, 1, 2}) {
                gkvec_cart_array_(x, ig) = Gkc[x];
            }
        }
    }
};

/// Helper class to manage G-vector shells and redistribute G-vectors for symmetrization.
/** G-vectors are remapped from default distribution which balances both the local number
    of z-columns and G-vectors to the distribution of G-vector shells in which each MPI rank stores
    local set of complete G-vector shells such that the "rotated" G-vector remains on the same MPI rank.
 */
class Gvec_shells
{
  private:
    /// Sending counts and offsets.
    block_data_descriptor a2a_send_;

    /// Receiving counts and offsets.
    block_data_descriptor a2a_recv_;

    /// Split global index of G-shells between MPI ranks.
    splindex<splindex_t::block_cyclic> spl_num_gsh_;

    /// List of G-vectors in the remapped storage.
    sddk::mdarray<int, 2> gvec_remapped_;

    /// Mapping between index of local G-vector and global index of G-vector shell.
    sddk::mdarray<int, 1> gvec_shell_remapped_;

    /// Alias for the G-vector communicator.
    Communicator const& comm_;

    Gvec const& gvec_;

    /// A mapping between G-vector and it's local index in the new distribution.
    std::map<vector3d<int>, int> idx_gvec_;

  public:

    Gvec_shells(Gvec const& gvec__);

    inline void print_gvec() const
    {
        pstdout pout(gvec_.comm());
        pout.printf("rank: %i\n", gvec_.comm().rank());
        pout.printf("-- list of G-vectors in the remapped distribution --\n");
        for (int igloc = 0; igloc < gvec_count_remapped(); igloc++) {
            auto G = gvec_remapped(igloc);

            int igsh = gvec_shell_remapped(igloc);
            pout.printf("igloc=%i igsh=%i G=%i %i %i\n", igloc, igsh, G[0], G[1], G[2]);
        }
        pout.printf("-- reverse list --\n");
        for (auto const& e: idx_gvec_) {
            pout.printf("G=%i %i %i, igloc=%i\n", e.first[0], e.first[1], e.first[2], e.second);
        }
    }

    /// Local number of G-vectors in the remapped distribution with complete shells on each rank.
    int gvec_count_remapped() const
    {
        return a2a_recv_.size();
    }

    /// G-vector by local index (in the remapped set).
    vector3d<int> gvec_remapped(int igloc__) const
    {
        return vector3d<int>(gvec_remapped_(0, igloc__), gvec_remapped_(1, igloc__), gvec_remapped_(2, igloc__));
    }

    /// Return local index of the G-vector in the remapped set.
    int index_by_gvec(vector3d<int> G__) const
    {
        if (idx_gvec_.count(G__)) {
            return idx_gvec_.at(G__);
        } else {
            return -1;
        }
    }

    /// Index of the G-vector shell by the local G-vector index (in the remapped set).
    int gvec_shell_remapped(int igloc__) const
    {
        return gvec_shell_remapped_(igloc__);
    }

    template <typename T>
    std::vector<T> remap_forward(T* data__) const
    {
        PROFILE("sddk::Gvec_shells::remap_forward");

        std::vector<T> send_buf(gvec_.count());
        std::vector<int> counts(comm_.size(), 0);
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig                                     = gvec_.offset() + igloc;
            int igsh                                   = gvec_.shell(ig);
            int r                                      = spl_num_gsh_.local_rank(igsh);
            send_buf[a2a_send_.offsets[r] + counts[r]] = data__[igloc];
            counts[r]++;
        }

        std::vector<T> recv_buf(gvec_count_remapped());

        comm_.alltoall(send_buf.data(), a2a_send_.counts.data(), a2a_send_.offsets.data(), recv_buf.data(),
                       a2a_recv_.counts.data(), a2a_recv_.offsets.data());

        return recv_buf;
    }

    template <typename T>
    void remap_backward(std::vector<T> buf__, T* data__) const
    {
        PROFILE("sddk::Gvec_shells::remap_backward");

        std::vector<T> recv_buf(gvec_.count());

        comm_.alltoall(buf__.data(), a2a_recv_.counts.data(), a2a_recv_.offsets.data(), recv_buf.data(),
                       a2a_send_.counts.data(), a2a_send_.offsets.data());

        std::vector<int> counts(comm_.size(), 0);
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig        = gvec_.offset() + igloc;
            int igsh      = gvec_.shell(ig);
            int r         = spl_num_gsh_.local_rank(igsh);
            data__[igloc] = recv_buf[a2a_send_.offsets[r] + counts[r]];
            counts[r]++;
        }
    }

    inline Gvec const& gvec() const
    {
        return gvec_;
    }
};

} // namespace sddk

#endif //__GVEC_HPP__

/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file gvec.hpp
 *
 *  \brief Declaration and implementation of Gvec class.
 */

#ifndef __GVEC_HPP__
#define __GVEC_HPP__

#include <numeric>
#include <map>
#include <iostream>
#include <limits>
#include <type_traits>
#include "core/fft/fft3d_grid.hpp"
#include "core/r3/r3.hpp"
#include "core/splindex.hpp"
#include "core/serializer.hpp"
#include "core/mpi/pstdout.hpp"
#include "core/rte/rte.hpp"
#include "core/profiler.hpp"

namespace sirius {

namespace fft {

/// Descriptor of the z-column (x,y fixed, z varying) of the G-vectors.
/** Sphere of G-vectors within a given plane-wave cutoff is represented as a set of z-columns with different lengths. */
struct z_column_descriptor
{
    /// X-coordinate (can be negative and positive).
    int x;
    /// Y-coordinate (can be negative and positive).
    int y;
    /// Minimum z-coordinate.
    int z_min;
    /// Maximum z-coordinate.
    int z_max;
    /// List of the Z-coordinates of the column.
    std::vector<int> z;
    /// Constructor.
    z_column_descriptor(int x__, int y__, std::vector<int> z__)
        : x(x__)
        , y(y__)
        , z(z__)
    {
        z_min = std::numeric_limits<int>::max();
        z_max = std::numeric_limits<int>::min();
        for (auto e : z__) {
            z_min = std::min(z_min, e);
            z_max = std::max(z_max, e);
        }
    }
    /// Default constructor.
    z_column_descriptor()
    {
    }
};

/// Serialize a single z-column descriptor.
inline void
serialize(serializer& s__, z_column_descriptor const& zcol__)
{
    serialize(s__, zcol__.x);
    serialize(s__, zcol__.y);
    serialize(s__, zcol__.z_min);
    serialize(s__, zcol__.z_max);
    serialize(s__, zcol__.z);
}

/// Deserialize a single z-column descriptor.
inline void
deserialize(serializer& s__, z_column_descriptor& zcol__)
{
    deserialize(s__, zcol__.x);
    deserialize(s__, zcol__.y);
    deserialize(s__, zcol__.z_min);
    deserialize(s__, zcol__.z_max);
    deserialize(s__, zcol__.z);
}

/// Serialize a vector of z-column descriptors.
inline void
serialize(serializer& s__, std::vector<z_column_descriptor> const& zcol__)
{
    serialize(s__, zcol__.size());
    for (auto& e : zcol__) {
        serialize(s__, e);
    }
}

/// Deserialize a vector of z-column descriptors.
inline void
deserialize(serializer& s__, std::vector<z_column_descriptor>& zcol__)
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
void
serialize(serializer& s__, Gvec const& gv__);
void
deserialize(serializer& s__, Gvec& gv__);
Gvec
send_recv(mpi::Communicator const& comm__, Gvec const& gv_src__, int source__, int dest__);

/// A set of G-vectors for FFTs and G+k basis functions.
/** Current implemntation supports up to 2^12 (4096) z-dimension of the FFT grid and 2^20 (1048576) number of
 *  z-columns. The order of z-sticks and G-vectors is not fixed and depends on the number of MPI ranks used
 *  for the parallelization. */
class Gvec
{
  private:
    /// k-vector of G+k.
    r3::vector<double> vk_{0, 0, 0};

    /// Cutoff for |G+k| vectors.
    double Gmax_{0};

    /// Reciprocal lattice vectors.
    r3::matrix<double> lattice_vectors_;

    /// Total communicator which is used to distribute G or G+k vectors.
    mpi::Communicator comm_;

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
    mdarray<double, 1> gvec_shell_len_;

    /// Local number of G-vector shells for the local number of G-vectors.
    /** G-vectors are distributed by sticks, not by G-shells. This means that each rank stores local fraction of
        G-vectors with a non-consecutive G-shell index and not all G-shells are present at a given rank. This
        variable stores the number of G-shells which this rank holds. */
    int num_gvec_shells_local_;

    /// Radii of G-vector shells in the local index counting [0, num_gvec_shells_local)
    std::vector<double> gvec_shell_len_local_;

    /// Mapping between local index of G-vector and local G-shell index.
    std::vector<int> gvec_shell_idx_local_;

    mdarray<int, 3> gvec_index_by_xy_;

    /// Global list of non-zero z-columns.
    std::vector<z_column_descriptor> z_columns_;

    /// Fine-grained distribution of G-vectors.
    mpi::block_data_descriptor gvec_distr_;

    /// Fine-grained distribution of z-columns.
    mpi::block_data_descriptor zcol_distr_;

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

    // Theta- and phi- angles of G-vectors.
    mdarray<double, 2> gvec_tp_;

    // Theta- and phi- angles of G+k-vectors.
    mdarray<double, 2> gkvec_tp_;

    /// Offset in the global index for the local part of G-vectors.
    int offset_{-1};

    /// Local number of G-vectors.
    int count_{-1};

    /// Local number of z-columns.
    int num_zcol_local_{-1};

    /// Symmetry tolerance of the real-space lattice.
    double sym_tol_{1e-6};

    /// Return corresponding G-vector for an index in the range [0, num_gvec).
    r3::vector<int>
    gvec_by_full_index(uint32_t idx__) const;

    /// Find z-columns of G-vectors inside a sphere with Gmax radius.
    /** This function also computes the total number of G-vectors. */
    void
    find_z_columns(double Gmax__, fft::Grid const& fft_box__);

    /// Distribute z-columns between MPI ranks.
    void
    distribute_z_columns();

    /// Find a list of G-vector shells.
    /** G-vectors belonging to the same shell have the same length and transform to each other
        under a lattice symmetry operation. */
    void
    find_gvec_shells();

    /// Initialize lattice coordinates of the local fraction of G-vectors.
    void
    init_gvec_local();

    /// Initialize Cartesian coordinates of the local fraction of G-vectors.
    void
    init_gvec_cart_local();

    /// Initialize everything.
    void
    init(fft::Grid const& fft_grid);

    friend void
    serialize(serializer& s__, Gvec const& gv__);

    friend void
    deserialize(serializer& s__, Gvec& gv__);

    /* copy constructor is forbidden */
    Gvec(Gvec const& src__) = delete;

    /* copy assignment operator is forbidden */
    Gvec&
    operator=(Gvec const& src__) = delete;

  public:
    /// Constructor for G+k vectors.
    /** \param [in] vk          K-point vector of G+k
     *  \param [in] M           Reciprocal lattice vectors in column order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
     *  \param [in] reduce_gvec True if G-vectors need to be reduced by inversion symmetry.
     *  \param [in] sym_tol     Unit cell lattice symmetry tolerance.
     */
    Gvec(r3::vector<double> vk__, r3::matrix<double> M__, double Gmax__, mpi::Communicator const& comm__,
         bool reduce_gvec__, double sym_tol__ = 1e-6)
        : vk_{vk__}
        , Gmax_{Gmax__}
        , lattice_vectors_{M__}
        , comm_{comm__}
        , reduce_gvec_{reduce_gvec__}
        , bare_gvec_{false}
        , sym_tol_{sym_tol__}
    {
        init(fft::get_min_grid(Gmax__, M__));
    }

    /// Constructor for G-vectors.
    /** \param [in] M           Reciprocal lattice vectors in column order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
     *  \param [in] reduce_gvec True if G-vectors need to be reduced by inversion symmetry.
     *  \param [in] sym_tol     Unit cell lattice symmetry tolerance.
     */
    Gvec(r3::matrix<double> M__, double Gmax__, mpi::Communicator const& comm__, bool reduce_gvec__,
         double sym_tol__ = 1e-6)
        : Gmax_{Gmax__}
        , lattice_vectors_{M__}
        , comm_{comm__}
        , reduce_gvec_{reduce_gvec__}
        , sym_tol_{sym_tol__}
    {
        init(fft::get_min_grid(Gmax__, M__));
    }

    /// Constructor for G-vectors.
    /** \param [in] M           Reciprocal lattice vectors in column order
     *  \param [in] Gmax        Cutoff for G+k vectors
     *  \param [in] fft_grid    Provide explicit boundaries for the G-vector min and max frequencies.
     *  \param [in] comm        Total communicator which is used to distribute G-vectors
     *  \param [in] reduce_gvec True if G-vectors need to be reduced by inversion symmetry.
     *  \param [in] sym_tol     Unit cell lattice symmetry tolerance.
     */
    Gvec(r3::matrix<double> M__, double Gmax__, fft::Grid const& fft_grid__, mpi::Communicator const& comm__,
         bool reduce_gvec__, double sym_tol__ = 1e-6)
        : Gmax_{Gmax__}
        , lattice_vectors_{M__}
        , comm_{comm__}
        , reduce_gvec_{reduce_gvec__}
        , sym_tol_{sym_tol__}
    {
        init(fft_grid__);
    }

    /// Constructor for G-vector distribution based on a previous set.
    /** Previous set of G-vectors must be a subset of the current set. */
    Gvec(double Gmax__, Gvec const& gvec_base__)
        : Gmax_{Gmax__}
        , lattice_vectors_{gvec_base__.lattice_vectors_}
        , comm_{gvec_base__.comm()}
        , reduce_gvec_{gvec_base__.reduced()}
        , gvec_base_{&gvec_base__}
        , sym_tol_{gvec_base__.sym_tol_}
    {
        init(fft::get_min_grid(Gmax__, lattice_vectors_));
    }

    /// Constructor for G-vectors with mpi_comm_self()
    Gvec(r3::matrix<double> M__, double Gmax__, bool reduce_gvec__, double sym_tol__ = 1e-6)
        : Gmax_{Gmax__}
        , lattice_vectors_{M__}
        , comm_{mpi::Communicator::self()}
        , reduce_gvec_{reduce_gvec__}
        , sym_tol_{sym_tol__}
    {
        init(fft::get_min_grid(Gmax__, M__));
    }

    /// Construct with the defined order of G-vectors.
    Gvec(r3::vector<double> vk__, r3::matrix<double> M__, int ngv_loc__, int const* gv__,
         mpi::Communicator const& comm__, bool reduce_gvec__)
        : vk_(vk__)
        , lattice_vectors_(M__)
        , comm_(comm__)
        , reduce_gvec_(reduce_gvec__)
        , bare_gvec_(false)
        , count_(ngv_loc__)
    {
        mdarray<int, 2> G({3, ngv_loc__}, const_cast<int*>(gv__));

        gvec_  = mdarray<int, 2>({3, count()}, mdarray_label("gvec_"));
        gkvec_ = mdarray<double, 2>({3, count()}, mdarray_label("gkvec_"));

        /* do a first pass: determine boundaries of the grid */
        int xmin{0}, xmax{0};
        int ymin{0}, ymax{0};
        for (int i = 0; i < ngv_loc__; i++) {
            xmin = std::min(xmin, G(0, i));
            xmax = std::max(xmax, G(0, i));
            ymin = std::min(ymin, G(1, i));
            ymax = std::max(ymax, G(1, i));
        }
        comm_.allreduce<int, mpi::op_t::min>(&xmin, 1);
        comm_.allreduce<int, mpi::op_t::min>(&ymin, 1);
        comm_.allreduce<int, mpi::op_t::max>(&xmax, 1);
        comm_.allreduce<int, mpi::op_t::max>(&ymax, 1);

        mdarray<int, 2> zcol({index_range(xmin, xmax + 1), index_range(ymin, ymax + 1)});
        zcol.zero();
        for (int ig = 0; ig < ngv_loc__; ig++) {
            zcol(G(0, ig), G(1, ig))++;
            for (int x : {0, 1, 2}) {
                gvec_(x, ig)  = G(x, ig);
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

        gvec_distr_ = mpi::block_data_descriptor(comm().size());
        comm().allgather(&count_, gvec_distr_.counts.data(), 1, comm_.rank());
        gvec_distr_.calc_offsets();
        offset_ = gvec_distr_.offsets[comm().rank()];

        zcol_distr_ = mpi::block_data_descriptor(comm().size());
        comm().allgather(&num_zcol_local_, zcol_distr_.counts.data(), 1, comm_.rank());
        zcol_distr_.calc_offsets();

        num_gvec_ = count_;
        comm().allreduce(&num_gvec_, 1);
    }

    /// Constructor for empty set of G-vectors.
    Gvec(mpi::Communicator const& comm__)
        : comm_(comm__)
    {
    }

    /// Move assignment operator.
    Gvec&
    operator=(Gvec&& src__) = default;

    /// Move constructor.
    Gvec(Gvec&& src__) = default;

    inline auto const&
    vk() const
    {
        return vk_;
    }

    inline mpi::Communicator const&
    comm() const
    {
        return comm_;
    }

    /// Set the new reciprocal lattice vectors.
    /** For the varibale-cell relaxation runs we need an option to preserve the number of G- and G+k vectors.
     *  Here we can set the new lattice vectors and update the relevant members of the Gvec class. */
    inline auto const&
    lattice_vectors(r3::matrix<double> lattice_vectors__)
    {
        lattice_vectors_ = lattice_vectors__;
        find_gvec_shells();
        init_gvec_cart_local();
        return lattice_vectors_;
    }

    /// Retrn a const reference to the reciprocal lattice vectors.
    inline auto const&
    lattice_vectors() const
    {
        return lattice_vectors_;
    }

    inline auto const
    unit_cell_lattice_vectors() const
    {
        double const twopi = 6.2831853071795864769;
        auto r             = r3::transpose(r3::inverse(lattice_vectors_)) * twopi;
        return r;
    }

    /// Return the volume of the real space unit cell that corresponds to the reciprocal lattice of G-vectors.
    inline double
    omega() const
    {
        double const twopi_pow3 = 248.050213442398561403810520537;
        return twopi_pow3 / std::abs(lattice_vectors().det());
    }

    /// Return the total number of G-vectors within the cutoff.
    inline int
    num_gvec() const
    {
        return num_gvec_;
    }

    /// Number of z-columns for a fine-grained distribution.
    inline int
    zcol_count(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return zcol_distr_.counts[rank__];
    }

    /// Offset in the global index of z-columns for a given rank.
    inline int
    zcol_offset(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return zcol_distr_.offsets[rank__];
    }

    /// Number of G-vectors for a fine-grained distribution.
    inline int
    count(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return gvec_distr_.counts[rank__];
    }

    /// Number of G-vectors for a fine-grained distribution for the current MPI rank.
    /** The \em count and \em offset are borrowed from the MPI terminology for data distribution. */
    inline int
    count() const
    {
        return count_;
    }

    /// Offset (in the global index) of G-vectors for a fine-grained distribution.
    inline int
    offset(int rank__) const
    {
        RTE_ASSERT(rank__ < comm().size());
        return gvec_distr_.offsets[rank__];
    }

    /// Offset (in the global index) of G-vectors for a fine-grained distribution for a current MPI rank.
    /** The \em count and \em offset are borrowed from the MPI terminology for data distribution. */
    inline int
    offset() const
    {
        return offset_;
    }

    /// Local starting index of G-vectors if G=0 is not counted.
    inline int
    skip_g0() const
    {
        return (comm().rank() == 0) ? 1 : 0;
    }

    inline auto
    global_index(gvec_index_t::local ig__) const
    {
        return gvec_index_t::global(ig__.get() + offset());
    }

    /// Return number of G-vector shells.
    inline int
    num_shells() const
    {
        return num_gvec_shells_;
    }

    /// Return global G vector in fractional coordinates.
    inline auto
    gvec(gvec_index_t::global ig__) const
    {
        return gvec_by_full_index(gvec_full_index_(ig__));
    }

    /// Return local G vector in fractional coordinates.
    inline auto
    gvec(gvec_index_t::local ig__) const
    {
        return r3::vector<int>(gvec_(0, ig__), gvec_(1, ig__), gvec_(2, ig__));
    }

    /// Return global G vector in Cartesian coordinates.
    inline auto
    gvec_cart(gvec_index_t::global ig__) const
    {
        auto G = this->gvec(ig__);
        return dot(lattice_vectors_, G);
    }

    /// Return local G vector in Cartesian coordinates.
    inline auto
    gvec_cart(gvec_index_t::local ig__) const
    {
        return r3::vector<double>(gvec_cart_(0, ig__), gvec_cart_(1, ig__), gvec_cart_(2, ig__));
    }

    /// Return global G+k vector in fractional coordinates.
    inline auto
    gkvec(gvec_index_t::global ig__) const
    {
        return this->gvec(ig__) + vk_;
    }

    /// Return local G+k vector in fractional coordinates.
    inline auto
    gkvec(gvec_index_t::local ig__) const
    {
        return r3::vector<double>(gkvec_(0, ig__), gkvec_(1, ig__), gkvec_(2, ig__));
    }

    /// Return global G+k vector in fractional coordinates.
    inline auto
    gkvec_cart(gvec_index_t::global ig__) const
    {
        auto Gk = this->gvec(ig__) + vk_;
        return dot(lattice_vectors_, Gk);
    }

    /// Return local G+k vector in fractional coordinates.
    inline auto
    gkvec_cart(gvec_index_t::local ig__) const
    {
        return r3::vector<double>(gkvec_cart_(0, ig__), gkvec_cart_(1, ig__), gkvec_cart_(2, ig__));
    }

    /// Return index of the G-vector shell by the G-vector index.
    inline int
    shell(int ig__) const
    {
        return gvec_shell_(ig__);
    }

    inline int
    shell(r3::vector<int> const& G__) const
    {
        return this->shell(index_by_gvec(G__));
    }

    /// Return length of the G-vector shell.
    inline double
    shell_len(int igs__) const
    {
        return gvec_shell_len_(igs__);
    }

    /// Get lengths of all G-vector shells.
    inline auto
    shells_len() const
    {
        std::vector<double> q(this->num_shells());
        for (int i = 0; i < this->num_shells(); i++) {
            q[i] = this->shell_len(i);
        }
        return q;
    }

    /// Return length of global G-vector.
    inline auto
    gvec_len(gvec_index_t::global ig__) const
    {
        return gvec_shell_len_(gvec_shell_(ig__));
    }

    /// Return length of local G-vector.
    inline auto
    gvec_len(gvec_index_t::local ig__) const
    {
        return gvec_len_(ig__);
    }

    inline int
    index_g12(r3::vector<int> const& g1__, r3::vector<int> const& g2__) const
    {
        auto v  = g1__ - g2__;
        int idx = index_by_gvec(v);
        RTE_ASSERT(idx >= 0);
        RTE_ASSERT(idx < num_gvec());
        return idx;
    }

    std::pair<int, bool>
    index_g12_safe(r3::vector<int> const& g1__, r3::vector<int> const& g2__) const;

    // inline int index_g12_safe(int ig1__, int ig2__) const
    //{
    //     STOP();
    //     return 0;
    // }

    /// Return a global G-vector index in the range [0, num_gvec) by the G-vector.
    /** The information about a G-vector index is encoded by two numbers: a starting index for the
     *  column of G-vectors and column's size. Depending on the geometry of the reciprocal lattice,
     *  z-columns may have only negative, only positive or both negative and positive frequencies for
     *  a given x and y. This information is used to compute the offset which is added to the starting index
     *  in order to get a full G-vector index. Check find_z_columns() to see how the z-columns are found and
     *  added to the list of columns. */
    int
    index_by_gvec(r3::vector<int> const& G__) const;

    inline bool
    reduced() const
    {
        return reduce_gvec_;
    }

    inline bool
    bare() const
    {
        return bare_gvec_;
    }

    /// Return global number of z-columns.
    inline int
    num_zcol() const
    {
        return static_cast<int>(z_columns_.size());
    }

    /// Return local number of z-columns.
    inline int
    num_zcol_local() const
    {
        return num_zcol_local_;
    }

    inline z_column_descriptor const&
    zcol(size_t idx__) const
    {
        return z_columns_[idx__];
    }

    inline int
    gvec_base_mapping(int igloc_base__) const
    {
        RTE_ASSERT(gvec_base_ != nullptr);
        return gvec_base_mapping_(igloc_base__);
    }

    inline int
    num_gvec_shells_local() const
    {
        return num_gvec_shells_local_;
    }

    inline double
    gvec_shell_len_local(int idx__) const
    {
        return gvec_shell_len_local_[idx__];
    }

    inline int
    gvec_shell_idx_local(int igloc__) const
    {
        return gvec_shell_idx_local_[igloc__];
    }

    /// Return local list of G-vectors.
    inline auto const&
    gvec_local() const
    {
        return gvec_;
    }

    /// Return local list of G-vectors for a given rank.
    /** This function must be called by all MPI ranks of the G-vector communicator. */
    inline auto
    gvec_local(int rank__) const
    {
        int ngv = this->count();
        this->comm().bcast(&ngv, 1, rank__);
        mdarray<int, 2> result({3, ngv});
        if (this->comm().rank() == rank__) {
            RTE_ASSERT(ngv == this->count());
            copy(this->gvec_, result);
        }
        this->comm().bcast(&result(0, 0), 3 * ngv, rank__);
        return result;
    }

    inline auto&
    gvec_tp()
    {
        return gvec_tp_;
    }

    inline auto const&
    gvec_tp() const
    {
        return gvec_tp_;
    }

    inline auto&
    gkvec_tp()
    {
        return gkvec_tp_;
    }

    inline auto const&
    gkvec_tp() const
    {
        return gkvec_tp_;
    }
};

/// Stores information about G-vector partitioning between MPI ranks for the FFT transformation.
/** FFT driver works with a small communicator. G-vectors are distributed over the entire communicator which is
    larger than the FFT communicator. In order to transform the functions, G-vectors must be redistributed to the
    FFT-friendly "fat" slabs based on the FFT communicator size. */
class Gvec_fft
{
  private:
    /// Reference to the G-vector instance.
    Gvec const& gvec_;

    /// Communicator for the FFT.
    mpi::Communicator const& comm_fft_;

    /// Communicator which is orthogonal to FFT communicator.
    mpi::Communicator const& comm_ortho_fft_;

    /// Distribution of G-vectors for FFT.
    mpi::block_data_descriptor gvec_distr_fft_;

    /// Local number of z-columns.
    int num_zcol_local_{0};

    /// Distribution of G-vectors inside FFT-friendly "fat" slab.
    mpi::block_data_descriptor gvec_fft_slab_;

    /// Mapping of MPI ranks used to split G-vectors to a 2D grid.
    mdarray<int, 2> rank_map_;

    /// Lattice coordinates of a local set of G-vectors.
    /** These are also known as Miller indices */
    mdarray<int, 2> gvec_array_;

    /// Cartesian coordinaes of a local set of G+k-vectors.
    mdarray<double, 2> gkvec_cart_array_;

    void
    build_fft_distr();

    /// Stack together the G-vector slabs to make a larger ("fat") slab for a FFT driver.
    void
    pile_gvec();

  public:
    Gvec_fft(Gvec const& gvec__, mpi::Communicator const& fft_comm__, mpi::Communicator const& comm_ortho_fft__);

    /// Return FFT communicator
    inline mpi::Communicator const&
    comm_fft() const
    {
        return comm_fft_;
    }

    /// Return a communicator that is orthogonal to the FFT communicator.
    inline mpi::Communicator const&
    comm_ortho_fft() const
    {
        return comm_ortho_fft_;
    }

    /// Local number of G-vectors in the FFT distribution for a given rank.
    inline int
    count(int rank__) const
    {
        return gvec_distr_fft_.counts[rank__];
    }

    /// Local number of G-vectors for FFT-friendly distribution for this rank.
    inline int
    count() const
    {
        return this->count(comm_fft().rank());
    }

    /// Return local number of z-columns.
    inline int
    zcol_count() const
    {
        return num_zcol_local_;
    }

    /// Represents a "fat" slab of G-vectors in the FFT-friendly distribution.
    inline auto const&
    gvec_slab() const
    {
        return gvec_fft_slab_;
    }

    /// Return the original (not reshuffled) G-vector class.
    inline Gvec const&
    gvec() const
    {
        return gvec_;
    }

    /// Return the Cartesian coordinates of the local G-vector.
    inline auto
    gkvec_cart(int igloc__) const
    {
        return r3::vector<double>(&gkvec_cart_array_(0, igloc__));
    }

    /// Return the full array of the local G-vector Cartesian coodinates.
    inline auto const&
    gvec_array() const
    {
        return gvec_array_;
    }

    template <typename T> // TODO: document
    void
    gather_pw_fft(std::complex<T> const* f_pw_local__, std::complex<T>* f_pw_fft__) const
    {
        int rank = gvec().comm().rank();
        /* collect scattered PW coefficients */
        comm_ortho_fft().allgather(f_pw_local__, gvec().count(rank), f_pw_fft__, gvec_slab().counts.data(),
                                   gvec_slab().offsets.data());
    }

    template <typename T> // TODO: document
    void
    gather_pw_global(std::complex<T> const* f_pw_fft__, std::complex<T>* f_pw_global__) const
    {
        for (int ig = 0; ig < gvec().count(); ig++) {
            /* position inside fft buffer */
            int ig1                             = gvec_slab().offsets[comm_ortho_fft().rank()] + ig;
            f_pw_global__[gvec().offset() + ig] = f_pw_fft__[ig1];
        }
        gvec().comm().allgather(&f_pw_global__[0], gvec().count(), gvec().offset());
    }

    template <typename T>
    void
    scatter_pw_global(std::complex<T> const* f_pw_global__, std::complex<T>* f_pw_fft__) const
    {
        for (int i = 0; i < comm_ortho_fft_.size(); i++) {
            /* offset in global index */
            int offset = this->gvec_.offset(rank_map_(comm_fft_.rank(), i));
            for (int ig = 0; ig < gvec_fft_slab_.counts[i]; ig++) {
                f_pw_fft__[gvec_fft_slab_.offsets[i] + ig] = f_pw_global__[offset + ig];
            }
        }
    }

    /// Update Cartesian coordinates after a change in lattice vectors.
    void
    update_gkvec_cart()
    {
        for (int ig = 0; ig < this->count(); ig++) {
            auto G   = r3::vector<int>(&gvec_array_(0, ig));
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
    mpi::block_data_descriptor a2a_send_;

    /// Receiving counts and offsets.
    mpi::block_data_descriptor a2a_recv_;

    /// Split global index of G-shells between MPI ranks.
    splindex_block_cyclic<> spl_num_gsh_;

    /// List of G-vectors in the remapped storage.
    mdarray<int, 2> gvec_remapped_;

    /// Mapping between index of local G-vector and global index of G-vector shell.
    mdarray<int, 1> gvec_shell_remapped_;

    /// Alias for the G-vector communicator.
    mpi::Communicator const& comm_;

    Gvec const& gvec_;

    /// A mapping between G-vector and it's local index in the new distribution.
    std::map<r3::vector<int>, int> idx_gvec_;

  public:
    Gvec_shells(Gvec const& gvec__);

    inline void
    print_gvec(std::ostream& out__) const
    {
        mpi::pstdout pout(gvec_.comm());
        pout << "rank: " << gvec_.comm().rank() << std::endl;
        pout << "-- list of G-vectors in the remapped distribution --" << std::endl;
        for (int igloc = 0; igloc < gvec_count_remapped(); igloc++) {
            auto G = gvec_remapped(igloc);

            int igsh = gvec_shell_remapped(igloc);
            pout << "igloc=" << igloc << " igsh=" << igsh << " G=" << G[0] << " " << G[1] << " " << G[2] << std::endl;
        }
        pout << "-- reverse list --" << std::endl;
        for (auto const& e : idx_gvec_) {
            pout << "G=" << e.first[0] << " " << e.first[1] << " " << e.first[2] << ", igloc=" << e.second << std::endl;
        }
        out__ << pout.flush(0);
    }

    /// Local number of G-vectors in the remapped distribution with complete shells on each rank.
    int
    gvec_count_remapped() const
    {
        return a2a_recv_.size();
    }

    /// G-vector by local index (in the remapped set).
    r3::vector<int>
    gvec_remapped(int igloc__) const
    {
        return r3::vector<int>(gvec_remapped_(0, igloc__), gvec_remapped_(1, igloc__), gvec_remapped_(2, igloc__));
    }

    /// Return local index of the G-vector in the remapped set.
    int
    index_by_gvec(r3::vector<int> G__) const
    {
        if (idx_gvec_.count(G__)) {
            return idx_gvec_.at(G__);
        } else {
            return -1;
        }
    }

    /// Index of the G-vector shell by the local G-vector index (in the remapped set).
    int
    gvec_shell_remapped(int igloc__) const
    {
        return gvec_shell_remapped_(igloc__);
    }

    template <typename T>
    auto
    remap_forward(T* data__) const
    {
        PROFILE("fft::Gvec_shells::remap_forward");

        std::vector<T> send_buf(gvec_.count());
        std::vector<int> counts(comm_.size(), 0);
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig                                     = gvec_.offset() + igloc;
            int igsh                                   = gvec_.shell(ig);
            int r                                      = spl_num_gsh_.location(igsh).ib;
            send_buf[a2a_send_.offsets[r] + counts[r]] = data__[igloc];
            counts[r]++;
        }

        std::vector<T> recv_buf(gvec_count_remapped());

        comm_.alltoall(send_buf.data(), a2a_send_.counts.data(), a2a_send_.offsets.data(), recv_buf.data(),
                       a2a_recv_.counts.data(), a2a_recv_.offsets.data());

        return recv_buf;
    }

    template <typename T>
    void
    remap_backward(std::vector<T> buf__, T* data__) const
    {
        PROFILE("fft::Gvec_shells::remap_backward");

        std::vector<T> recv_buf(gvec_.count());

        comm_.alltoall(buf__.data(), a2a_recv_.counts.data(), a2a_recv_.offsets.data(), recv_buf.data(),
                       a2a_send_.counts.data(), a2a_send_.offsets.data());

        std::vector<int> counts(comm_.size(), 0);
        for (int igloc = 0; igloc < gvec_.count(); igloc++) {
            int ig        = gvec_.offset() + igloc;
            int igsh      = gvec_.shell(ig);
            int r         = spl_num_gsh_.location(igsh).ib;
            data__[igloc] = recv_buf[a2a_send_.offsets[r] + counts[r]];
            counts[r]++;
        }
    }

    inline Gvec const&
    gvec() const
    {
        return gvec_;
    }
};

/// This is only for debug purpose.
inline std::shared_ptr<Gvec>
gkvec_factory(double gk_cutoff__, mpi::Communicator const& comm__)
{
    auto M = r3::matrix<double>({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    return std::make_shared<Gvec>(r3::vector<double>({0, 0, 0}), M, gk_cutoff__, comm__, false);
}

inline std::shared_ptr<Gvec>
gkvec_factory(r3::vector<double> vk__, r3::matrix<double> reciprocal_lattice_vectors__, double gk_cutoff__,
              mpi::Communicator const& comm__ = mpi::Communicator::self(), bool gamma__ = false)
{
    return std::make_shared<Gvec>(vk__, reciprocal_lattice_vectors__, gk_cutoff__, comm__, gamma__);
}

inline void
print(std::ostream& out__, Gvec const& gvec__)
{
    std::map<int, std::vector<int>> gsh_map;
    for (int i = 0; i < gvec__.num_gvec(); i++) {
        int igsh = gvec__.shell(i);
        if (gsh_map.count(igsh) == 0) {
            gsh_map[igsh] = std::vector<int>();
        }
        gsh_map[igsh].push_back(i);
    }

    out__ << "num_gvec : " << gvec__.num_gvec() << std::endl;
    out__ << "num_gvec_shells : " << gvec__.num_shells() << std::endl;

    for (int igsh = 0; igsh < gvec__.num_shells(); igsh++) {
        auto len = gvec__.shell_len(igsh);
        out__ << "shell : " << igsh << ", length : " << len << std::endl;
        for (auto ig : gsh_map[igsh]) {
            auto G  = gvec__.gvec(gvec_index_t::global(ig));
            auto Gc = gvec__.gvec_cart(gvec_index_t::global(ig));
            out__ << "  ig : " << ig << ", G = " << G << ", length diff : " << std::abs(Gc.length() - len) << std::endl;
        }
    }
    // mpi::pstdout pout(gvec__.comm());
    // pout << "rank: " << gvec_.comm().rank() << std::endl;
    // pout << "-- list of G-vectors in the remapped distribution --" << std::endl;
    // for (int igloc = 0; igloc < gvec_count_remapped(); igloc++) {
    //     auto G = gvec_remapped(igloc);

    //    int igsh = gvec_shell_remapped(igloc);
    //    pout << "igloc=" << igloc << " igsh=" << igsh << " G=" << G[0] << " " << G[1] << " " << G[2] << std::endl;
    //}
    // pout << "-- reverse list --" << std::endl;
    // for (auto const& e: idx_gvec_) {
    //    pout << "G=" << e.first[0] << " " << e.first[1] << " " << e.first[2] << ", igloc=" << e.second << std::endl;
    //}
    // out__ << pout.flush(0);
}

class gvec_iterator_t
{
  private:
    gvec_index_t::local igloc_{-1};
    gvec_index_t::value_type offset_{-1};

  public:
    using difference_type = std::ptrdiff_t;

    gvec_iterator_t(gvec_index_t::local igloc__, gvec_index_t::value_type offset__)
        : igloc_{igloc__}
        , offset_{offset__}
    {
    }

    gvec_iterator_t(gvec_index_t::local igloc__)
        : igloc_{igloc__}
    {
    }

    inline bool
    operator!=(gvec_iterator_t const& rhs__)
    {
        return this->igloc_ != rhs__.igloc_;
    }

    inline gvec_iterator_t&
    operator++()
    {
        this->igloc_++;
        return *this;
    }

    inline gvec_iterator_t
    operator++(int)
    {
        gvec_iterator_t tmp(this->igloc_);
        this->igloc_++;
        return tmp;
    }

    inline auto
    operator*()
    {
        struct
        {
            typename gvec_index_t::global ig;
            typename gvec_index_t::local igloc;
        } ret{gvec_index_t::global(this->offset_ + this->igloc_.get()), this->igloc_};
        return ret;
    }

    inline difference_type
    operator-(gvec_iterator_t const& rhs__) const
    {
        return this->igloc_ - rhs__.igloc_;
    }

    inline gvec_iterator_t&
    operator+=(difference_type rhs__)
    {
        this->igloc_ += rhs__;
        return *this;
    }
};

class gvec_skip_g0
{
  private:
    Gvec const& gv_;

  public:
    gvec_skip_g0(Gvec const& gv__)
        : gv_{gv__}
    {
    }
    friend auto
    begin(gvec_skip_g0 const&);
    friend auto
    end(gvec_skip_g0 const&);
};

inline auto
skip_g0(Gvec const& gv__)
{
    return gvec_skip_g0(gv__);
}

inline auto
begin(Gvec const& gv__)
{
    return gvec_iterator_t(gvec_index_t::local(0), gv__.offset());
}

inline auto
begin(gvec_skip_g0 const& gv__)
{
    return gvec_iterator_t(gvec_index_t::local(gv__.gv_.skip_g0()), gv__.gv_.offset());
}

inline auto
end(Gvec const& gv__)
{
    return gvec_iterator_t(gvec_index_t::local(gv__.count()));
}

inline auto
end(gvec_skip_g0 const& gv__)
{
    return gvec_iterator_t(gvec_index_t::local(gv__.gv_.count()));
}

} // namespace fft

} // namespace sirius

#endif //__GVEC_HPP__

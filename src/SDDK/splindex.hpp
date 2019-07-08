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

/** \file splindex.hpp
 *
 *  \brief Contains definition of sddk::splindex_base and specializations of sddk::splindex class.
 */

#ifndef __SPLINDEX_HPP__
#define __SPLINDEX_HPP__

#include <algorithm>
#include <vector>
#include <sstream>
#include <limits>
#include <cassert>

namespace sddk {

/// Type of split index.
enum class splindex_t
{
    /// Block distribution.
    block,
    /// Block-cyclic distribution.
    block_cyclic,
    /// Custom distribution in continuous chunks of arbitrary size.
    chunk
};

/// Type of index domain.
enum class index_domain_t
{
    /// Global index.
    global,
    /// Local index.
    local
};

/// Base class for split index.
template <typename T>
class splindex_base
{
  protected:
    /// Rank of the block with local fraction of the global index.
    int rank_{-1};

    /// Number of ranks over which the global index is distributed.
    int num_ranks_{-1};

    /// size of the global index
    T global_index_size_;

    /// Default constructor.
    splindex_base()
    {
    }

    /// Pair of <local index, rank> describing the location of a global index.
    struct location_t
    {
        T local_index;
        int rank;
        location_t(T local_index__, int rank__)
            : local_index(local_index__)
            , rank(rank__)
        {
        }
    };

  public:
    /// Rank id.
    inline int rank() const
    {
        return rank_;
    }

    /// Number of ranks that are participating in the distribution of an index.
    inline int num_ranks() const
    {
        return num_ranks_;
    }

    inline T global_index_size() const
    {
        return global_index_size_;
    }

    static inline T block_size(T size__, int num_ranks__)
    {
        return size__ / num_ranks__ + std::min(T(1), size__ % num_ranks__);
    }
};

/// Split index.
template <splindex_t type, typename T = int>
class splindex : public splindex_base<T>
{
};

/// Specialization for the block distribution.
template <typename T>
class splindex<splindex_t::block, T> : public splindex_base<T>
{
  private:
    T block_size_;

    void init(T global_index_size__, int num_ranks__, int rank__)
    {
        this->global_index_size_ = global_index_size__;

        if (num_ranks__ < 0) {
            std::stringstream s;
            s << "wrong number of ranks: " << num_ranks__;
            throw std::runtime_error(s.str());
        }
        this->num_ranks_ = num_ranks__;

        if (rank__ < 0 || rank__ >= num_ranks__) {
            std::stringstream s;
            s << "wrong rank: " << rank__;
            throw std::runtime_error(s.str());
        }
        this->rank_ = rank__;

        block_size_ = this->block_size(global_index_size__, num_ranks__);
    }

  public:
    /// Default constructor
    splindex()
    {
    }

    /// Constructor.
    splindex(T global_index_size__, int num_ranks__, int rank__)
    {
        init(global_index_size__, num_ranks__, rank__);
    }

    /// Return "local index, rank" pair for a global index.
    inline typename splindex_base<T>::location_t location(T idxglob__) const
    {
        assert(idxglob__ < this->global_index_size_);

        int rank = int(idxglob__ / block_size_);
        T idxloc = idxglob__ - rank * block_size_;

        return typename splindex_base<T>::location_t(idxloc, rank);
    }

    /// Return local size of the split index for an arbitrary rank.
    inline T local_size(int rank__) const
    {
        assert(rank__ >= 0 && rank__ < this->num_ranks_);

        if (this->global_index_size_ == 0) {
            return 0;
        }

        int n = static_cast<int>(this->global_index_size_ / block_size_);
        if (rank__ < n) {
            return block_size_;
        } else if (rank__ == n) {
            return this->global_index_size_ - rank__ * block_size_;
        } else {
            return 0;
        }
    }

    /// Return local size of the split index for a current rank.
    inline T local_size() const
    {
        return local_size(this->rank_);
    }

    /// Return rank which holds the element with the given global index.
    inline int local_rank(T idxglob__) const
    {
        return location(idxglob__).rank;
    }

    /// Return local index of the element for the rank which handles the given global index.
    inline T local_index(T idxglob__) const
    {
        return location(idxglob__).local_index;
    }

    /// Return global index of an element by local index and rank.
    inline T global_index(T idxloc__, int rank__) const
    {
        assert(rank__ >= 0 && rank__ < this->num_ranks_);

        if (local_size(rank__) == 0) {
            return std::numeric_limits<T>::max();
        }

        assert(idxloc__ < local_size(rank__));

        return rank__ * block_size_ + idxloc__;
    }

    inline T global_offset() const
    {
        return global_index(0, this->rank_);
    }

    inline T global_offset(int rank__) const
    {
        return global_index(0, rank__);
    }

    inline T operator[](T idxloc__) const
    {
        return global_index(idxloc__, this->rank_);
    }

    inline std::vector<T> offsets() const
    {
        std::vector<T> v(this->num_ranks_);
        for (int i = 0; i < this->num_ranks_; i++) {
            v[i] = global_offset(i);
        }
        return v;
    }

    inline std::vector<T> counts() const
    {
        std::vector<T> v(this->num_ranks_);
        for (int i = 0; i < this->num_ranks_; i++) {
            v[i] = local_size(i);
        }
        return v;
    }
};

/// Specialization for the block-cyclic distribution.
template <typename T>
class splindex<splindex_t::block_cyclic, T> : public splindex_base<T>
{
  private:
    /// cyclic block size of the distribution
    int block_size_{-1};

    // Check and initialize variables.
    void init(T global_index_size__, int num_ranks__, int rank__, int block_size__)
    {
        this->global_index_size_ = global_index_size__;

        if (num_ranks__ < 0) {
            std::stringstream s;
            s << "wrong number of ranks: " << num_ranks__;
            throw std::runtime_error(s.str());
        }
        this->num_ranks_ = num_ranks__;

        if (rank__ < 0 || rank__ >= num_ranks__) {
            std::stringstream s;
            s << "wrong rank: " << rank__;
            throw std::runtime_error(s.str());
        }
        this->rank_ = rank__;

        if (block_size__ <= 0) {
            std::stringstream s;
            s << "wrong block size: " << block_size__;
            throw std::runtime_error(s.str());
        }
        block_size_ = block_size__;
    }

  public:
    struct iterator
    {
        T idxloc_;
        T idxglob_;
        T num_blocks_min_;
        int block_size_;
        int rank_;
        int num_ranks_;

        iterator(T idxglob__, int num_ranks__, int block_size__)
            : idxglob_(idxglob__)
            , num_ranks_(num_ranks__)
            , block_size_(block_size__)
        {
            /* number of full blocks */
            T num_blocks = idxglob__ / block_size_;
            num_blocks_min_ = num_blocks / num_ranks_;
            idxloc_ = num_blocks_min_ * block_size_ + idxglob_ % block_size_;
            rank_ = static_cast<int>(num_blocks % num_ranks_);
        }

        bool operator!=(iterator const& rhs__) const
        {
            return idxglob_ != rhs__.idxglob_;
        }

        iterator& operator++()
        {
            idxglob_++;
            idxloc_++;
            if (idxloc_ % block_size_ == 0) {
                rank_++;
                if (rank_ % num_ranks_ == 0) {
                    num_blocks_min_++;
                    rank_ = 0;
                }
                idxloc_ = num_blocks_min_ * block_size_;// + idxglob_ % block_size_;
            }
        }
    };

    /// Default constructor
    splindex()
    {
    }

    /// Constructor with implicit cyclic block size
    splindex(T global_index_size__, int num_ranks__, int rank__, int bs__)
    {
        init(global_index_size__, num_ranks__, rank__, bs__);
    }

    iterator at(T idxglob__) const
    {
        return iterator(idxglob__, this->num_ranks_, this->block_size_);
    }

    /// Return "local index, rank" pair for a global index.
    inline typename splindex_base<T>::location_t location(T idxglob__) const
    {
        assert(idxglob__ < this->global_index_size_);

        /* number of full blocks */
        T num_blocks = idxglob__ / block_size_;

        /* local index */
        T idxloc = (num_blocks / this->num_ranks_) * block_size_ + idxglob__ % block_size_;

        /* corresponding rank */
        int rank = static_cast<int>(num_blocks % this->num_ranks_);

        return typename splindex_base<T>::location_t(idxloc, rank);
    }

    /// Return local size of the split index for an arbitrary rank.
    inline T local_size(int rank__) const
    {
        assert(rank__ >= 0 && rank__ < this->num_ranks_);

        /* number of full blocks */
        T num_blocks = this->global_index_size_ / block_size_;

        T n = (num_blocks / this->num_ranks_) * block_size_;

        int rank_offs = static_cast<int>(num_blocks % this->num_ranks_);

        if (rank__ < rank_offs) {
            n += block_size_;
        } else if (rank__ == rank_offs) {
            n += this->global_index_size_ % block_size_;
        }
        return n;
    }

    /// Return local size of the split index for a current rank.
    inline T local_size() const
    {
        return local_size(this->rank_);
    }

    /// Return rank which holds the element with the given global index.
    inline int local_rank(T idxglob__) const
    {
        return location(idxglob__).rank;
    }

    /// Return local index of the element for the rank which handles the given global index.
    inline T local_index(T idxglob__) const
    {
        return location(idxglob__).local_index;
    }

    /// Get a global index by local index of a rank.
    inline T global_index(T idxloc__, int rank__) const
    {
        assert(rank__ >= 0 && rank__ < this->num_ranks_);
        assert(idxloc__ < local_size(rank__));

        T nb = idxloc__ / block_size_;

        return (nb * this->num_ranks_ + rank__) * block_size_ + idxloc__ % block_size_;
    }

    /// Get global index of this rank.
    inline T operator[](T idxloc__) const
    {
        return global_index(idxloc__, this->rank_);
    }
};

/// Specialization for the block distribution.
template <typename T>
class splindex<splindex_t::chunk, T> : public splindex_base<T>
{
  private:
    std::vector<std::vector<T>> global_index_;
    std::vector<typename splindex_base<T>::location_t> locations_;

  public:
    /// Default constructor.
    splindex()
    {
    }

    /// Constructor with specific partitioning.
    splindex(T global_index_size__, int num_ranks__, int rank__, std::vector<T> const& counts__)
    {
        this->global_index_size_ = global_index_size__;

        if (num_ranks__ < 0) {
            std::stringstream s;
            s << "wrong number of ranks: " << num_ranks__;
            throw std::runtime_error(s.str());
        }
        this->num_ranks_ = num_ranks__;

        if (rank__ < 0 || rank__ >= num_ranks__) {
            std::stringstream s;
            s << "wrong rank: " << rank__;
            throw std::runtime_error(s.str());
        }
        this->rank_ = rank__;

        for (int r = 0; r < num_ranks__; r++) {
            global_index_.push_back(std::vector<T>());
            for (int i = 0; i < counts__[r]; i++) {
                global_index_.back().push_back(static_cast<T>(locations_.size()));
                locations_.push_back(typename splindex_base<T>::location_t(i, r));
            }
        }

        assert(static_cast<T>(locations_.size()) == global_index_size__);
    }

    inline T local_size(int rank__) const
    {
        assert(rank__ >= 0);
        assert(rank__ < this->num_ranks_);
        return static_cast<T>(global_index_[rank__].size());
    }

    inline T local_size() const
    {
        return local_size(this->rank_);
    }

    inline int local_rank(T idxglob__) const
    {
        return locations_[idxglob__].rank;
    }

    inline T local_index(T idxglob__) const
    {
        return locations_[idxglob__].local_index;
    }

    inline T global_index(T idxloc__, int rank__) const
    {
        if (local_size(rank__) == 0) {
            return std::numeric_limits<T>::max();
        }

        assert(idxloc__ < local_size(rank__));

        return global_index_[rank__][idxloc__];
    }

    inline T operator[](T idxloc__) const
    {
        return global_index(idxloc__, this->rank_);
    }

    inline T global_offset() const
    {
        return global_index(0, this->rank_);
    }
};

} // namespace sddk

#endif // __SPLINDEX_HPP__

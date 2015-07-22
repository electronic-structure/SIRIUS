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

/** \file splindex.hpp
 *  
 *  \brief Contains splindex template class specialization for block and block-cyclic data distributions.
 */

#include <limits>

/// Specialization for the block distribution.
template<> 
class splindex<block>: public splindex_base
{
    private:
        
        size_t block_size_;

        void init(size_t global_index_size__, int num_ranks__, int rank__)
        {
            global_index_size_ = global_index_size__;

            if (num_ranks__ < 0) error_local(__FILE__, __LINE__, "wrong number of ranks");
            num_ranks_ = num_ranks__;

            if (rank__ < 0 || rank__ >= num_ranks__) error_local(__FILE__, __LINE__, "wrong rank");
            rank_ = rank__;

            block_size_ = block_size(global_index_size__, num_ranks__);
        }
        
    public:
       
        /// Default constructor
        splindex()
        {
        }
        
        /// Constructor.
        splindex(size_t global_index_size__, int num_ranks__, int rank__)
        {
            init(global_index_size__, num_ranks__, rank__); 
        }

        /// Return "local index, rank" pair for a global index.
        inline std::pair<size_t, int> location(size_t idxglob__) const
        {
            assert(idxglob__ < global_index_size_);

            int rank = int(idxglob__ / block_size_);
            size_t idxloc = idxglob__ - rank * block_size_;

            return std::pair<size_t, int>(idxloc, rank);
        }

        /// Return local size of the split index for an arbitrary rank.
        inline size_t local_size(int rank__) const
        {
            assert(rank__ >= 0 && rank__ < num_ranks_);
            
            int n = int(global_index_size_ / block_size_);
            if (rank__ < n)
            {
                return block_size_;
            }
            else if (rank__ == n)
            {
                return global_index_size_ - rank__ * block_size_;
            }
            else return 0;
        }

        /// Return local size of the split index for a current rank.
        inline size_t local_size() const
        {
            return local_size(rank_);
        }
        
        /// Return rank which holds the element with the given global index.
        inline int local_rank(size_t idxglob__) const
        {
            return location(idxglob__).second;
        }
        
        /// Return local index of the element for the rank which handles the given global index.
        inline size_t local_index(size_t idxglob__) const
        {
            return location(idxglob__).first;
        }
        
        /// Return global index of an element by local index and rank.
        inline size_t global_index(size_t idxloc__, int rank__) const
        {
            assert(rank__ >= 0 && rank__ < num_ranks_);

            if (local_size(rank__) == 0) return std::numeric_limits<size_t>::max();

            assert(idxloc__ < local_size(rank__));

            return rank__ * block_size_ + idxloc__;
        }

        inline size_t global_offset() const
        {
            return global_index(0, rank_);
        }

        inline size_t global_offset(int rank__) const
        {
            return global_index(0, rank__);
        }

        inline size_t operator[](size_t idxloc__) const
        {
            return global_index(idxloc__, rank_);
        }

        inline std::vector<int> offsets() const
        {
            std::vector<int> v(num_ranks_);
            for (int i = 0; i < num_ranks_; i++) v[i] = (int)global_offset(i);
            return v;
        }

        inline std::vector<int> counts() const
        {
            std::vector<int> v(num_ranks_);
            for (int i = 0; i < num_ranks_; i++) v[i] = (int)local_size(i);
            return v;
        }
};

/// Specialization for the block-cyclic distribution.
template<> 
class splindex<block_cyclic>: public splindex_base
{
    private:

        /// cyclic block size of the distribution
        int block_size_;

        // Check and initialize variables.
        void init(size_t global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            global_index_size_ = global_index_size__;

            if (num_ranks__ < 0) error_local(__FILE__, __LINE__, "wrong number of ranks");
            num_ranks_ = num_ranks__;

            if (rank__ < 0 || rank__ >= num_ranks__) error_local(__FILE__, __LINE__, "wrong rank");
            rank_ = rank__;

            if (block_size__ <= 0) error_local(__FILE__, __LINE__, "wrong block size");
            block_size_ = block_size__;
        }

    public:
        
        /// Default constructor
        splindex() : block_size_(-1)
        {
        }
        
        /// Constructor with implicit cyclic block size
        splindex(size_t global_index_size__, int num_ranks__, int rank__, int bs__)
        {
            init(global_index_size__, num_ranks__, rank__, bs__); 
        }

        /// Return "local index, rank" pair for a global index.
        inline std::pair<size_t, int> location(size_t idxglob__) const
        {
            assert(idxglob__ < global_index_size_);
            
            /* number of full blocks */
            size_t num_blocks = idxglob__ / block_size_;

            /* local index */
            size_t idxloc = (num_blocks / num_ranks_) * block_size_ + idxglob__ % block_size_;
            
            /* corresponding rank */
            int rank = static_cast<int>(num_blocks % num_ranks_);

            return std::pair<size_t, int>(idxloc, rank);
        }

        /// Return local size of the split index for an arbitrary rank.
        inline size_t local_size(int rank__) const
        {
            assert(rank__ >= 0 && rank__ < num_ranks_);
            
            /* number of full blocks */
            size_t num_blocks = global_index_size_ / block_size_;

            size_t n = (num_blocks / num_ranks_) * block_size_;

            int rank_offs = static_cast<int>(num_blocks % num_ranks_);

            if (rank__ < rank_offs) 
            {
                n += block_size_;
            }
            else if (rank__ == rank_offs)
            {
                n += global_index_size_ % block_size_;
            }
            return n;
        }

        /// Return local size of the split index for a current rank.
        inline size_t local_size() const
        {
            return local_size(rank_);
        }

        /// Return rank which holds the element with the given global index.
        inline int local_rank(size_t idxglob__) const
        {
            return location(idxglob__).second;
        }
        
        /// Return local index of the element for the rank which handles the given global index.
        inline size_t local_index(size_t idxglob__) const
        {
            return location(idxglob__).first;
        }

        inline size_t global_index(size_t idxloc__, int rank__) const
        {
            assert(rank__ >= 0 && rank__ < num_ranks_);
            assert(idxloc__ < local_size(rank__));

            size_t nb = idxloc__ / block_size_;
            
            return (nb * num_ranks_ + rank__) * block_size_ + idxloc__ % block_size_;
        }

        inline size_t operator[](size_t idxloc__) const
        {
            return global_index(idxloc__, rank_);
        }
};

template<> 
class splindex<dyadic>
{
    private:
        splindex<block_cyclic> spl1_;
        splindex<block> spl2_;
    
    public:
        splindex()
        {
        }

        splindex(size_t global_index_size__, int num_ranks1__, int rank1__, int bs__, int num_ranks2__, int rank2__)
        {
            spl1_ = splindex<block_cyclic>(global_index_size__, num_ranks1__, rank1__, bs__);
            spl2_ = splindex<block>(spl1_.local_size(), num_ranks2__, rank2__);
        }

        inline size_t local_size() const
        {
            return spl2_.local_size();
        }

        inline size_t operator[](size_t idxloc2__) const
        {
            return spl2_[idxloc2__];
        }
};

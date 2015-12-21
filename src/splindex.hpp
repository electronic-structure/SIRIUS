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
template<typename T>
class splindex<block, T>: public splindex_base<T>
{
    private:
        
        T block_size_;

        void init(T global_index_size__, int num_ranks__, int rank__)
        {
            this->global_index_size_ = global_index_size__;

            if (num_ranks__ < 0)
            {
                std::stringstream s;
                s << "wrong number of ranks: " << num_ranks__;
                TERMINATE(s);
            }
            this->num_ranks_ = num_ranks__;

            if (rank__ < 0 || rank__ >= num_ranks__)
            {
                std::stringstream s;
                s << "wrong rank: " << rank__;
                TERMINATE(s);
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
        inline std::pair<T, int> location(T idxglob__) const
        {
            assert(idxglob__ < this->global_index_size_);

            int rank = int(idxglob__ / block_size_);
            T idxloc = idxglob__ - rank * block_size_;

            return std::pair<T, int>(idxloc, rank);
        }

        /// Return local size of the split index for an arbitrary rank.
        inline T local_size(int rank__) const
        {
            assert(rank__ >= 0 && rank__ < this->num_ranks_);
            
            int n = int(this->global_index_size_ / block_size_);
            if (rank__ < n)
            {
                return block_size_;
            }
            else if (rank__ == n)
            {
                return this->global_index_size_ - rank__ * block_size_;
            }
            else return 0;
        }

        /// Return local size of the split index for a current rank.
        inline T local_size() const
        {
            return local_size(this->rank_);
        }
        
        /// Return rank which holds the element with the given global index.
        inline int local_rank(T idxglob__) const
        {
            return location(idxglob__).second;
        }
        
        /// Return local index of the element for the rank which handles the given global index.
        inline T local_index(T idxglob__) const
        {
            return location(idxglob__).first;
        }
        
        /// Return global index of an element by local index and rank.
        inline T global_index(T idxloc__, int rank__) const
        {
            assert(rank__ >= 0 && rank__ < this->num_ranks_);

            if (local_size(rank__) == 0) return std::numeric_limits<T>::max();

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
            for (int i = 0; i < this->num_ranks_; i++) v[i] = global_offset(i);
            return v;
        }

        inline std::vector<T> counts() const
        {
            std::vector<T> v(this->num_ranks_);
            for (int i = 0; i < this->num_ranks_; i++) v[i] = local_size(i);
            return v;
        }
};

/// Specialization for the block-cyclic distribution.
template<typename T>
class splindex<block_cyclic, T>: public splindex_base<T>
{
    private:

        /// cyclic block size of the distribution
        int block_size_;

        // Check and initialize variables.
        void init(T global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            this->global_index_size_ = global_index_size__;

            if (num_ranks__ < 0)
            {
                std::stringstream s;
                s << "wrong number of ranks: " << num_ranks__;
                TERMINATE(s);
            }
            this->num_ranks_ = num_ranks__;

            if (rank__ < 0 || rank__ >= num_ranks__)
            {
                std::stringstream s;
                s << "wrong rank: " << rank__;
                TERMINATE(s);
            }
            this->rank_ = rank__;

            if (block_size__ <= 0)
            {
                std::stringstream s;
                s << "wrong block size: " << block_size__;
                TERMINATE(s);
            }
            block_size_ = block_size__;
        }

    public:
        
        /// Default constructor
        splindex() : block_size_(-1)
        {
        }
        
        /// Constructor with implicit cyclic block size
        splindex(T global_index_size__, int num_ranks__, int rank__, int bs__)
        {
            init(global_index_size__, num_ranks__, rank__, bs__); 
        }

        /// Return "local index, rank" pair for a global index.
        inline std::pair<T, int> location(T idxglob__) const
        {
            assert(idxglob__ < this->global_index_size_);
            
            /* number of full blocks */
            T num_blocks = idxglob__ / block_size_;

            /* local index */
            T idxloc = (num_blocks / this->num_ranks_) * block_size_ + idxglob__ % block_size_;
            
            /* corresponding rank */
            int rank = static_cast<int>(num_blocks % this->num_ranks_);

            return std::pair<T, int>(idxloc, rank);
        }

        /// Return local size of the split index for an arbitrary rank.
        inline T local_size(int rank__) const
        {
            assert(rank__ >= 0 && rank__ < this->num_ranks_);
            
            /* number of full blocks */
            T num_blocks = this->global_index_size_ / block_size_;

            T n = (num_blocks / this->num_ranks_) * block_size_;

            int rank_offs = static_cast<int>(num_blocks % this->num_ranks_);

            if (rank__ < rank_offs) 
            {
                n += block_size_;
            }
            else if (rank__ == rank_offs)
            {
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
            return location(idxglob__).second;
        }
        
        /// Return local index of the element for the rank which handles the given global index.
        inline T local_index(T idxglob__) const
        {
            return location(idxglob__).first;
        }

        inline T global_index(T idxloc__, int rank__) const
        {
            assert(rank__ >= 0 && rank__ < this->num_ranks_);
            assert(idxloc__ < local_size(rank__));

            T nb = idxloc__ / block_size_;
            
            return (nb * this->num_ranks_ + rank__) * block_size_ + idxloc__ % block_size_;
        }

        inline T operator[](T idxloc__) const
        {
            return global_index(idxloc__, this->rank_);
        }
};

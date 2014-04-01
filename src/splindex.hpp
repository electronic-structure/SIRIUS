// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

// TODO: size_t instead of int

/// Specialization for the block distribution.
template<> 
class splindex<block>: public splindex_base
{
    private:

        int min_num_elements_;
        int num_ranks_with_extra_element_;
        int num_prime_chunk_; 

        void init(int global_index_size__, int num_ranks__, int rank__)
        {
            if (global_index_size__ < 0) error_local(__FILE__, __LINE__, "wrong global index size");
            global_index_size_ = global_index_size__;

            if (num_ranks__ < 0) error_local(__FILE__, __LINE__, "wrong number of ranks");
            num_ranks_ = num_ranks__;

            if (rank__ < 0 || rank__ >= num_ranks__) error_local(__FILE__, __LINE__, "wrong rank");
            rank_ = rank__;

            min_num_elements_ = global_index_size_ / num_ranks_;

            num_ranks_with_extra_element_ = global_index_size_ % num_ranks_; 

            num_prime_chunk_ = (min_num_elements_ + 1) * num_ranks_with_extra_element_;
        }
        
    public:
       
        /// Default constructor
        splindex()
        {
        }
        
        splindex(int global_index_size__, int num_ranks__, int rank__)
        {
            init(global_index_size__, num_ranks__, rank__); 
        }

        inline int local_size()
        {
            return local_size(rank_);
        }

        inline int local_size(int rank)
        {
            assert(rank >= 0 && rank < num_ranks_);
            return min_num_elements_ + (rank < num_ranks_with_extra_element_ ? 1 : 0); // minimum number of elements +1 if rank < m
        }

        inline std::pair<int, int> location(int idxglob)
        {
            assert(idxglob >= 0 && idxglob < global_index_size_);

            int rank;
            int offs;

            if (idxglob < num_prime_chunk_)
            {
                rank = idxglob / (min_num_elements_ + 1);
                offs = idxglob % (min_num_elements_ + 1);
            }
            else
            {
                assert(min_num_elements_ != 0);

                int k = idxglob - num_prime_chunk_;
                offs = k % min_num_elements_;
                rank = num_ranks_with_extra_element_ + k / min_num_elements_;
            }

            return std::pair<int, int>(offs, rank);
        }
        
        inline int location(int offset_or_rank, int idxglob) // TODO: rename
        {
            switch (offset_or_rank)
            {
                case _splindex_offs_:
                {
                    return location(idxglob).first;
                    break;
                }
                case _splindex_rank_:
                {
                    return location(idxglob).second;
                    break;
                }
                default:
                {
                    return -1;
                }
            }
        }
        
        /// Return global index of an element by local index and rank.
        inline int global_index(int idxloc, int rank)
        {
            assert(rank >= 0 && rank < num_ranks_);

            if (local_size(rank) == 0) return -1;

            assert(idxloc >= 0 && idxloc < local_size(rank));

            return (rank < num_ranks_with_extra_element_) ? (min_num_elements_+ 1) * rank + idxloc : 
                rank * min_num_elements_ + num_ranks_with_extra_element_ + idxloc; 
        }

        inline int global_offset()
        {
            return global_index(0, rank_);
        }

        inline int global_offset(int rank__)
        {
            return global_index(0, rank__);
        }

        inline int operator[](int idxloc)
        {
            return global_index(idxloc, rank_);
        }
};

/// Specialization for the block-cyclic distribution.
template<> 
class splindex<block_cyclic>: public splindex_base
{
    private:

        static int cyclic_block_size_;
        
        /// cyclic block size of the distribution
        int block_size_;

        // Check and initialize variables.
        void init(int global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            if (global_index_size__ <= 0) error_local(__FILE__, __LINE__, "wrong global index size");
            global_index_size_ = global_index_size__;

            if (num_ranks__ < 0) error_local(__FILE__, __LINE__, "wrong number of ranks");
            num_ranks_ = num_ranks__;

            if (rank__ < 0 || rank__ >= num_ranks__) error_local(__FILE__, __LINE__, "wrong rank");
            rank_ = rank__;

            if (block_size__ <= 0) error_local(__FILE__, __LINE__, "wrong block size");
            block_size_ = block_size__;
        }

    public:
        
        // Default constructor
        splindex() : block_size_(-1)
        {
        }
        
        // Constructor with implicit cyclic block size
        splindex(int global_index_size__, int num_ranks__, int rank__)
        {
            int bs = cyclic_block_size_;
            init(global_index_size__, num_ranks__, rank__, bs); 
        }

        static void set_cyclic_block_size(int cyclic_block_size__)
        {
            cyclic_block_size_ = cyclic_block_size__;
        }

        inline int local_size(int rank)
        {
            assert(rank >= 0 && rank < num_ranks_);

            int num_blocks = global_index_size_ / block_size_; // number of full blocks

            int n = (num_blocks / num_ranks_) * block_size_;

            int rank_offs = num_blocks % num_ranks_;

            if (rank < rank_offs) 
            {
                n += block_size_;
            }
            else if (rank == rank_offs)
            {
                n += global_index_size_ % block_size_;
            }
            return n;
        }

        inline int local_size()
        {
            return local_size(rank_);
        }

        inline int location(int offset_or_rank, int idxglob) // TODO: rename
        {
            assert(idxglob >= 0 && idxglob < global_index_size_);

            int num_blocks = idxglob / block_size_; // number of full blocks

            int n = (num_blocks / num_ranks_) * block_size_ + idxglob % block_size_;

            int rank = num_blocks % num_ranks_;

            switch (offset_or_rank)
            {
                case _splindex_offs_:
                {
                    return n;
                    break;
                }
                case _splindex_rank_:
                {
                    return rank;
                    break;
                }
            }
            return -1; // make compiler happy
        }

        inline std::pair<int, int> location(int idx_glob)
        {
            assert(idx_glob >= 0 && idx_glob < global_index_size_);

            int num_blocks = idx_glob / block_size_; // number of full blocks

            int offs = (num_blocks / num_ranks_) * block_size_ + idx_glob % block_size_;

            int rank = num_blocks % num_ranks_;

            return std::pair<int, int>(offs, rank);
        }

        inline int global_index(int idxloc, int rank)
        {
            assert(rank >= 0 && rank < num_ranks_);
            assert(idxloc >= 0 && idxloc < local_size(rank));

            int nb = idxloc / block_size_;
            
            int idx = (nb * num_ranks_ + rank) * block_size_ + idxloc % block_size_;
            return idx;
        }

        inline int operator[](int idxloc)
        {
            return global_index(idxloc, rank_);
        }
};


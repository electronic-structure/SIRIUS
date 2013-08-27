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

#ifndef __SPLINDEX_H__
#define __SPLINDEX_H__

/** \file splindex.h
    
    \brief Split index implementation
*/

const int _splindex_offs_ = 0;
const int _splindex_rank_ = 1;

class splindex_base
{
    private:
        
        /// forbid copy constructor
        splindex_base(const splindex_base& src);

        /// forbid assigment operator
        splindex_base& operator=(const splindex_base& src);
    
    protected:
        
        /// rank of the block with local fraction of the global index
        int rank_;
        
        /// number of ranks over which the global index is distributed
        int num_ranks_;

        int global_index_size_;

        /// local index size for each rank
        std::vector<int> local_size_;
        
        /// global index by rank and local index
        mdarray<int, 2> global_index_;

        /// location (local index and rank) of global index
        mdarray<int, 2> location_;
        
        /// Default constructor.
        splindex_base() : rank_(-1), num_ranks_(-1)
        {
        }

        /// Initialize the split index.
        void init()
        {
            if (num_ranks_ == 1) assert(local_size_[0] == global_index_size_);

            location_.set_dimensions(2, global_index_size_);
            location_.allocate();

            for (int i1 = 0; i1 < global_index_.size(_splindex_rank_); i1++)
            {
                for (int i0 = 0; i0 < global_index_.size(_splindex_offs_); i0++)
                {
                    int j = global_index_(i0, i1);
                    if (j >= 0)
                    {
                        location_(_splindex_offs_, j) = i0;
                        location_(_splindex_rank_, j) = i1;
                    }
                }
            }
        }
    
    public:

        inline int local_size()
        {
            assert(rank_ >= 0);
            
            return local_size_[rank_];
        }

        inline int local_size(int rank)
        {
            assert((rank >= 0) && rank < (num_ranks_));

            return local_size_[rank];
        }

        inline int global_index(int idxloc, int rank)
        {
            return global_index_(idxloc, rank);
        }

        inline int location(int i, int idxglob)
        {
            return location_(i, idxglob);
        }

        inline int operator[](int idxloc)
        {
            return global_index_(idxloc, rank_);
        }

        inline int global_size()
        {
            return global_index_size_;
        }
};

template <splindex_t type> class splindex: public splindex_base
{
};

template<> class splindex<block>: public splindex_base
{
    public:
        
        splindex()
        {
        }
        
        splindex(int global_index_size__, int num_ranks__, int rank__)
        {
            assert(global_index_size__ > 0);

            split(global_index_size__, num_ranks__, rank__); 
        }
        
        void split(int global_index_size__, int num_ranks__, int rank__)
        {
            assert(num_ranks__ > 0);
            assert((rank__ >= 0) && (rank__ < num_ranks__));
            assert(global_index_size__ >= 0);

            if (global_index_size__ == 0)
                error_local(__FILE__, __LINE__, "need to think what to do with zero index size");

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            global_index_size_ = global_index_size__;
            
            local_size_.resize(num_ranks_);

            // minimum size
            int n1 = global_index_size_ / num_ranks_;

            // first n2 ranks have one more index element
            int n2 = global_index_size_ % num_ranks_;
            
            global_index_.set_dimensions(n1 + std::min(1, n2), num_ranks_);
            global_index_.allocate();
            for (int i1 = 0; i1 < global_index_.size(_splindex_rank_); i1++)
            {
                for (int i0 = 0; i0 < global_index_.size(_splindex_offs_); i0++) global_index_(i0, i1) = -1;
            }

            for (int i1 = 0; i1 < num_ranks_; i1++)
            {
                int global_index_offset;
                if (i1 < n2)
                {
                    local_size_[i1] = n1 + 1; 
                    global_index_offset = (n1 + 1) * i1;
                }
                else
                {
                    local_size_[i1] = n1; 
                    global_index_offset = (n1 > 0) ? (n1 + 1) * n2 + n1 * (i1 - n2) : -1;
                }
                for (int i0 = 0; i0 < local_size_[i1]; i0++) global_index_(i0, i1) = global_index_offset + i0;
            }

            init();
        }

        inline int global_offset()
        {
            return global_index_(0, rank_);
        }

};

template<> class splindex<block_cyclic>: public splindex_base
{
    private:
        
        int block_size_;

    public:
        
        splindex()
        {
        }

        splindex(int global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            assert(global_index_size__ > 0);

            split(global_index_size__, num_ranks__, rank__, block_size__); 
        }
        
        void split(int global_index_size__, int num_ranks__, int rank__, int block_size__)
        {
            assert(num_ranks__ > 0);
            assert((rank__ >= 0) && (rank__ < num_ranks__));
            assert(global_index_size__ >= 0);
            assert(block_size__ > 0);

            if (global_index_size__ == 0) 
                error_local(__FILE__, __LINE__, "need to think what to do with zero index size");

            rank_ = rank__;
            num_ranks_ = num_ranks__;
            global_index_size_ = global_index_size__;
            block_size_ = block_size__;
            
            local_size_.resize(num_ranks_);


            int nblocks = (global_index_size_ / block_size_) +           // number of full blocks
                          std::min(1, global_index_size_ % block_size_); // extra partial block

            int max_size = ((nblocks / num_ranks_) +            // minimum number of blocks per rank
                            std::min(1, nblocks % num_ranks_)); // some ranks get extra block
            max_size *= block_size_;
            
            global_index_.set_dimensions(max_size, num_ranks_);
            global_index_.allocate();
            
            int irank = 0;
            int iblock = 0;
            std::vector< std::vector<int> > iv(num_ranks_);
            for (int i = 0; i < global_index_size_; i++)
            {
                iv[irank].push_back(i);
                if ((++iblock) == block_size_)
                {
                    iblock = 0;
                    irank = (irank + 1) % num_ranks_;
                }
            }
           
            for (int i1 = 0; i1 < global_index_.size(_splindex_rank_); i1++)
            {
                local_size_[i1] = (int)iv[i1].size();
                for (int i0 = 0; i0 < global_index_.size(_splindex_offs_); i0++)
                    global_index_(i0, i1) = (i0 < (int)iv[i1].size()) ? iv[i1][i0] : -1;
            }

            init();
        }
};

#endif // __SPLINDEX_H__


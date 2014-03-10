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
    
    \brief Contains definition of splindex_base and splindex_iterator classes.
*/

const int _splindex_offs_ = 0;
const int _splindex_rank_ = 1;

/// Base class for split index.
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

        /// size of the global index 
        int global_index_size_;

        /// Default constructor.
        splindex_base() : rank_(-1), num_ranks_(-1)
        {
        }
};

template <splindex_t type> 
class splindex: public splindex_base
{
};

#include "splindex.hpp"

/// Iterator for split index.
/** Split index iterator is introduced to simplify the loop over local part of global index.

    Example:
    \code{.cpp}
        splindex<block> spl(17, Platform::num_mpi_ranks(), Platform::mpi_rank());
        #pragma omp parallel
        for (auto it = splindex_iterator<block>(spl); it.valid(); it++)
        {
            printf("thread_id: %i, local index : %i, global index : %i\n", 
                   Platform::thread_id(), it.idx_local(), it.idx());
        }
    \endcode
*/ 
template <splindex_t type> 
class splindex_iterator
{
    private:
        
        /// current global index
        int idx_;
        
        /// current local index
        int idx_local_;

        /// incremental step for operator++
        int inc_;
        
        /// pointer to split index
        splindex<type>& splindex_;

    public:
        
        /// Constructor
        splindex_iterator(splindex<type>& splindex__) 
            : idx_(-1), 
              idx_local_(Platform::thread_id()), 
              inc_(Platform::num_threads()),
              splindex_(splindex__)
        {
            valid();
        }
        
        /// Incremental operator
        splindex_iterator<type>& operator++(int)
        {
            this->idx_local_ += inc_;
            return *this;
        }

        /// Update the global index and check if it is valid.
        /** Global index is updated using the current value of the local index. Return true if the index is valid,
            otherwise return false. */
        inline bool valid()
        {
            if (idx_local_ >= splindex_.local_size()) return false;
            idx_ = splindex_[idx_local_];
            return true;
        }
        
        /// Return current global index.
        inline int idx()
        {
            return idx_;
        }

        /// Return current local index.
        inline int idx_local()
        {
            return idx_local_;
        }
};

#endif // __SPLINDEX_H__


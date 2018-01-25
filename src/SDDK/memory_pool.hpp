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

/** \file memory_pool.hpp
 *   
 *  \brief Contains implementation of simple memory pool object.
 */

#ifndef __MEMORY_POOL_HPP__
#define __MEMORY_POOL_HPP__

#include <list>
#include "mdarray.hpp"

class memory_pool {
  private:
    size_t pos_;
    
    sddk::mdarray<int8_t, 1> pool_;

    std::list<std::unique_ptr<sddk::mdarray<int8_t, 1>>> tmp_pool_;

  public:
    memory_pool()
    {
    }
    
    /// Allocate n elements of type T.
    template <typename T>
    T* allocate(size_t n__)
    {
        //static_assert(std::is_pod<T>::value, "not a simple data type");
        
        size_t sz = n__ * sizeof(T);
        if (pos_ + sz <= pool_.size()) {
            T* ptr = reinterpret_cast<T*>(&pool_[pos_]);
            pos_ += sz;
            return ptr;
        } else {
            tmp_pool_.emplace_back(new sddk::mdarray<int8_t, 1>(sz));
            return reinterpret_cast<T*>(tmp_pool_.back()->template at<sddk::CPU>());
        }
    }

    void reset()
    {
        size_t sz{0};
        for (auto& e: tmp_pool_) {
            sz += e->size();
        }
        if (sz != 0) {
            sz += pool_.size();
            tmp_pool_.clear();
            pool_ = sddk::mdarray<int8_t, 1>(sz);
        }
        pos_ = 0;
    }
};

#endif  // __MEMORY_POOL_HPP__

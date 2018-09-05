// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

namespace sddk {

class memory_pool {
  private:

    std::map<memory_t, mdarray<int8_t, 1>> pool_;
    std::map<memory_t, size_t> pool_pos_;
    std::map<memory_t, std::vector<mdarray<int8_t, 1>>> tmp_pool_;

  public:
    
    /// Allocate n elements of type T in a specified memory.
    template <typename T, memory_t mem_type>
    T* allocate(size_t n__)
    {
        size_t sz = n__ * sizeof(T);

        if (!pool_pos_.count(mem_type)) {
            pool_pos_[mem_type] = 0;
        }
        //if (!pool_.count(mem_type)) {
        //    pool_[mem_type] = mdarray<int8_t, 1>();
        //}
        //if (!tmp_pool_.count(mem_type)) {
        //    tmp_pool_[mem_type];
        //}

        size_t pos = pool_pos_[mem_type];
        if (pos + sz <= pool_[mem_type].size()) {
            T* ptr = reinterpret_cast<T*>(pool_[mem_type].template at<device<mem_type>::type>(pos));
            pool_pos_[mem_type] += sz;
            return ptr;
        } else {
            tmp_pool_[mem_type].emplace_back(mdarray<int8_t, 1>(sz, mem_type));
            return reinterpret_cast<T*>(tmp_pool_[mem_type].back().template at<device<mem_type>::type>());
        }
    }

    template <memory_t mem_type>
    void reset()
    {
        size_t sz{0};
        if (tmp_pool_.count(mem_type)) {
            for (auto& e: tmp_pool_[mem_type]) {
                sz += e.size();
            }
        }
        if (sz != 0) {
            sz += pool_[mem_type].size();
            tmp_pool_[mem_type].clear();
            pool_[mem_type] = mdarray<int8_t, 1>(sz, mem_type);
        }
        pool_pos_[mem_type] = 0;
    }
};

//class memory_pool {
//  private:
//    size_t pos_{0};
//
//    memory_t mem_type_{memory_t::none};
//    
//    sddk::mdarray<int8_t, 1> pool_;
//
//    std::list<std::unique_ptr<sddk::mdarray<int8_t, 1>>> tmp_pool_;
//
//  public:
//    memory_pool(memory_t mem_type__ = memory_t::host)
//        : mem_type_(mem_type__)
//    {
//    }
//    
//    /// Allocate n elements of type T.
//    template <typename T, device_t pu>
//    T* allocate(size_t n__)
//    {
//        //static_assert(std::is_pod<T>::value, "not a simple data type");
//        
//        size_t sz = n__ * sizeof(T);
//        if (pos_ + sz <= pool_.size()) {
//            T* ptr = reinterpret_cast<T*>(pool_.at<pu>(pos_));
//            pos_ += sz;
//            return ptr;
//        } else {
//            tmp_pool_.emplace_back(new sddk::mdarray<int8_t, 1>(sz, mem_type_));
//            return reinterpret_cast<T*>(tmp_pool_.back()->template at<pu>());
//        }
//    }
//
//    void reset()
//    {
//        size_t sz{0};
//        for (auto& e: tmp_pool_) {
//            sz += e->size();
//        }
//        if (sz != 0) {
//            sz += pool_.size();
//            tmp_pool_.clear();
//            pool_ = sddk::mdarray<int8_t, 1>(sz, mem_type_);
//        }
//        pos_ = 0;
//    }
//};

}

#endif  // __MEMORY_POOL_HPP__

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

/// Descriptor of the allocated memory block.
struct memory_block_descriptor
{
    /// Raw pointer to the block of memory.
    uint8_t* ptr_{nullptr};
    /// Size of the memory block.
    size_t size_{0};
    /// True if this block is used.
    bool used_{false};
    /// Shared pointer to an allocated memory block.
    std::shared_ptr<mdarray<uint8_t, 1>> buf_;
    /// Keep track of the used sub-blocks of the shared memory buffer.
    std::shared_ptr<int> used_count_;
};

/// Memory pool.
/** This class stores list of allocated memory blocks. When block is deallocated it is merged with previous or next
 *  free block. */
class memory_pool
{
  private:
    /// List of blocks of allocated memory.
    std::map<memory_t, std::list<memory_block_descriptor>> memory_blocks_;
    size_t size_allocated_{0};
    size_t max_allocated_{0};

    template <memory_t mem_type>
    std::list<memory_block_descriptor>::iterator remove_block(std::list<memory_block_descriptor>::iterator it__)
    {
        (*it__->used_count_)--;
        size_allocated_ -= it__->size_;
        if (it__->buf_.use_count() > 1) {
            /* this is not the last sub-block: just remove it */
            return memory_blocks_[mem_type].erase(it__);
        } else {
            /* this is the last sub-block in the series */
            it__->used_ = false;
            it__->ptr_  = it__->buf_->at<device<mem_type>::type>();
            it__->size_ = it__->buf_->size();

            /* try to merge block pointed by it0 with the block pointed by it__ */
            auto merge_blocks = [&](std::list<memory_block_descriptor>::iterator& it0)
            {
                if (!it0->used_ && it0->buf_.use_count() == 1) {
                    it__->size_ = it__->buf_->size() + it0->buf_->size();
                    it__->buf_ = std::shared_ptr<mdarray<uint8_t, 1>>(new mdarray<uint8_t, 1>(it__->size_, mem_type));
                    it__->ptr_ = it__->buf_->at<device<mem_type>::type>();
                    /* erase the merged block */
                    memory_blocks_[mem_type].erase(it0);
                }
            };

            if (it__ != memory_blocks_[mem_type].begin()) {
                /* check previous block */
                auto it0 = it__;
                it0--;
                /* if previous block is not used and its memory is not shared with other blocks */
                merge_blocks(it0);
            }
            /* check next block */
            auto it1 = it__;
            it1++;
            if (it1 != memory_blocks_[mem_type].end()) {
                /* if next block is not used and its memory is not shared with other blocks */
                merge_blocks(it1);
            }
            it__++;
            return it__;
        }
    }

  public:
    /// Constructor
    memory_pool()
    {
    }

    template <typename T, memory_t mem_type>
    T* allocate(size_t num_elements__)
    {
        if (memory_blocks_.count(mem_type) == 0) {
            memory_blocks_[mem_type] =  std::list<memory_block_descriptor>();
        }
        /* size of the memory block in bytes */
        size_t size = num_elements__ * sizeof(T);
        size_allocated_ += size;
        max_allocated_ = std::max(max_allocated_, size_allocated_);
        /* iterate over existing blocks */
        for (auto it = memory_blocks_[mem_type].begin(); it != memory_blocks_[mem_type].end(); ++it) {
            /* if unused block with suitable size is found */
            if (!it->used_ && it->size_ >= size) {
                /* divide block into two parts (head and tail) */
                memory_block_descriptor tail_block;
                tail_block.ptr_        = it->ptr_ + size;
                tail_block.size_       = it->size_ - size;
                tail_block.used_       = false;
                tail_block.buf_        = it->buf_;
                tail_block.used_count_ = it->used_count_;
                /* shrink the size of the current block */
                it->size_ = size;
                it->used_ = true;
                (*it->used_count_)++;
                auto ptr = it->ptr_;
                /* insert the tail block */
                if (tail_block.size_ > 0) {
                    it++;
                    memory_blocks_[mem_type].insert(it, std::move(tail_block));
                }
                return reinterpret_cast<T*>(ptr);
            }
        }
        /* if the free block is not found, create a new one */
        memory_block_descriptor new_block;
        new_block.buf_        = std::shared_ptr<mdarray<uint8_t, 1>>(new mdarray<uint8_t, 1>(size, mem_type));
        new_block.ptr_        = new_block.buf_->at<device<mem_type>::type>();
        new_block.size_       = size;
        new_block.used_       = true;
        new_block.used_count_ = std::shared_ptr<int>(new int);
        (*new_block.used_count_) = 1;
        auto ptr = new_block.ptr_;
        memory_blocks_[mem_type].push_back(std::move(new_block));
        return reinterpret_cast<T*>(ptr);
    }

    template <memory_t mem_type>
    void free(void* ptr__)
    {
        /* iterate over memory blocks */
        for (auto it = memory_blocks_[mem_type].begin(); it != memory_blocks_[mem_type].end(); ++it) {
            if (it->ptr_ == ptr__) {
                if (!it->used_) {
                    throw std::runtime_error("wrong pointer");
                }
                remove_block<mem_type>(it);
                return;
            }
        }
        throw std::runtime_error("wrong pointer");
    }

    /// Free all the allocated blocks and merge them into one big piece of free memory.
    template <memory_t mem_type>
    void reset()
    {
        auto mem_type_entry = memory_blocks_.find(mem_type);
        if (mem_type_entry != memory_blocks_.end()) {
            /* iterate over memory blocks */
            auto it = mem_type_entry->second.begin();
            while (it != mem_type_entry->second.end()) {
                it = remove_block<mem_type>(it);
            }
            it = mem_type_entry->second.begin();
            if (mem_type_entry->second.size() != 1 || it->used_ || (it->size_ != it->buf_->size())) {
                std::stringstream s;
                s << "error in memory_pool::reset()\n"
                  << "  list size   : " << mem_type_entry->second.size() << " (expecting 1)\n"
                  << "  used        : " << it->used_ << " (expecting false)\n"
                  << "  buffer size : " << it->size_ << " " <<  it->buf_->size() << " (expecting equal)";
                TERMINATE(s);
            }
        }
    }

    template <memory_t mem_type>
    void print()
    {
        std::cout << "--- memory pool status ---\n";
        int i{0};
        for (auto it = memory_blocks_[mem_type].begin(); it != memory_blocks_[mem_type].end(); ++it) {
            std::cout << i << ", size: " << it->size_ << ", used: " << it->used_ << ", parent_size: " << it->buf_->size() << "\n";
            i++;
        }
        std::cout << "max_allocated: " << max_allocated_ << "\n";
    }
};


//class memory_pool {
//  private:
//
//    std::map<memory_t, mdarray<int8_t, 1>> pool_;
//    std::map<memory_t, size_t> pool_pos_;
//    std::map<memory_t, std::vector<mdarray<int8_t, 1>>> tmp_pool_;
//
//  public:
//    
//    /// Allocate n elements of type T in a specified memory.
//    template <typename T, memory_t mem_type>
//    T* allocate(size_t n__)
//    {
//        size_t sz = n__ * sizeof(T);
//
//        if (!pool_pos_.count(mem_type)) {
//            pool_pos_[mem_type] = 0;
//        }
//        //if (!pool_.count(mem_type)) {
//        //    pool_[mem_type] = mdarray<int8_t, 1>();
//        //}
//        //if (!tmp_pool_.count(mem_type)) {
//        //    tmp_pool_[mem_type];
//        //}
//
//        size_t pos = pool_pos_[mem_type];
//        if (pos + sz <= pool_[mem_type].size()) {
//            T* ptr = reinterpret_cast<T*>(pool_[mem_type].template at<device<mem_type>::type>(pos));
//            pool_pos_[mem_type] += sz;
//            return ptr;
//        } else {
//            tmp_pool_[mem_type].emplace_back(mdarray<int8_t, 1>(sz, mem_type));
//            return reinterpret_cast<T*>(tmp_pool_[mem_type].back().template at<device<mem_type>::type>());
//        }
//    }
//
//    template <memory_t mem_type>
//    void reset()
//    {
//        size_t sz{0};
//        if (tmp_pool_.count(mem_type)) {
//            for (auto& e: tmp_pool_[mem_type]) {
//                sz += e.size();
//            }
//        }
//        if (sz != 0) {
//            sz += pool_[mem_type].size();
//            tmp_pool_[mem_type].clear();
//            pool_[mem_type] = mdarray<int8_t, 1>(sz, mem_type);
//        }
//        pool_pos_[mem_type] = 0;
//    }
//};

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

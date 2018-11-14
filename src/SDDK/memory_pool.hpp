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
#include <iostream>
#include <map>
#ifdef __GPU
#include "GPU/cuda.hpp"
#endif

namespace sddk {

/// Type of memory.
/** List the types of memory on which the code can store data.
    Various combinations of flags can be used. To check for any host memory (pinned or non-pinned):
    \code{.cpp}
    mem_type & memory_t::host == memory_t::host
    \endcode
    To check for pinned memory:
    \code{.cpp}
    mem_type & memory_t::host_pinned == memory_t::host_pinned
    \endcode
    To check for device memory:
    \code{.cpp}
    mem_type & memory_t::device == memory_t::device
    \endcode
*/
enum class memory_t : unsigned int
{
    /// Nothing.
    none        = 0b0000,
    /// Host memory.
    host        = 0b0001,
    /// Pinned host memory. This is host memory + extra bit flag.
    host_pinned = 0b0011,
    /// Device memory.
    device      = 0b0100,
    /// Managed memory (accessible from both host and device).
    managed     = 0b1000
};

/// Allocate n elements in a specified memory.
/** Allocate a memory block of the memory_t type. Return a nullptr if this memory is not available, otherwise
 *  return a pointer to an allocated block. */
template <typename T>
inline T* allocate(size_t n__, memory_t M__)
{
    switch (M__) {
        case memory_t::none: {
            return nullptr;
        }
        case memory_t::host: {
            return static_cast<T*>(std::malloc(n__ * sizeof(T)));
        }
        case memory_t::host_pinned: {
#ifdef __GPU
            return acc::allocate_host<T>(n__);
#else
            return nullptr;
#endif
        }
        case memory_t::device: {
#ifdef __GPU
            return acc::allocate<T>(n__);
#else
            return nullptr;
#endif
        }
        default: {
            throw std::runtime_error("unknown memory type");
        }
    }
}

/// Deallocate pointer.
inline void deallocate(void* ptr__, memory_t M__)
{
    switch (M__) {
        case memory_t::none: {
            break;
        }
        case memory_t::host: {
            std::free(ptr__);
            break;
        }
        case memory_t::host_pinned: {
#ifdef __GPU
            acc::deallocate_host(ptr__);
#endif
            break;
        }
        case memory_t::device: {
#ifdef __GPU
            acc::deallocate(ptr__);
#endif
            break;
        }
        default: {
            throw std::runtime_error("unknown memory type");
        }
    }
}

/// Deleter for the smart pointers.
struct memory_t_deleter
{
    memory_t M_{memory_t::none};
    memory_t_deleter(memory_t M__)
        : M_(M__)
    {
    }
    inline void operator()(void* ptr__)
    {
        sddk::deallocate(ptr__, M_);
    }
};

template <typename T>
inline std::unique_ptr<T, memory_t_deleter> get_unique_ptr(size_t n__, memory_t M__)
{
    return std::move(std::unique_ptr<T, memory_t_deleter>(allocate<T>(n__, M__), M__));
}

template <typename T>
inline std::shared_ptr<T> get_shared_ptr(size_t n__, memory_t M__)
{
    return std::move(std::shared_ptr<T>(allocate<T>(n__, M__), M__));
}

/// Descriptor of the allocated memory block.
struct memory_block_descriptor
{
    /// Storage buffer for the memory blocks.
    std::unique_ptr<uint8_t, memory_t_deleter> buffer_;
    /// Size of the storage buffer.
    size_t size_{0};
    /// List of <offset, size> pairs of the free subblocks.
    /** The list is ordered, i.e. offset of the next free block is greater or equal to the offset + size of the
     *  previous block. */
    std::list<std::pair<size_t, size_t>> free_subblocks_;

    /// Create a new empty memory block.
    memory_block_descriptor(size_t size__, memory_t M__)
        : buffer_(get_unique_ptr<uint8_t>(size__, M__))
        , size_(size__)
    {
        free_subblocks_.push_back(std::make_pair(0, size_));
    }

    /// Check if the memory block is empty.
    inline bool is_empty() const
    {
        return (free_subblocks_.size() == 1 &&
                free_subblocks_.front().first == 0 &&
                free_subblocks_.front().second == size_);
    }

    uint8_t* allocate_subblock(size_t size__)
    {
        uint8_t* ptr{nullptr};
         for (auto it = free_subblocks_.begin(); it != free_subblocks_.end(); it++) {
            /* if this free subblock can fit the "size" elements */
            if (size__ <= it->second) {
                /* pointer to the beginning of subblock */
                ptr = buffer_.get() + it->first;
                it->first += size__;
                it->second -= size__;
                if (it->second == 0) {
                    free_subblocks_.erase(it);
                }
                break;
            }
        }
        return ptr;
    }

    void free_subblock(uint8_t* ptr__, size_t size__)
    {
        /* offset from the beginning of the memory buffer */
        size_t offset = static_cast<size_t>(ptr__ - buffer_.get());

        for (auto it = free_subblocks_.begin(); it != free_subblocks_.end(); it++) {
            /* check if we can attach released subblock before this subblock */
            if (it->first == offset + size__) {
                it->first = offset;
                it->second += size__;
                return;
            }
            /* check if we can attach released subblock after this subblock */
            if (it->first + it->second == offset) {
                it->second += size__;
                return;
            }
            /* finally, check if the released subblock is before this subblock, but not touching it */
            if (offset + size__ < it->first) {
                free_subblocks_.insert(it, std::make_pair(offset, size__));
                return;
            }
        }
        /* otherwise this is the tail subblock */
        free_subblocks_.push_back(std::make_pair(offset, size__));
    }

    /// Return the total size of the free subblocks.
    size_t get_free_size() const
    {
        size_t sz{0};
        for (auto& e: free_subblocks_) {
            sz += e.second;
        }
        return sz;
    }
};

/// Store information about the allocated subblock: iterator in the list of memory blocks and subblock size;
using memory_subblock_descriptor = std::pair<std::list<memory_block_descriptor>::iterator, size_t>;

/// Memory pool.
/** This class stores list of allocated memory blocks. When block is deallocated it is merged with previous or next
 *  free block. */
class memory_pool
{
  private:
    memory_t M_;
    /// List of blocks of allocated memory.
    std::list<memory_block_descriptor> memory_blocks_;
    /// Mapping between an allocated pointer and a subblock descriptor.
    std::map<uint8_t*, memory_subblock_descriptor> map_ptr_;

    /// Deleter for the smart pointers when memory_pool was used for allocation.
    struct deleter
    {
        memory_pool* mp_{nullptr};
        deleter()
        {
        }
        deleter(memory_pool& mp__)
            : mp_(&mp__)
        {
        }
        deleter(deleter&& src__)
        {
            this->mp_ = src__.mp_;
            src__.mp_ = nullptr;
        }
        inline deleter& operator=(deleter&& src__)
        {
            if (this != &src__) {
                this->mp_ = src__.mp_;
                src__.mp_ = nullptr;
            }
            return *this;
        }
        inline void operator()(void* ptr__)
        {
            if (mp_) {
                mp_->free(ptr__);
            }
        }
    };

  public:

    template <typename T>
    using unique_ptr = std::unique_ptr<T, deleter>;

    /// Constructor
    memory_pool(memory_t M__)
        : M_(M__)
    {
    }

    template <typename T>
    T* allocate(size_t num_elements__)
    {
        /* size of the memory block in bytes */
        size_t size = num_elements__ * sizeof(T);

        uint8_t* ptr{nullptr};

        /* iterate over existing blocks */
        auto it = memory_blocks_.begin();
        for (; it != memory_blocks_.end(); it++) {
            /* try to allocate a block */
            ptr = it->allocate_subblock(size);
            /* break if this memory block can store the subblock */
            if (ptr) {
                break;
            }
        }
        /* if memory block was not found, add another one */
        if (!ptr) {
            memory_blocks_.push_back(memory_block_descriptor(size, M_));
            it = memory_blocks_.end();
            it--;
            ptr = it->allocate_subblock(size);
        }
        if (!ptr) {
            throw std::runtime_error("memory allocation failed");
        }
        memory_subblock_descriptor msb;
        msb.first = it;
        msb.second = size;
        /* add to the hash table */
        map_ptr_[ptr] = msb;
        return reinterpret_cast<T*>(ptr);
    }

    void free(void* ptr__)
    {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(ptr__);
        auto& msb = map_ptr_.at(ptr);
        msb.first->free_subblock(ptr, msb.second);
        /* merge memory blocks; this is not strictly necessary but can lead to a better performance */
        auto it = msb.first;
        if (it->is_empty()) {
            if (it != memory_blocks_.begin()) {
                auto it0 = it;
                it0--;
                if (it0->is_empty()) {
                    size_t size = it->size_ + it0->size_;
                    (*it) = memory_block_descriptor(size, M_);
                    memory_blocks_.erase(it0);
                }
            }
            auto it0 = it;
            it0++;
            if (it0 != memory_blocks_.end()) {
                if (it0->is_empty()) {
                    size_t size = it->size_ + it0->size_;
                    (*it) = memory_block_descriptor(size, M_);
                    memory_blocks_.erase(it0);
                }
            }
        }
        /* remove this pointer from the hash table */
        map_ptr_.erase(ptr);
    }

    template <typename T>
    unique_ptr<T> get_unique_ptr(size_t n__)
    {
        return std::move(unique_ptr<T>(allocate<T>(n__), *this));
    }

    /// Free all the allocated blocks.
    void reset()
    {
        for (auto it = memory_blocks_.begin(); it != memory_blocks_.end(); it++) {
            it->free_subblocks_.clear();
            it->free_subblocks_.push_back(std::make_pair(0, it->size_));
        }
        map_ptr_.clear();
    }

    void print()
    {
        std::cout << "--- memory pool status ---\n";
        int i{0};
        for (auto& e: memory_blocks_) {
            std::cout << "memory block: " << i << ", capacity: " << e.size_
                      << ", free size: " << e.get_free_size() << "\n";
            i++;
        }
    }
};

}

#endif  // __MEMORY_POOL_HPP__

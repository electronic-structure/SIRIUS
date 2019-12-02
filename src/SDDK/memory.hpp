// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file memory.hpp
 *
 *  \brief Memory management functions and classes.
 */

#ifndef __MEMORY_HPP__
#define __MEMORY_HPP__

#include <list>
#include <iostream>
#include <map>
#include <memory>
#include <cstring>
#include <functional>
#include <algorithm>
#include <array>
#include <complex>
#include <cassert>
#include "GPU/acc.hpp"

namespace sddk {

/// Check is the type is a complex number; by default it is not.
template <typename T>
struct is_complex
{
    constexpr static bool value{false};
};

/// Check is the type is a complex number: for std::complex<T> it is true.
template<typename T>
struct is_complex<std::complex<T>>
{
    constexpr static bool value{true};
};

/// Memory types where the code can store data.
/** All memory types can be divided into two (possibly overlapping) groups: accessible by the CPU and accessible by the
 *  device. */
enum class memory_t : unsigned int
{
    /// Nothing.
    none        = 0b0000,
    /// Host memory.
    host        = 0b0001,
    /// Pinned host memory. This is host memory + extra bit flag.
    host_pinned = 0b0011,
    /// Device memory.
    device      = 0b1000,
    /// Managed memory (accessible from both host and device).
    managed     = 0b1101,
};

/// Check if this is a valid host memory (memory, accessible by the host).
inline bool is_host_memory(memory_t mem__)
{
    return static_cast<unsigned int>(mem__) & 0b0001;
}

/// Check if this is a valid device memory (memory, accessible by the device).
inline bool is_device_memory(memory_t mem__)
{
    return static_cast<unsigned int>(mem__) & 0b1000;
}

/// Get a memory type from a string.
inline memory_t get_memory_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);

    std::map<std::string, memory_t> const map_to_type = {
        {"none",        memory_t::none},
        {"host",        memory_t::host},
        {"host_pinned", memory_t::host_pinned},
        {"managed",     memory_t::managed},
        {"device",      memory_t::device}
    };

    if (map_to_type.count(name__) == 0) {
        std::stringstream s;
        s << "wrong label of memory type: " << name__;
        throw std::runtime_error(s.str());
    }

    return map_to_type.at(name__);
}

/// Type of the main processing unit.
/** List the processing units on which the code can run. */
enum class device_t
{
    /// CPU device.
    CPU = 0,

    /// GPU device (with CUDA programming model).
    GPU = 1
};

/// Get type of device by memory type.
inline device_t get_device_t(memory_t mem__)
{
    switch (mem__) {
        case memory_t::host:
        case memory_t::host_pinned: {
            return device_t::CPU;
        }
        case memory_t::device: {
            return device_t::GPU;
        }
        default: {
            throw std::runtime_error("get_device_t(): wrong memory type");
        }
    }
    return device_t::CPU; // make compiler happy
}

/// Get device type from the string.
inline device_t get_device_t(std::string name__)
{
    std::transform(name__.begin(), name__.end(), name__.begin(), ::tolower);
    if (name__ == "cpu") {
        return device_t::CPU;
    } else if (name__ == "gpu") {
        return device_t::GPU;
    } else {
        throw std::runtime_error("get_device_t(): wrong processing unit");
    }
    return device_t::CPU; // make compiler happy
}

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
            throw std::runtime_error("allocate(): unknown memory type");
        }
    }
}

/// Deallocate pointer of a given memory type.
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
            throw std::runtime_error("deallocate(): unknown memory type");
        }
    }
}

/// Copy between different memory types.
template <typename T>
inline void copy(memory_t from_mem__, T const* from_ptr__, memory_t to_mem__, T* to_ptr__, size_t n__)
{
    if (is_host_memory(to_mem__) && is_host_memory(from_mem__)) {
        std::memcpy(to_ptr__, from_ptr__, n__ * sizeof(T));
        return;
    }
#if defined(__GPU)
    if (is_device_memory(to_mem__) && is_device_memory(from_mem__)) {
        acc::copy(to_ptr__, from_ptr__, n__);
        return;
    }
    if (is_device_memory(to_mem__) && is_host_memory(from_mem__)) {
        acc::copyin(to_ptr__, from_ptr__, n__);
        return;
    }
    if (is_host_memory(to_mem__) && is_device_memory(from_mem__)) {
        acc::copyout(to_ptr__, from_ptr__, n__);
        return;
    }
#endif
}

/* forward declaration */
class memory_pool;

/// Base class for smart pointer deleters.
class memory_t_deleter_base
{
  protected:
    class memory_t_deleter_base_impl
    {
      public:
        virtual void free(void* ptr__) = 0;
        virtual ~memory_t_deleter_base_impl()
        {
        }
    };
    std::unique_ptr<memory_t_deleter_base_impl> impl_;

  public:
    void operator()(void* ptr__)
    {
        impl_->free(ptr__);
    }
};

/// Deleter for the allocated memory pointer of a given type.
class memory_t_deleter: public memory_t_deleter_base
{
  protected:
    class memory_t_deleter_impl: public memory_t_deleter_base_impl
    {
      protected:
        memory_t M_{memory_t::none};
      public:
        memory_t_deleter_impl(memory_t M__)
            : M_(M__)
        {
        }
        inline void free(void* ptr__)
        {
            if (M_ != memory_t::none) {
                sddk::deallocate(ptr__, M_);
            }
        }
    };
  public:
    explicit memory_t_deleter(memory_t M__)
    {
        impl_ = std::unique_ptr<memory_t_deleter_base_impl>(new memory_t_deleter_impl(M__));
    }
};

/// Deleter for the allocated memory pointer from a given memory pool.
class memory_pool_deleter: public memory_t_deleter_base
{
  protected:
    class memory_pool_deleter_impl: public memory_t_deleter_base_impl
    {
      protected:
        memory_pool* mp_{nullptr};
      public:
        memory_pool_deleter_impl(memory_pool* mp__)
            : mp_(mp__)
        {
        }
        inline void free(void* ptr__);
    };

  public:
    explicit memory_pool_deleter(memory_pool* mp__)
    {
        impl_ = std::unique_ptr<memory_t_deleter_base_impl>(new memory_pool_deleter_impl(mp__));
    }
};

/// Allocate n elements and return a unique pointer.
template <typename T>
inline std::unique_ptr<T, memory_t_deleter_base> get_unique_ptr(size_t n__, memory_t M__)
{
    return std::unique_ptr<T, memory_t_deleter_base>(allocate<T>(n__, M__), memory_t_deleter(M__));
}

/// Descriptor of the allocated memory block.
/** Internally the block might be split into sub-blocks. */
struct memory_block_descriptor
{
    /// Storage buffer for the memory blocks.
    std::unique_ptr<uint8_t, memory_t_deleter_base> buffer_;
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

    /// Return total size of the block.
    inline size_t size() const
    {
        return size_;
    }

    /// Try to allocate a subblock of memory.
    /** Return a valid pointer in case of success and nullptr if empty space can't be found in this memory block.
        The returned pointer is not aligned. */
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

    /// Free the pointer and its memory to the list of free subblocks.
    void free_subblock(uint8_t* ptr__, size_t size__)
    {
        /* offset from the beginning of the memory buffer */
        size_t offset = static_cast<size_t>(ptr__ - buffer_.get());
        auto check_free_subblocks = [&]()
        {
#ifndef NDEBUG
            if (free_subblocks_.size() <= 1) {
                return;
            }
            auto it = free_subblocks_.begin();
            auto it1 = it;
            it1++;
            for (; it1 != free_subblocks_.end(); it1++) {
                /* if offse + size of the previous free block is larger than the offset of next block
                   this is an error */
                if (it->first + it->second > it1->first) {
                    throw std::runtime_error("wrong order of free memory blocks");
                }
            }
#endif
        };

        for (auto it = free_subblocks_.begin(); it != free_subblocks_.end(); it++) {
            /* check if we can attach released subblock before this subblock */
            if (it->first == offset + size__) {
                it->first = offset;
                it->second += size__;
                check_free_subblocks();
                return;
            }
            /* check if we can attach released subblock after this subblock */
            if (it->first + it->second == offset) {
                it->second += size__;
                /* now check if we can attach this subblock to the top of the next one */
                auto it1 = it;
                it1++;
                if (it1 != free_subblocks_.end()) {
                    if (it->first + it->second == it1->first) {
                        /* merge second block into first and erase it */
                        it->second += it1->second;
                        free_subblocks_.erase(it1);
                    }
                }
                check_free_subblocks();
                return;
            }
            /* finally, check if the released subblock is before this subblock, but not touching it */
            if (offset + size__ < it->first) {
                free_subblocks_.insert(it, std::make_pair(offset, size__));
                check_free_subblocks();
                return;
            }
        }
        /* otherwise this will be the last free subblock */
        free_subblocks_.push_back(std::make_pair(offset, size__));
        check_free_subblocks();
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
struct memory_subblock_descriptor
{
    /// Iterator in the list of block descriptors stored by memory pool.
    /** Iterator points to a memory block in which this sub-block was allocated */
    std::list<memory_block_descriptor>::iterator it_;
    /// Size of the sub-block.
    size_t size_;
    /// This is the precise beginning of the memory sub-block.
    /** Used to compute the exact location of the sub-block inside a memory block. */
    uint8_t* unaligned_ptr_;
};

//// Memory pool.
/** This class stores list of allocated memory blocks. Each of the blocks can be devided into subblocks. When subblock
 *  is deallocated it is merged with previous or next free subblock in the memory block. If this was the last subblock
 *  in the block of memory, the (now) free block of memory is merged with the neighbours (if any are available).
 */
class memory_pool
{
  private:
    /// Type of memory that is handeled by this pool.
    memory_t M_;
    /// List of blocks of allocated memory.
    std::list<memory_block_descriptor> memory_blocks_;
    /// Mapping between an allocated pointer and a subblock descriptor.
    std::map<uint8_t*, memory_subblock_descriptor> map_ptr_;

  public:

    /// Constructor
    memory_pool(memory_t M__, size_t initial_size__ = 0)
        : M_(M__)
    {
        if (initial_size__) {
            memory_blocks_.push_back(memory_block_descriptor(initial_size__, M_));
        }
    }

    /// Return a pointer to a memory block for n elements of type T.
    template <typename T>
    T* allocate(size_t num_elements__)
    {
#if defined(__USE_MEMORY_POOL)
        /* memory block descriptor returns an unaligned memory; here we compute the the aligment value */
        size_t align_size = std::max(size_t(64), alignof(T));
        /* size of the memory block in bytes */
        size_t size = num_elements__ * sizeof(T) + align_size;

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

        /* if memory chunk was not found in the list of available blocks, add a new memory block with enough capacity */
        if (!ptr) {
            /* free all empty blocks and get their total size */
            size_t new_size{0};
            auto i = memory_blocks_.begin();
            while (i != memory_blocks_.end()) {
                if (i->is_empty()) {
                    new_size += i->size();
                    memory_blocks_.erase(i++);
                } else {
                    ++i;
                }
            }
            /* get upper limit for the size of the new block */
            new_size = std::max(new_size, size);

            memory_blocks_.push_back(memory_block_descriptor(new_size, M_));
            it = memory_blocks_.end();
            it--;
            ptr = it->allocate_subblock(size);
        }
        if (!ptr) {
            throw std::runtime_error("memory allocation failed");
        }
        /* save the information about the allocated memory sub-block */
        memory_subblock_descriptor msb;
        /* location in the list of blocks */
        msb.it_ = it;
        /* total size including the aligment */
        msb.size_ = size;
        /* beginning of the block (unaligned) */
        msb.unaligned_ptr_ = ptr;
        auto uip = reinterpret_cast<std::uintptr_t>(ptr);
        /* align the pointer */
        if (uip % align_size) {
            uip += (align_size - uip % align_size);
        }
        auto aligned_ptr = reinterpret_cast<uint8_t*>(uip);
        /* add to the hash table */
        map_ptr_[aligned_ptr] = msb;
        return reinterpret_cast<T*>(aligned_ptr);
#else
        return sddk::allocate<T>(num_elements__, M_);
#endif
    }

    /// Delete a pointer and add its memory back to the pool.
    void free(void* ptr__)
    {
#if defined(__USE_MEMORY_POOL)
        auto ptr = reinterpret_cast<uint8_t*>(ptr__);
        /* get a descriptor of this pointer */
        auto& msb = map_ptr_.at(ptr);
        /* free the sub-block */
        msb.it_->free_subblock(msb.unaligned_ptr_, msb.size_);
        /* remove this pointer from the hash table */
        map_ptr_.erase(ptr);
#else
        sddk::deallocate(ptr__, M_);
#endif
    }

    /// Return a unique pointer to the allocated memory.
    template <typename T>
    std::unique_ptr<T, memory_t_deleter_base> get_unique_ptr(size_t n__)
    {
#if defined(__USE_MEMORY_POOL)
        return std::unique_ptr<T, memory_t_deleter_base>(this->allocate<T>(n__), memory_pool_deleter(this));
#else
        return sddk::get_unique_ptr<T>(n__, M_);
#endif
    }

    /// Free all the allocated blocks.
    /** All pointers and smart pointers, allocated by the pool are invalidated. */
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

    /// Return the type of memory this pool is managing.
    inline memory_t memory_type() const
    {
        return M_;
    }

    /// Return the total capacity of the memory pool.
    size_t total_size() const
    {
        size_t s{0};
        for (auto it = memory_blocks_.begin(); it != memory_blocks_.end(); it++) {
            s += it->size_;
        }
        return s;
    }

    /// Get the total free size of the memory pool.
    size_t free_size() const
    {
        size_t s{0};
        for (auto it = memory_blocks_.begin(); it != memory_blocks_.end(); it++) {
            s += it->get_free_size();
        }
        return s;
    }

    /// Get the number of free memory blocks.
    size_t num_blocks() const
    {
        size_t s{0};
        for (auto it = memory_blocks_.begin(); it != memory_blocks_.end(); it++) {
            s += it->free_subblocks_.size();
        }
        return s;
    }

    /// Get the number of stored pointers.
    size_t num_stored_ptr() const
    {
        return map_ptr_.size();
    }
};

void memory_pool_deleter::memory_pool_deleter_impl::free(void* ptr__)
{
    mp_->free(ptr__);
}

//#ifdef __GPU
//extern "C" void add_checksum_gpu(cuDoubleComplex* wf__,
//                                 int num_rows_loc__,
//                                 int nwf__,
//                                 cuDoubleComplex* result__);
//#endif

#ifdef NDEBUG
#define mdarray_assert(condition__)
#else
#define mdarray_assert(condition__)                             \
{                                                               \
    if (!(condition__)) {                                       \
        std::printf("Assertion (%s) failed ", #condition__);         \
        std::printf("at line %i of file %s\n", __LINE__, __FILE__);  \
        std::printf("array label: %s\n", label_.c_str());            \
        for (int i = 0; i < N; i++) {                           \
            std::printf("dim[%i].size = %li\n", i, dims_[i].size()); \
        }                                                       \
        raise(SIGTERM);                                         \
        exit(-13);                                              \
    }                                                           \
}
#endif

/// Index descriptor of mdarray.
class mdarray_index_descriptor
{
  public:
    using index_type = int64_t;

  private:
    /// Beginning of index.
    index_type begin_{0};

    /// End of index.
    index_type end_{-1};

    /// Size of index.
    size_t size_{0};

  public:

    /// Constructor of empty descriptor.
    mdarray_index_descriptor()
    {
    }

    /// Constructor for index range [0, size).
    mdarray_index_descriptor(size_t const size__)
        : end_(size__ - 1)
        , size_(size__)
    {
    }

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(index_type const begin__, index_type const end__)
        : begin_(begin__)
        , end_(end__)
        , size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(std::pair<int, int> const range__)
        : begin_(range__.first)
        , end_(range__.second)
        , size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Return first index value.
    inline index_type begin() const
    {
        return begin_;
    }

    /// Return last index value.
    inline index_type end() const
    {
        return end_;
    }

    /// Return index size.
    inline size_t size() const
    {
        return size_;
    }
};

/// Multidimensional array with the column-major (Fortran) order.
/** The implementation supports two memory pointers: one is accessible by CPU and second is accessible by a device.
    The following constructors are implemented:
    \code{.cpp}
    // wrap a host memory pointer and create 2D array 10 x 20.
    mdarray<T, 2>(ptr, 10, 20);

    // wrap a host and device pointers
    mdarray<T, 2>(ptr, ptr_d, 10, 20);

    // wrap a device pointers only
    mdarray<T, 2>(nullptr, ptr_d, 10, 20);

    // create 10 x 20 2D array in main memory
    mdarray<T, 2>(10, 20);

    // create 10 x 20 2D array in device memory
    mdarray<T, 2>(10, 20, memory_t::device);

    // create from the pool memory (pool of any memory type is allowed)
    memory_pool mp(memory_t::host);
    mdarray<T, 2>(mp, 10, 20);
    \endcode
    The pointers can be wrapped only in constructor. Memory allocation can be done by a separate call to .allocate()
    method.
 */
template <typename T, int N>
class mdarray
{
  public:
    using index_type = mdarray_index_descriptor::index_type;

  private:
    /// Optional array label.
    std::string label_;

    /// Unique pointer to the allocated memory.
    std::unique_ptr<T, memory_t_deleter_base> unique_ptr_{nullptr};

    /// Raw pointer.
    T* raw_ptr_{nullptr};
#ifdef __GPU
    /// Unique pointer to the allocated GPU memory.
    std::unique_ptr<T, memory_t_deleter_base> unique_ptr_device_{nullptr};

    /// Raw pointer to GPU memory
    T* raw_ptr_device_{nullptr};
#endif
    /// Array dimensions.
    std::array<mdarray_index_descriptor, N> dims_;

    /// List of offsets to compute the element location by dimension indices.
    std::array<index_type, N> offsets_;

    /// Initialize the offsets used to compute the index of the elements.
    void init_dimensions(std::array<mdarray_index_descriptor, N> const dims__)
    {
        dims_ = dims__;

        offsets_[0] = -dims_[0].begin();
        size_t ld{1};
        for (int i = 1; i < N; i++) {
            ld *= dims_[i - 1].size();
            offsets_[i] = ld;
            offsets_[0] -= ld * dims_[i].begin();
        }
    }

    /// Return linear index in the range [0, size) by the N-dimensional indices.(i0, i1, ...)
    template <typename... Args>
    inline index_type idx(Args... args) const
    {
        static_assert(N == sizeof...(args), "wrong number of dimensions");
        std::array<index_type, N> i = {args...};

        for (int j = 0; j < N; j++) {
            mdarray_assert(i[j] >= dims_[j].begin() && i[j] <= dims_[j].end());
        }

        size_t idx = offsets_[0] + i[0];
        for (int j = 1; j < N; j++) {
            idx += i[j] * offsets_[j];
        }
        mdarray_assert(idx >= 0 && idx < size());
        return idx;
    }

    /// Return cosnt pointer to an element at a given index.
    inline T const* at_idx(memory_t mem__, index_type const idx__) const
    {
        switch (mem__) {
            case memory_t::host:
            case memory_t::host_pinned: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case memory_t::device: {
#ifdef __GPU
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                std::printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                throw std::runtime_error("");
#endif
            }
            default: {
                throw std::runtime_error("mdarray::at_idx(): wrong memory type");
            }
        }
        return nullptr; // make compiler happy;
    }

    /// Return pointer to an element at a given index.
    inline T* at_idx(memory_t mem__, index_type const idx__)
    {
        return const_cast<T*>(static_cast<mdarray<T, N> const&>(*this).at_idx(mem__, idx__));
    }

    // Call constructor on non-trivial data. Complex numbers are treated as trivial.
    inline void call_constructor()
    {
        if (!(std::is_trivial<T>::value || is_complex<T>::value)) {
            for (size_t i = 0; i < size(); i++) {
                new (raw_ptr_ + i) T();
            }
        }
    }

    // Call destructor on non-trivial data. Complex numbers are treated as trivial.
    inline void call_destructor()
    {
        if (!(std::is_trivial<T>::value || is_complex<T>::value)) {
            for (size_t i = 0; i < this->size(); i++) {
                (raw_ptr_ + i)->~T();
            }
        }
    }

    /// Copy constructor is forbidden
    mdarray(mdarray<T, N> const& src) = delete;

    /// Assignment operator is forbidden
    mdarray<T, N>& operator=(mdarray<T, N> const& src) = delete;

  public:

    /// Default constructor.
    mdarray()
    {
    }

    /// Destructor.
    ~mdarray()
    {
        deallocate(memory_t::host);
        deallocate(memory_t::device);
    }

    /// N-dimensional array with index bounds.
    mdarray(std::array<mdarray_index_descriptor, N> const dims__,
            memory_t memory__ = memory_t::host,
            std::string label__ = "")
        : label_(label__)
    {
        this->init_dimensions(dims__);
        this->allocate(memory__);
    }

    /// 1D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->allocate(memory__);
    }

    /// 2D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->allocate(memory__);
    }

    /// 3D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->allocate(memory__);
    }

    /// 4D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 4, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3});
        this->allocate(memory__);
    }

    /// 5D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 5, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4});
        this->allocate(memory__);
    }

    /// 6D array with memory allocation.
    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            mdarray_index_descriptor const& d5,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 6, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4, d5});
        this->allocate(memory__);
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->raw_ptr_device_ = ptr_device__;
#endif
    }

    /// 1D array with memory pool allocation.
    mdarray(mdarray_index_descriptor const& d0, memory_pool& mp__, std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->allocate(mp__);
    }

    /// Wrap a pointer into 2D array.
    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->raw_ptr_device_ = ptr_device__;
#endif
    }

    /// 2D array with memory pool allocation.
    mdarray(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1, memory_pool& mp__,
            std::string label__ = "")
    {
        static_assert(N == 2, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1});
        this->allocate(mp__);
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->raw_ptr_ = ptr__;
#ifdef __GPU
        this->raw_ptr_device_ = ptr_device__;
#endif
    }

    /// 3D array with memory pool allocation.
    mdarray(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1, mdarray_index_descriptor const& d2,
        memory_pool& mp__, std::string label__ = "")
    {
        static_assert(N == 3, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->allocate(mp__);
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            std::string label__ = "")
    {
        static_assert(N == 4, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            std::string label__ = "")
    {
        static_assert(N == 5, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            mdarray_index_descriptor const& d4,
            mdarray_index_descriptor const& d5,
            std::string label__ = "")
    {
        static_assert(N == 6, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3, d4, d5});
        this->raw_ptr_ = ptr__;
    }

    /// Move constructor
    mdarray(mdarray<T, N>&& src)
        : label_(src.label_)
        , unique_ptr_(std::move(src.unique_ptr_))
        , raw_ptr_(src.raw_ptr_)
#ifdef __GPU
        , unique_ptr_device_(std::move(src.unique_ptr_device_))
        , raw_ptr_device_(src.raw_ptr_device_)
#endif
    {
        for (int i = 0; i < N; i++) {
            dims_[i]    = src.dims_[i];
            offsets_[i] = src.offsets_[i];
        }
        src.raw_ptr_ = nullptr;
#ifdef __GPU
        src.raw_ptr_device_ = nullptr;
#endif
    }

    /// Move assigment operator
    inline mdarray<T, N>& operator=(mdarray<T, N>&& src)
    {
        if (this != &src) {
            label_       = src.label_;
            unique_ptr_  = std::move(src.unique_ptr_);
            raw_ptr_     = src.raw_ptr_;
            src.raw_ptr_ = nullptr;
#ifdef __GPU
            unique_ptr_device_  = std::move(src.unique_ptr_device_);
            raw_ptr_device_     = src.raw_ptr_device_;
            src.raw_ptr_device_ = nullptr;
#endif
            for (int i = 0; i < N; i++) {
                dims_[i]    = src.dims_[i];
                offsets_[i] = src.offsets_[i];
            }
        }
        return *this;
    }

    /// Allocate memory for array.
    inline mdarray<T, N>& allocate(memory_t memory__)
    {
        /* do nothing for zero-sized array */
        if (!this->size()) {
            return *this;
        }

        /* host allocation */
        if (is_host_memory(memory__)) {
            unique_ptr_ = get_unique_ptr<T>(this->size(), memory__);
            raw_ptr_    = unique_ptr_.get();
            call_constructor();
        }
#ifdef __GPU
        /* device allocation */
        if (is_device_memory(memory__)) {
            unique_ptr_device_ = get_unique_ptr<T>(this->size(), memory__);
            raw_ptr_device_    = unique_ptr_device_.get();
        }
#endif
        return *this;
    }

    /// Allocate memory from the pool.
    inline mdarray<T, N>& allocate(memory_pool& mp__)
    {
        /* do nothing for zero-sized array */
        if (!this->size()) {
            return *this;
        }
        /* host allocation */
        if (is_host_memory(mp__.memory_type())) {
            unique_ptr_ = mp__.get_unique_ptr<T>(this->size());
            raw_ptr_    = unique_ptr_.get();
            call_constructor();
        }
#ifdef __GPU
        /* device allocation */
        if (is_device_memory(mp__.memory_type())) {
            unique_ptr_device_ = mp__.get_unique_ptr<T>(this->size());
            raw_ptr_device_    = unique_ptr_device_.get();
        }
#endif
        return *this;
    }

    /// Deallocate host or device memory.
    inline void deallocate(memory_t memory__)
    {
        if (is_host_memory(memory__)) {
            /* call destructor for non-primitive objects */
            if (unique_ptr_) {
                call_destructor();
            }
            unique_ptr_.reset(nullptr);
            raw_ptr_ = nullptr;
        }
#ifdef __GPU
        if (is_device_memory(memory__)) {
            unique_ptr_device_.reset(nullptr);
            raw_ptr_device_ = nullptr;
        }
#endif
    }

    /// Access operator() for the elements of multidimensional array.
    template <typename... Args>
    inline T const& operator()(Args... args) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(args...)];
    }

    /// Access operator() for the elements of multidimensional array.
    template <typename... Args>
    inline T& operator()(Args... args)
    {
        return const_cast<T&>(static_cast<mdarray<T, N> const&>(*this)(args...));
    }

    /// Access operator[] for the elements of multidimensional array using a linear index in the range [0, size).
    inline T const& operator[](size_t const idx__) const
    {
        assert(idx__ >= 0 && idx__ < size());
        return raw_ptr_[idx__];
    }

    /// Access operator[] for the elements of multidimensional array using a linear index in the range [0, size).
    inline T& operator[](size_t const idx__)
    {
        return const_cast<T&>(static_cast<mdarray<T, N> const&>(*this)[idx__]);
    }

    template <typename... Args>
    inline T const* at(memory_t mem__, Args... args) const
    {
        return at_idx(mem__, idx(args...));
    }

    template <typename... Args>
    inline T* at(memory_t mem__, Args... args)
    {
        return const_cast<T*>(static_cast<mdarray<T, N> const&>(*this).at(mem__, args...));
    }

    /// Return pointer to the beginning of array.
    inline T const* at(memory_t mem__) const
    {
        return at_idx(mem__, 0);
    }

    /// Return pointer to the beginning of array.
    inline T* at(memory_t mem__)
    {
        return const_cast<T*>(static_cast<mdarray<T, N> const&>(*this).at(mem__));
    }

    /// Return total size (number of elements) of the array.
    inline size_t size() const
    {
        size_t size_{1};

        for (int i = 0; i < N; i++) {
            size_ *= dims_[i].size();
        }

        return size_;
    }

    /// Return size of particular dimension.
    inline size_t size(int i) const
    {
        mdarray_assert(i < N);
        return dims_[i].size();
    }

    /// Return a descriptor of a dimension.
    inline mdarray_index_descriptor dim(int i) const
    {
        mdarray_assert(i < N);
        return dims_[i];
    }

    /// Return leading dimension size.
    inline uint32_t ld() const
    {
        mdarray_assert(dims_[0].size() < size_t(1 << 31));

        return (int32_t)dims_[0].size();
    }

    /// Compute hash of the array
    /** Example: std::printf("hash(h) : %16llX\n", h.hash()); */
    inline uint64_t hash(uint64_t h__ = 5381) const
    {
        for (size_t i = 0; i < size() * sizeof(T); i++) {
            h__ = ((h__ << 5) + h__) + ((unsigned char*)raw_ptr_)[i];
        }

        return h__;
    }

    /// Compute weighted checksum.
    inline T checksum_w(size_t idx0__, size_t size__) const
    {
        T cs{0};
        for (size_t i = 0; i < size__; i++) {
            cs += raw_ptr_[idx0__ + i] * static_cast<double>((i & 0xF) - 8);
        }
        return cs;
    }

    /// Compute checksum.
    inline T checksum(size_t idx0__, size_t size__) const
    {
        T cs{0};
        for (size_t i = 0; i < size__; i++) {
            cs += raw_ptr_[idx0__ + i];
        }
        return cs;
    }

    inline T checksum() const
    {
        return checksum(0, size());
    }

    //== template <device_t pu>
    //== inline T checksum() const
    //== {
    //==     switch (pu) {
    //==         case CPU: {
    //==             return checksum();
    //==         }
    //==         case GPU: {
    //==            auto cs = acc::allocate<T>(1);
    //==            acc::zero(cs, 1);
    //==            add_checksum_gpu(raw_ptr_device_, (int)size(), 1, cs);
    //==            T cs1;
    //==            acc::copyout(&cs1, cs, 1);
    //==            acc::deallocate(cs);
    //==            return cs1;
    //==         }
    //==     }
    //== }

    /// Copy the content of the array to another array of identical size.
    /** For example:
        \code{.cpp}
        mdarray<double, 2> src(10, 20);
        mdarray<double, 2> dest(10, 20);
        src >> dest;
        \endcode
     */
    void operator>>(mdarray<T, N>& dest__) const
    {
        for (int i = 0; i < N; i++) {
            if (dest__.dims_[i].begin() != dims_[i].begin() || dest__.dims_[i].end() != dims_[i].end()) {
                std::printf("error at line %i of file %s: array dimensions don't match\n", __LINE__, __FILE__);
                raise(SIGTERM);
                exit(-1);
            }
        }
        std::memcpy(dest__.raw_ptr_, raw_ptr_, size() * sizeof(T));
    }

    /// Zero n elements starting from idx0.
    inline void zero(memory_t mem__, size_t idx0__, size_t n__)
    {
        mdarray_assert(idx0__ + n__ <= size());
        if (n__ && is_host_memory(mem__)) {
            mdarray_assert(raw_ptr_ != nullptr);
            //std::fill(raw_ptr_ + idx0__, raw_ptr_ + idx0__ + n__, 0);
            std::memset((void*)&raw_ptr_[idx0__], 0, n__ * sizeof(T));
        }
#ifdef __GPU
        if (n__ && on_device() && is_device_memory(mem__)) {
            mdarray_assert(raw_ptr_device_ != nullptr);
            acc::zero(&raw_ptr_device_[idx0__], n__);
        }
#endif
    }

    /// Zero the entire array.
    inline void zero(memory_t mem__ = memory_t::host)
    {
        zero(mem__, 0, size());
    }

    /// Copy n elements starting from idx0 from one memory type to another.
    inline void copy_to(memory_t mem__, size_t idx0__, size_t n__, stream_id sid = stream_id(-1))
    {
#ifdef __GPU
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);
        mdarray_assert(idx0__ + n__ <= size());
        if (is_host_memory(mem__) && is_device_memory(mem__)) {
            throw std::runtime_error("mdarray::copy_to(): memory is both host and device, check what to do with this case");
        }
        /* copy to device memory */
        if (is_device_memory(mem__)) {
            if (sid() == -1) {
                /* synchronous (blocking) copy */
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__);
            } else {
                /* asynchronous (non-blocking) copy */
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__, sid);
            }
        }
        /* copy back from device to host */
        if (is_host_memory(mem__)) {
            if (sid() == -1) {
                /* synchronous (blocking) copy */
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__);
            } else {
                /* asynchronous (non-blocking) copy */
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__, sid);
            }
        }
#endif
    }

    /// Copy entire array from one memory type to another.
    inline void copy_to(memory_t mem__, stream_id sid = stream_id(-1))
    {
        this->copy_to(mem__, 0, size(), sid);
    }

    /// Check if device pointer is available.
    inline bool on_device() const
    {
#ifdef __GPU
        return (raw_ptr_device_ != nullptr);
#else
        return false;
#endif
    }

    mdarray<T, N>& operator=(std::function<T(void)> f__)
    {
        for (size_t i = 0; i < this->size(); i++) {
            (*this)[i] = f__();
        }
        return *this;
    }

    mdarray<T, N>& operator=(std::function<T(index_type)> f__)
    {
        static_assert(N == 1, "wrong number of dimensions");

        for (index_type i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
            (*this)(i0) = f__(i0);
        }
        return *this;
    }

    mdarray<T, N>& operator=(std::function<T(index_type, index_type)> f__)
    {
        static_assert(N == 2, "wrong number of dimensions");

        for (index_type i1 = this->dims_[1].begin(); i1 <= this->dims_[1].end(); i1++) {
            for (index_type i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
                (*this)(i0, i1) = f__(i0, i1);
            }
        }
        return *this;
    }
};

// Alias for matrix
template <typename T>
using matrix = mdarray<T, 2>;

/// Serialize to std::ostream
template <typename T, int N>
std::ostream& operator<<(std::ostream& out, mdarray<T, N>& v)
{
    if (v.size()) {
        out << v[0];
        for (size_t i = 1; i < v.size(); i++) {
            out << std::string(" ") << v[i];
        }
    }
    return out;
}

}

#endif  // __MEMORY_HPP__

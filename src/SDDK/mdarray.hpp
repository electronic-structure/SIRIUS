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

/** \file mdarray.hpp
 *
 *  \brief Contains implementation of multidimensional array class.
 */

#ifndef __MDARRAY_HPP__
#define __MDARRAY_HPP__

#include <signal.h>
#include <cassert>
#include <memory>
#include <string>
#include <atomic>
#include <vector>
#include <array>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <functional>
#ifdef __GPU
#include "GPU/cuda.hpp"
#endif

namespace sddk {

//#ifdef __GPU
//extern "C" void add_checksum_gpu(cuDoubleComplex* wf__,
//                                 int num_rows_loc__,
//                                 int nwf__,
//                                 cuDoubleComplex* result__);
//#endif

#ifdef NDEBUG
#define mdarray_assert(condition__)
#else
#define mdarray_assert(condition__)                                 \
    {                                                               \
        if (!(condition__)) {                                       \
            printf("Assertion (%s) failed ", #condition__);         \
            printf("at line %i of file %s\n", __LINE__, __FILE__);  \
            printf("array label: %s\n", label_.c_str());            \
            for (int i = 0; i < N; i++)                             \
                printf("dim[%i].size = %li\n", i, dims_[i].size()); \
            raise(SIGTERM);                                         \
            exit(-13);                                              \
        }                                                           \
    }
#endif

/// Type of the main processing unit.
enum device_t
{
    /// CPU device.
    CPU = 0,

    /// GPU device (with CUDA programming model).
    GPU = 1
};

/// Type of memory.
/** Various combinations of flags can be used. To check for any host memory (pinned or non-pinned):
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
    none        = 0b000,
    /// Host memory.
    host        = 0b001,
    /// Pinned host memory. This is host memory + extra bit flag.
    host_pinned = 0b011,
    /// Device memory.
    device      = 0b100
};

inline constexpr memory_t operator&(memory_t a__, memory_t b__)
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) & static_cast<unsigned int>(b__));
}

inline constexpr memory_t operator|(memory_t a__, memory_t b__)
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
}

inline constexpr bool on_device(memory_t mem_type__)
{
    return (mem_type__ & memory_t::device) == memory_t::device ? true : false;
}

/// Index descriptor of mdarray.
class mdarray_index_descriptor
{
  private:
    /// Beginning of index.
    int64_t begin_{0};

    /// End of index.
    int64_t end_{-1};

    /// Size of index.
    size_t size_{0};

  public:
    /// Constructor of empty descriptor.
    mdarray_index_descriptor()
    {
    }

    /// Constructor for index range [0, size).
    mdarray_index_descriptor(size_t const size__)
        : begin_(0)
        , end_(size__ - 1)
        , size_(size__)
    {
    }

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(int64_t const begin__, int64_t const end__)
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
    inline int64_t begin() const
    {
        return begin_;
    }

    /// Return last index value.
    inline int64_t end() const
    {
        return end_;
    }

    /// Return index size.
    inline size_t size() const
    {
        return size_;
    }
};

struct mdarray_mem_count
{
    static std::atomic<int64_t>& allocated()
    {
        static std::atomic<int64_t> allocated_{0};
        return allocated_;
    }

    static std::atomic<int64_t>& allocated_max()
    {
        static std::atomic<int64_t> allocated_max_{0};
        return allocated_max_;
    }
};

/// Simple memory manager handler which keeps track of allocated and deallocated memory.
template <typename T>
struct mdarray_mem_mgr
{
    /// Number of elements of the current allocation.
    size_t size_{0};

    /// Type of allocated memory.
    memory_t mode_{memory_t::none};

    mdarray_mem_mgr()
    {
    }

    mdarray_mem_mgr(size_t const size__, memory_t mode__)
        : size_(size__)
        , mode_(mode__)
    {
        if ((mode_ & memory_t::host) == memory_t::host) {
            mdarray_mem_count::allocated() += size_ * sizeof(T);
            mdarray_mem_count::allocated_max() = std::max(mdarray_mem_count::allocated().load(),
                                                          mdarray_mem_count::allocated_max().load());
        }
    }

    /// Called by std::unique_ptr when the object is destroyed.
    void operator()(T* p__) const
    {
        if ((mode_ & memory_t::host) == memory_t::host) {
            mdarray_mem_count::allocated() -= size_ * sizeof(T);
            /* call destructor for non-primitive objects */
            if (!std::is_pod<T>::value) {
                for (size_t i = 0; i < size_; i++) {
                    (p__ + i)->~T();
                }
            }
        }

        /* host memory can be of two types */
        if ((mode_ & memory_t::host) == memory_t::host) {
            /* check if the memory is host pinned */
            if ((mode_ & memory_t::host_pinned) == memory_t::host_pinned) {
#ifdef __GPU
                acc::deallocate_host(p__);
#endif
            } else {
                free(p__);
            }
        }

        if ((mode_ & memory_t::device) == memory_t::device) {
#ifdef __GPU
            acc::deallocate(p__);
#endif
        }
    }
};

/// Base class of multidimensional array.
template <typename T, int N>
class mdarray_base
{
  protected:
    /// Optional array label.
    std::string label_;

    /// Unique pointer to the allocated memory.
    std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_{nullptr};

    /// Raw pointer.
    T* raw_ptr_{nullptr};

#ifdef __GPU
    /// Unique pointer to the allocated GPU memory.
    std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_device_{nullptr};

    /// Raw pointer to GPU memory
    T* raw_ptr_device_{nullptr};
#endif

    /// Array dimensions.
    std::array<mdarray_index_descriptor, N> dims_;

    /// List of offsets to compute the element location by dimension indices.
    std::array<int64_t, N> offsets_;

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

  private:
    inline int64_t idx(int64_t const i0) const
    {
        static_assert(N == 1, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        size_t i = offsets_[0] + i0;
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1) const
    {
        static_assert(N == 2, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1];
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2) const
    {
        static_assert(N == 3, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2];
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3) const
    {
        static_assert(N == 4, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        mdarray_assert(i3 >= dims_[3].begin() && i3 <= dims_[3].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2] + i3 * offsets_[3];
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3, int64_t const i4) const
    {
        static_assert(N == 5, "wrong number of dimensions");
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        mdarray_assert(i3 >= dims_[3].begin() && i3 <= dims_[3].end());
        mdarray_assert(i4 >= dims_[4].begin() && i4 <= dims_[4].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2] + i3 * offsets_[3] + i4 * offsets_[4];
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    template <device_t pu>
    inline T* at_idx(int64_t const idx__)
    {
        switch (pu) {
            case CPU: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case GPU: {
#ifdef __GPU
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                exit(0);
#endif
            }
        }
        return nullptr;
    }

    template <device_t pu>
    inline T const* at_idx(int64_t const idx__) const
    {
        switch (pu) {
            case CPU: {
                mdarray_assert(raw_ptr_ != nullptr);
                return &raw_ptr_[idx__];
            }
            case GPU: {
#ifdef __GPU
                mdarray_assert(raw_ptr_device_ != nullptr);
                return &raw_ptr_device_[idx__];
#else
                printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                exit(0);
#endif
            }
        }
        return nullptr;
    }

    /// Copy constructor is forbidden
    mdarray_base(mdarray_base<T, N> const& src) = delete;

    /// Assignment operator is forbidden
    mdarray_base<T, N>& operator=(mdarray_base<T, N> const& src) = delete;

  public:
    /// Constructor of an empty array.
    mdarray_base()
    {
    }

    /// Destructor.
    ~mdarray_base()
    {
    }

    /// Move constructor
    mdarray_base(mdarray_base<T, N>&& src)
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
    inline mdarray_base<T, N>& operator=(mdarray_base<T, N>&& src)
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
    void allocate(memory_t memory__)
    {
#ifndef __GPU
        if ((memory__ & memory_t::host_pinned) == memory_t::host_pinned) {
            memory__ = memory_t::host;
        }
#endif

        size_t sz = size();

        /* host allocation */
        if ((memory__ & memory_t::host) == memory_t::host) {
            /* page-locked memory */
            if ((memory__ & memory_t::host_pinned) == memory_t::host_pinned) {
#ifdef __GPU
                raw_ptr_    = acc::allocate_host<T>(sz);
                unique_ptr_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_, mdarray_mem_mgr<T>(sz, memory_t::host_pinned));
#endif
            } else { /* regular mameory */
                raw_ptr_    = static_cast<T*>(malloc(sz * sizeof(T)));
                unique_ptr_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_, mdarray_mem_mgr<T>(sz, memory_t::host));
            }

            /* call constructor on non-trivial data */
            if (raw_ptr_ != nullptr && !std::is_pod<T>::value) {
                for (size_t i = 0; i < sz; i++) {
                    new (raw_ptr_ + i) T();
                }
            }
        }

        /* device allocation */
#ifdef __GPU
        if ((memory__ & memory_t::device) == memory_t::device) {
            raw_ptr_device_    = acc::allocate<T>(sz);
            unique_ptr_device_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_device_, mdarray_mem_mgr<T>(sz, memory_t::device));
            
            //printf("GPU memory [%p, %p) is allocated for array %s\n", raw_ptr_device_, raw_ptr_device_ + sz, label_.c_str());
            //for (int i = 0; i < N; i++) {
            //    printf("dim[%i].size = %li\n", i, dims_[i].size());
            //}
        }
#endif
    }

    void deallocate(memory_t memory__)
    {
        if ((memory__ & memory_t::host) == memory_t::host) {
            if (unique_ptr_) {
                unique_ptr_.reset(nullptr);
                raw_ptr_ = nullptr;
            }
        }
#ifdef __GPU
        if ((memory__ & memory_t::device) == memory_t::device) {
            if (unique_ptr_device_) {
                unique_ptr_device_.reset(nullptr);
                raw_ptr_device_ = nullptr;
            }
        }
#endif
    }

    inline T& operator()(int64_t const i0)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0)];
    }

    inline T const& operator()(int64_t const i0) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1, i2)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1, int64_t const i2) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1, i2)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1, i2, i3)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1, i2, i3)];
    }

    inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3, int64_t const i4)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1, i2, i3, i4)];
    }

    inline T const& operator()(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3, int64_t const i4) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        return raw_ptr_[idx(i0, i1, i2, i3, i4)];
    }

    inline T& operator[](size_t const idx__)
    {
        mdarray_assert(idx__ >= 0 && idx__ < size());
        return raw_ptr_[idx__];
    }

    inline T const& operator[](size_t const idx__) const
    {
        assert(idx__ >= 0 && idx__ < size());
        return raw_ptr_[idx__];
    }

    template <device_t pu>
    inline T* at()
    {
        return at_idx<pu>(0);
    }

    template <device_t pu>
    inline T const* at() const
    {
        return at_idx<pu>(0);
    }

    template <device_t pu>
    inline T* at(int64_t const i0)
    {
        return at_idx<pu>(idx(i0));
    }

    template <device_t pu>
    inline T const* at(int64_t const i0) const
    {
        return at_idx<pu>(idx(i0));
    }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1)
    {
        return at_idx<pu>(idx(i0, i1));
    }

    template <device_t pu>
    inline T const* at(int64_t const i0, int64_t const i1) const
    {
        return at_idx<pu>(idx(i0, i1));
    }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1, int64_t const i2)
    {
        return at_idx<pu>(idx(i0, i1, i2));
    }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3)
    {
        return at_idx<pu>(idx(i0, i1, i2, i3));
    }

    template <device_t pu>
    inline T* at(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3, int64_t const i4)
    {
        return at_idx<pu>(idx(i0, i1, i2, i3, i4));
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

    /// Return leading dimension size.
    inline uint32_t ld() const
    {
        mdarray_assert(dims_[0].size() < size_t(1 << 31));

        return (int32_t)dims_[0].size();
    }

    /// Compute hash of the array
    /** Example: printf("hash(h) : %16llX\n", h.hash()); */
    inline uint64_t hash(uint64_t h__ = 5381) const
    {
        for (size_t i = 0; i < size() * sizeof(T); i++) {
            h__ = ((h__ << 5) + h__) + ((unsigned char*)raw_ptr_)[i];
        }

        return h__;
    }

    inline T checksum_w(size_t idx0__, size_t size__) const
    {
        T cs{0};
        for (size_t i = 0; i < size__; i++) {
            cs += raw_ptr_[idx0__ + i] * static_cast<double>((i & 0xF) - 8);
        }
        return cs;
    }

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

    /// Copy the content of the array to dest
    void operator>>(mdarray_base<T, N>& dest__) const
    {
        for (int i = 0; i < N; i++) {
            if (dest__.dims_[i].begin() != dims_[i].begin() || dest__.dims_[i].end() != dims_[i].end()) {
                printf("error at line %i of file %s: array dimensions don't match\n", __LINE__, __FILE__);
                raise(SIGTERM);
                exit(-1);
            }
        }
        std::memcpy(dest__.raw_ptr_, raw_ptr_, size() * sizeof(T));
    }

    /// Copy n elements starting from idx0.
    template <memory_t from__, memory_t to__>
    inline void copy(size_t idx0__, size_t n__, int stream_id__ = -1)
    {
#ifdef __GPU
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);
        mdarray_assert(idx0__ + n__ <= size());

        if ((from__ & memory_t::host) == memory_t::host && (to__ & memory_t::device) == memory_t::device) {
            if (stream_id__ == -1) {
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__);
            } else {
                acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__, stream_id__);
            }
        }

        if ((from__ & memory_t::device) == memory_t::device && (to__ & memory_t::host) == memory_t::host) {
            if (stream_id__ == -1) {
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__);
            } else {
                acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__, stream_id__);
            }
        }
#endif
    }

    template <memory_t from__, memory_t to__>
    inline void copy(size_t n__)
    {
        copy<from__, to__>(0, n__);
    }

    template <memory_t from__, memory_t to__>
    inline void async_copy(size_t n__, int stream_id__)
    {
        copy<from__, to__>(0, n__, stream_id__);
    }

    template <memory_t from__, memory_t to__>
    inline void copy()
    {
        copy<from__, to__>(0, size());
    }

    template <memory_t from__, memory_t to__>
    inline void async_copy(int stream_id__)
    {
        copy<from__, to__>(0, size(), stream_id__);
    }

    /// Zero n elements starting from idx0.
    template <memory_t mem_type__>
    inline void zero(size_t idx0__, size_t n__)
    {
        mdarray_assert(idx0__ + n__ <= size());
        if (((mem_type__ & memory_t::host) == memory_t::host) && n__) {
            mdarray_assert(raw_ptr_ != nullptr);
            std::memset(&raw_ptr_[idx0__], 0, n__ * sizeof(T));
        }
#ifdef __GPU
        if (((mem_type__ & memory_t::device) == memory_t::device) && on_device() && n__) {
            mdarray_assert(raw_ptr_device_ != nullptr);
            acc::zero(&raw_ptr_device_[idx0__], n__);
        }
#endif
    }

    template <memory_t mem_type__ = memory_t::host>
    inline void zero()
    {
        zero<mem_type__>(0, size());
    }

    inline bool on_device() const
    {
#ifdef __GPU
        return (raw_ptr_device_ != nullptr);
#else
        return false;
#endif
    }
};

/// Multidimensional array with the column-major (Fortran) order.
template <typename T, int N>
class mdarray : public mdarray_base<T, N>
{
  public:
    mdarray()
    {
    }

    mdarray(mdarray_index_descriptor const& d0,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
        static_assert(N == 1, "wrong number of dimensions");

        this->label_ = label__;
        this->init_dimensions({d0});
        this->allocate(memory__);
    }

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

    mdarray<T, N>& operator=(std::function<T(int64_t)> f__)
    {
        static_assert(N == 1, "wrong number of dimensions");

        for (int64_t i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
            (*this)(i0) = f__(i0);
        }
        return *this;
    }

    mdarray<T, N>& operator=(std::function<T(int64_t, int64_t)> f__)
    {
        static_assert(N == 2, "wrong number of dimensions");

        for (int64_t i1 = this->dims_[1].begin(); i1 <= this->dims_[1].end(); i1++) {
            for (int64_t i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
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

} // namespace sddk

#endif // __MDARRAY_HPP__

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
#ifdef __GPU
#include "gpu.h"
#endif

namespace sddk {

#ifdef __GPU
extern "C" void add_checksum_gpu(cuDoubleComplex* wf__,
                                 int num_rows_loc__,
                                 int nwf__,
                                 cuDoubleComplex* result__);
#endif

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

/// Type of the main processing unit
enum device_t
{
    /// use CPU
    CPU = 0,

    /// use GPU (with CUDA programming model)
    GPU = 1
};

enum class memory_t : unsigned int
{
    none        = 0,
    host        = (1 << 0),
    host_pinned = (1 << 1),
    device      = (1 << 2)
};

inline constexpr memory_t operator&(memory_t a__, memory_t b__)
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) & static_cast<unsigned int>(b__));
}

inline constexpr memory_t operator|(memory_t a__, memory_t b__)
{
    return static_cast<memory_t>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
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
        : begin_(0),
          end_(size__ - 1),
          size_(size__)
    {
    }

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(int64_t const begin__, int64_t const end__)
        : begin_(begin__),
          end_(end__),
          size_(end_ - begin_ + 1)
    {
        assert(end_ >= begin_);
    };

    /// Constructor for index range [begin, end]
    mdarray_index_descriptor(std::pair<int, int> const range__)
        : begin_(range__.first),
          end_(range__.second),
          size_(end_ - begin_ + 1)
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

struct mdarray_mem_count // TODO: not clear if std::atomic can be mixed with openmp
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

/// Simple mameory manager handler which keeps track of allocated and deallocated memory.
template <typename T>
struct mdarray_mem_mgr
{
    /// Number of elements of the current allocation.
    size_t size_{0};

    memory_t mode_{memory_t::none};

    mdarray_mem_mgr()
    {
    }

    mdarray_mem_mgr(size_t const size__, memory_t mode__)
        : size_(size__),
          mode_(mode__)
    {
        if ((mode_ & memory_t::host) != memory_t::none || (mode_ & memory_t::host_pinned) != memory_t::none) {
            mdarray_mem_count::allocated() += size_ * sizeof(T);
            mdarray_mem_count::allocated_max() = std::max(mdarray_mem_count::allocated().load(),
                                                          mdarray_mem_count::allocated_max().load());
        }
    }

    /// Called by std::unique_ptr when the object is destroyed.
    void operator()(T* p__) const
    {
        if ((mode_ & memory_t::host) != memory_t::none || (mode_ & memory_t::host_pinned) != memory_t::none) {
            mdarray_mem_count::allocated() -= size_ * sizeof(T);
            if (!std::is_pod<T>::value) {
                for (size_t i = 0; i < size_; i++) {
                    (p__ + i)->~T();
                }
            }
        }

        if ((mode_ & memory_t::host) != memory_t::none) {
            free(p__);
        }

        if ((mode_ & memory_t::host_pinned) != memory_t::none) {
            #ifdef __GPU
            cuda_free_host(p__);
            #endif
        }

        if ((mode_ & memory_t::device) != memory_t::none) {
            #ifdef __GPU
            cuda_free(p__);
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
    mutable std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_{nullptr};

    /// Raw pointer.
    mutable T* raw_ptr_{nullptr};

    #ifdef __GPU
    /// Unique pointer to the allocated GPU memory.
    mutable std::unique_ptr<T[], mdarray_mem_mgr<T>> unique_ptr_device_{nullptr};

    /// Raw pointer to GPU memory
    mutable T* raw_ptr_device_{nullptr};
    #endif

    /// Array dimensions.
    std::array<mdarray_index_descriptor, N> dims_;

    /// List of offsets to compute the element location by dimension indices.
    std::array<int64_t, N> offsets_;

    void init_dimensions(std::initializer_list<mdarray_index_descriptor> const args)
    {
        assert(args.size() == N);

        int i{0};
        for (auto d : args) {
            dims_[i++] = d;
        }

        offsets_[0] = -dims_[0].begin();
        size_t ld   = 1;
        for (int i = 1; i < N; i++) {
            ld *= dims_[i - 1].size();
            offsets_[i] = ld;
            offsets_[0] -= ld * dims_[i].begin();
        }
    }

  private:
    inline int64_t idx(int64_t const i0) const
    {
        mdarray_assert(N == 1);
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        size_t i = offsets_[0] + i0;
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1) const
    {
        mdarray_assert(N == 2);
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1];
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2) const
    {
        mdarray_assert(N == 3);
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2];
        mdarray_assert(i >= 0 && i < size());
        return i;
    }

    inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3) const
    {
        mdarray_assert(N == 4);
        mdarray_assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
        mdarray_assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
        mdarray_assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
        mdarray_assert(i3 >= dims_[3].begin() && i3 <= dims_[3].end());
        size_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2] + i3 * offsets_[3];
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
    void allocate(memory_t memory__) const
    {
        if ((memory__ & memory_t::host) != memory_t::none && (memory__ & memory_t::host_pinned) != memory_t::none) {
            printf("error at line %i of file %s: host memory can only be of one type\n", __LINE__, __FILE__);
            exit(0);
        }

        #ifndef __GPU
        if ((memory__ & memory_t::host_pinned) != memory_t::none) {
            memory__ = memory_t::host;
        }
        #endif

        size_t sz = size();

        /* host allocation */
        if ((memory__ & memory_t::host) != memory_t::none || (memory__ & memory_t::host_pinned) != memory_t::none) {
            if ((memory__ & memory_t::host) != memory_t::none) {
                raw_ptr_    = static_cast<T*>(malloc(sz * sizeof(T)));
                unique_ptr_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_, mdarray_mem_mgr<T>(sz, memory_t::host));
            }

            if ((memory__ & memory_t::host_pinned) != memory_t::none) {
                #ifdef __GPU
                raw_ptr_    = static_cast<T*>(cuda_malloc_host(sz * sizeof(T)));
                unique_ptr_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_, mdarray_mem_mgr<T>(sz, memory_t::host_pinned));
                #endif
            }

            /* call constructor on non-trivial data */
            if (raw_ptr_ != nullptr && !std::is_pod<T>::value) {
                for (size_t i = 0; i < sz; i++) {
                    new (raw_ptr_ + i) T();
                }
            }
        }

        /* device allocation */
        if ((memory__ & memory_t::device) != memory_t::none) {
            #ifdef __GPU
            raw_ptr_device_    = static_cast<T*>(cuda_malloc(sz * sizeof(T)));
            unique_ptr_device_ = std::unique_ptr<T[], mdarray_mem_mgr<T>>(raw_ptr_device_, mdarray_mem_mgr<T>(sz, memory_t::device));
            #endif
        }
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

    inline void zero()
    {
        if (size() > 0) {
            mdarray_assert(raw_ptr_ != nullptr);
            std::memset(raw_ptr_, 0, size() * sizeof(T));
        }
    }

    /// Compute hash of the array
    /** Example: printf("hash(h) : %16llX\n", h.hash()); */
    inline uint64_t hash() const
    {
        uint64_t h{5381};

        for (size_t i = 0; i < size() * sizeof(T); i++) {
            h = ((h << 5) + h) + ((unsigned char*)raw_ptr_)[i];
        }

        return h;
    }

    inline T checksum() const
    {
        T cs{0};
        for (size_t i = 0; i < size(); i++) {
            cs += raw_ptr_[i];
        }
        return cs;
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
    inline void copy(size_t idx0__, size_t n__)
    {
        #ifdef __GPU
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);
        mdarray_assert(idx0__ + n__ <= size());

        if ((from__ & memory_t::host) != memory_t::none && (to__ & memory_t::device) != memory_t::none) {
            acc::copyin(&raw_ptr_device_[idx0__], &raw_ptr_[idx0__], n__);
        }

        if ((from__ & memory_t::device) != memory_t::none && (to__ & memory_t::host) != memory_t::none) {
            acc::copyout(&raw_ptr_[idx0__], &raw_ptr_device_[idx0__], n__);
        }
        #endif
    }

    template <memory_t from__, memory_t to__>
    inline void copy(size_t n__)
    {
        copy<from__, to__>(0, n__);
    }

    template <memory_t from__, memory_t to__>
    inline void copy()
    {
        copy<from__, to__>(0, size());
    }

    /// Zero n elements starting from idx0.
    template <memory_t mem_type__>
    inline void zero(size_t idx0__, size_t n__)
    {
        assert(idx0__ + n__ <= size());
        if ((mem_type__ & memory_t::host) != memory_t::none && n__) {
            mdarray_assert(raw_ptr_ != nullptr);
            std::memset(&raw_ptr_[idx0__], 0, n__ * sizeof(T));
        }
        #ifdef __GPU
        if ((mem_type__ & memory_t::device) != memory_t::none && on_device() && n__) {
            acc::zero(&raw_ptr_device_[idx0__], n__);
        }
        #endif
    }


    #ifdef __GPU
    void deallocate_on_device() const
    {
        unique_ptr_device_.reset(nullptr);
        raw_ptr_device_ = nullptr;
    }

    void copy_to_device() const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);

        acc::copyin(raw_ptr_device_, raw_ptr_, size());
    }

    void copy_to_device(size_t n__) const
    {
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);

        acc::copyin(raw_ptr_device_, raw_ptr_, n__);
    }

    void copy_to_host()
    {
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);

        acc::copyout(raw_ptr_, raw_ptr_device_, size());
    }

    void copy_to_host(size_t n__)
    {
        mdarray_assert(raw_ptr_ != nullptr);
        mdarray_assert(raw_ptr_device_ != nullptr);

        acc::copyout(raw_ptr_, raw_ptr_device_, n__);
    }

    void async_copy_to_device(int stream_id__ = -1) const
    {
        acc::copyin(raw_ptr_device_, raw_ptr_, size(), stream_id__);
    }

    void async_copy_to_host(int stream_id__ = -1)
    {
        acc::copyout(raw_ptr_, raw_ptr_device_, size(), stream_id__);
    }

    void zero_on_device()
    {
        acc::zero(raw_ptr_device_, size());
    }
    #endif

    inline bool on_device() const
    {
        #ifdef __GPU
        return (raw_ptr_device_ != nullptr);
        #else
        return false;
        #endif
    }
};

/// Multidimensional array.
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
        this->label_ = label__;
        this->init_dimensions({d0});
        this->allocate(memory__);
    }

    mdarray(mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            memory_t memory__   = memory_t::host,
            std::string label__ = "")
    {
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
        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3});
        this->allocate(memory__);
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
        this->label_ = label__;
        this->init_dimensions({d0});
        this->raw_ptr_ = ptr__;
    }

    mdarray(T* ptr__,
            T* ptr_device__,
            mdarray_index_descriptor const& d0,
            std::string label__ = "")
    {
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
        this->label_ = label__;
        this->init_dimensions({d0, d1, d2});
        this->ptr_ = ptr__;
        #ifdef __GPU
        this->ptr_device_ = ptr_device__;
        #endif
    }

    mdarray(T* ptr__,
            mdarray_index_descriptor const& d0,
            mdarray_index_descriptor const& d1,
            mdarray_index_descriptor const& d2,
            mdarray_index_descriptor const& d3,
            std::string label__ = "")
    {
        this->label_ = label__;
        this->init_dimensions({d0, d1, d2, d3});
        this->raw_ptr_ = ptr__;
    }

    mdarray<T, N>& operator=(std::function<T(int64_t)> f__)
    {
        assert(N == 1);

        for (int64_t i0 = this->dims_[0].begin(); i0 <= this->dims_[0].end(); i0++) {
            (*this)(i0) = f__(i0);
        }
        return *this;
    }

    mdarray<T, N>& operator=(std::function<T(int64_t, int64_t)> f__)
    {
        assert(N == 2);

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

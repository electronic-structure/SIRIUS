// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file mdarray.h
 *   
 *  \brief Contains implementation of mdarray class.
 */

#ifndef __MDARRAY_H__
#define __MDARRAY_H__

#include <signal.h>
#ifdef __GLIBC__
#include <execinfo.h>
#endif
#include <cassert>
#include <memory>
#include <atomic>
#include <vector>
#include <cstring>
#include <initializer_list>
#ifdef _GPU_
#include "gpu_interface.h"
#endif
#include "typedefs.h"

#ifdef NDEBUG
  #define mdarray_assert(condition__)
#else
  #ifdef __GLIBC__
    #define mdarray_assert(condition__)                                         \
    {                                                                           \
        if (!(condition__))                                                     \
        {                                                                       \
            printf("Assertion (%s) failed ", #condition__);                     \
            printf("at line %i of file %s\n", __LINE__, __FILE__);              \
            for (int i = 0; i < N; i++)                                         \
                printf("dim[%i].size = %li\n", i, dims_[i].size());             \
            void *array[10];                                                    \
            char **strings;                                                     \
            auto size = backtrace(array, 10);                                   \
            strings = backtrace_symbols(array, size);                           \
            printf ("Stack backtrace:\n");                                      \
            for (size_t i = 0; i < size; i++)                                   \
                printf ("%s\n", strings[i]);                                    \
            raise(SIGTERM);                                                     \
            exit(-13);                                                          \
        }                                                                       \
    }
  #else
    #define mdarray_assert(condition__)                                         \
    {                                                                           \
        if (!(condition__))                                                     \
        {                                                                       \
            printf("Assertion (%s) failed ", #condition__);                     \
            printf("at line %i of file %s\n", __LINE__, __FILE__);              \
            for (int i = 0; i < N; i++)                                         \
                printf("dim[%i].size = %li\n", i, dims_[i].size());             \
            raise(SIGTERM);                                                     \
            exit(-13);                                                          \
        }                                                                       \
    }
  #endif
#endif

/// Type of the main processing unit
enum processing_unit_t 
{
    /// use CPU
    CPU = 0, 

    /// use GPU (with CUDA programming model)
    GPU = 1
};

/// Index descriptor of mdarray.
class mdarray_index_descriptor
{
    private:
        
        /// Beginning of index.
        int64_t begin_;

        /// End of index.
        int64_t end_;

        /// Size of index.
        size_t size_;

    public:
  
        /// Constructor of empty descriptor.
        mdarray_index_descriptor() : begin_(0), end_(-1), size_(0)
        {
        }
        
        /// Constructor for index range [0, size).
        mdarray_index_descriptor(size_t const size__) : begin_(0), end_(size__ - 1), size_(size__)
        {
        }
    
        /// Constructor for index range [begin, end]
        mdarray_index_descriptor(int64_t const begin__, int64_t const end__)
            : begin_(begin__), 
              end_(end__) , 
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

struct mdarray_mem_count
{
    static std::atomic<int64_t>& allocated()
    {
        static std::atomic<int64_t> allocated_(0);
        return allocated_;
    }

    static std::atomic<int64_t>& allocated_max()
    {
        static std::atomic<int64_t> allocated_max_(0);
        return allocated_max_;
    }
};

/// Simple mameory manager handler which keeps track of allocated and deallocated memory.
template<typename T>
struct mdarray_mem_mgr
{
    /// Number of elements of the current allocation.
    size_t size_;

    int mode_;
    
    mdarray_mem_mgr() : size_(0), mode_(0)
    {
    }

    mdarray_mem_mgr(size_t const size__, int const mode__) : size_(size__), mode_(mode__)
    {
        if (mode_ == 0 || mode_ == 1)
        {
            mdarray_mem_count::allocated() += size_ * sizeof(T);
            mdarray_mem_count::allocated_max() = std::max(mdarray_mem_count::allocated().load(), 
                                                          mdarray_mem_count::allocated_max().load());
        }
    }
    
    /// Called by std::unique_ptr when the object is destroyed.
    void operator()(T* p__) const
    {
        if (mode_ == 0 || mode_ == 1)
        {
            mdarray_mem_count::allocated() -= size_ * sizeof(T);
        }

        switch (mode_)
        {
            case 0:
            {
                delete[] p__;
                break;
            }
            #ifdef _GPU_
            case 1:
            {
                cuda_free_host(p__);
                break;
            }
            case 2:
            {
                cuda_free(p__);
                break;
            }
            #endif
            default:
            {
                printf("error at line %i of file %s: wrong delete mode\n", __LINE__, __FILE__);
                raise(SIGTERM);
                exit(-1);
            }
        }
    }
};

/// Base class of multidimensional array.
template <typename T, int N> 
class mdarray_base
{
    protected:
    
        /// Unique pointer to the allocated memory.
        std::unique_ptr<T[], mdarray_mem_mgr<T> > unique_ptr_;
        
        /// Raw pointer.
        T* ptr_;
        
        #ifdef _GPU_
        /// Unique pointer to the allocated GPU memory.
        std::unique_ptr<T[], mdarray_mem_mgr<T> > unique_ptr_device_;
        
        /// Raw pointer to GPU memory
        T* ptr_device_;  
        #endif

        bool pinned_;
        
        /// Array dimensions.
        mdarray_index_descriptor dims_[N];
        
        /// List of offsets to compute the element location by dimension indices. 
        int64_t offsets_[N];

        void init_dimensions(std::initializer_list<mdarray_index_descriptor> const args)
        {
            assert(args.size() == N);

            int i = 0;
            for (auto d: args) dims_[i++] = d;
            
            offsets_[0] = -dims_[0].begin();
            size_t ld = 1;
            for (int i = 1; i < N; i++) 
            {
                ld *= dims_[i - 1].size();
                offsets_[i] = ld;
                offsets_[0] -= ld * dims_[i].begin();
            }
        }

    private:

        inline int64_t idx(int64_t const i0) const
        {
            assert(N == 1);
            assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
            int64_t i = offsets_[0] + i0;
            assert(i >= 0 && i < size());
            return i;
        }

        inline int64_t idx(int64_t const i0, int64_t const i1) const
        {
            assert(N == 2);
            assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
            assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
            int64_t i = offsets_[0] + i0 + i1 * offsets_[1];
            assert(i >= 0 && i < size());
            return i;
        }

        inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2) const
        {
            assert(N == 3);
            assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
            assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
            assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
            int64_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2];
            assert(i >= 0 && i < size());
            return i;
        }

        inline int64_t idx(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3) const
        {
            assert(N == 4);
            assert(i0 >= dims_[0].begin() && i0 <= dims_[0].end());
            assert(i1 >= dims_[1].begin() && i1 <= dims_[1].end());
            assert(i2 >= dims_[2].begin() && i2 <= dims_[2].end());
            assert(i3 >= dims_[3].begin() && i3 <= dims_[3].end());
            int64_t i = offsets_[0] + i0 + i1 * offsets_[1] + i2 * offsets_[2] + i3 * offsets_[3];
            assert(i >= 0 && i < size());
            return i;
        }

        template<processing_unit_t pu>
        inline T* at_idx(int64_t const idx__)
        {
            switch (pu)
            {
                case CPU:
                {
                    mdarray_assert(ptr_ != nullptr);
                    return &ptr_[idx__];
                }
                case GPU:
                {
                    #ifdef _GPU_
                    mdarray_assert(ptr_device_ != nullptr);
                    return &ptr_device_[idx__];
                    #else
                    printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                    exit(0);
                    #endif
                }
            }
            return nullptr;
        }

        template<processing_unit_t pu>
        inline T const* at_idx(int64_t const idx__) const
        {
            switch (pu)
            {
                case CPU:
                {
                    mdarray_assert(ptr_ != nullptr);
                    return &ptr_[idx__];
                }
                case GPU:
                {
                    #ifdef _GPU_
                    mdarray_assert(ptr_device_ != nullptr);
                    return &ptr_device_[idx__];
                    #else
                    printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                    exit(0);
                    #endif
                }
            }
            return nullptr;
        }

    public:
        
        /// Constructor of an empty array.
        mdarray_base() 
            : unique_ptr_(nullptr),
              ptr_(nullptr),
              #ifdef _GPU_
              unique_ptr_device_(nullptr),
              ptr_device_(nullptr), 
              #endif
              pinned_(false)
        { 
        }
        
        /// Destructor.
        ~mdarray_base()
        {
            deallocate();
            #ifdef _GPU_
            deallocate_on_device();
            #endif
        }

        /// Copy constructor is forbidden
        mdarray_base(mdarray_base<T, N> const& src) = delete;
        
        /// Assignment operator is forbidden
        mdarray_base<T, N>& operator=(mdarray_base<T, N> const& src) = delete;
        
        /// Move constructor
        mdarray_base(mdarray_base<T, N>&& src) 
            : unique_ptr_(std::move(src.unique_ptr_)),
              ptr_(src.ptr_),
              #ifdef _GPU_
              unique_ptr_device_(std::move(src.unique_ptr_device_)),
              ptr_device_(src.ptr_device_),
              #endif
              pinned_(src.pinned_)
        {
            src.pinned_ = false;
            for (int i = 0; i < N; i++)
            {
                dims_[i] = src.dims_[i];
                offsets_[i] = src.offsets_[i];
            }
        }

        /// Move assigment operator
        inline mdarray_base<T, N>& operator=(mdarray_base<T, N>&& src)
        {
            if (this != &src)
            {
                unique_ptr_ = std::move(src.unique_ptr_);
                ptr_ = src.ptr_;
                #ifdef _GPU_
                unique_ptr_device_ = std::move(src.unique_ptr_device_);
                ptr_device_ = src.ptr_device_;
                #endif
                pinned_ = src.pinned_;
                src.pinned_ = false;
                for (int i = 0; i < N; i++)
                {
                    dims_[i] = src.dims_[i];
                    offsets_[i] = src.offsets_[i];
                }
            }
            return *this;
        }

        inline T& operator()(int64_t const i0) 
        {
            assert(ptr_ != nullptr);
            return ptr_[idx(i0)];
        }

        inline T const& operator()(int64_t const i0) const
        {
            assert(ptr_ != nullptr);
            return ptr_[idx(i0)];
        }

        inline T& operator()(int64_t const i0, int64_t const i1) 
        {
            mdarray_assert(ptr_ != nullptr);
            return ptr_[idx(i0, i1)];
        }

        inline T const& operator()(int64_t const i0, int64_t const i1) const
        {
            mdarray_assert(ptr_ != nullptr);
            return ptr_[idx(i0, i1)];
        }

        inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2) 
        {
            assert(ptr_ != nullptr);
            return ptr_[idx(i0, i1, i2)];
        }

        inline T const& operator()(int64_t const i0, int64_t const i1, int64_t const i2) const
        {
            assert(ptr_ != nullptr);
            return ptr_[idx(i0, i1, i2)];
        }

        inline T& operator()(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3)
        {
            assert(ptr_ != nullptr);
            return ptr_[idx(i0, i1, i2, i3)];
        }

        inline T const& operator()(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3) const
        {
            assert(ptr_ != nullptr);
            return ptr_[idx(i0, i1, i2, i3)];
        }

        template <processing_unit_t pu>
        inline T* at()
        {
            return at_idx<pu>(0);
        }

        template <processing_unit_t pu>
        inline T const* at() const
        {
            return at_idx<pu>(0);
        }

        template <processing_unit_t pu>
        inline T* at(int64_t const i0)
        {
            return at_idx<pu>(idx(i0));
        }

        template <processing_unit_t pu>
        inline T* at(int64_t const i0, int64_t const i1)
        {
            return at_idx<pu>(idx(i0, i1));
        }

        template <processing_unit_t pu>
        inline T const* at(int64_t const i0, int64_t const i1) const
        {
            return at_idx<pu>(idx(i0, i1));
        }

        template <processing_unit_t pu>
        inline T* at(int64_t const i0, int64_t const i1, int64_t const i2)
        {
            return at_idx<pu>(idx(i0, i1, i2));
        }

        template <processing_unit_t pu>
        inline T* at(int64_t const i0, int64_t const i1, int64_t const i2, int64_t const i3)
        {
            return at_idx<pu>(idx(i0, i1, i2, i3));
        }

        /// Return total size (number of elements) of the array.
        inline size_t size() const
        {
            size_t size_ = 1;

            for (int i = 0; i < N; i++) size_ *= dims_[i].size();

            return size_;
        }
        
        /// Return size of particular dimension.
        inline size_t size(int i) const
        {
           assert(i < N);
           return dims_[i].size();
        }
        
        /// Return leading dimension size.
        inline uint32_t ld() const
        {
            assert(dims_[0].size() < size_t(1 << 31));

            return (int32_t)dims_[0].size();
        }

        /// Allocate memory for array.
        /** mode = 0: normal allocation of CPU memory \n
         *  mode = 1: page-locked allocation of CPU memory \n
         */
        void allocate(int mode__ = 0)
        {
            deallocate();

            size_t sz = size();

            if (mode__ == 0)
            {
                unique_ptr_ = std::unique_ptr< T[], mdarray_mem_mgr<T> >(new T[sz], mdarray_mem_mgr<T>(sz, 0));
                ptr_ = unique_ptr_.get();
            }

            if (mode__ == 1)
            {
                #ifdef _GPU_
                ptr_ = static_cast<T*>(cuda_malloc_host(sz * sizeof(T)));
                unique_ptr_ = std::unique_ptr< T[], mdarray_mem_mgr<T> >(ptr_, mdarray_mem_mgr<T>(sz, 1));
                #else
                printf("error at line %i of file %s: not compiled with GPU support\n", __LINE__, __FILE__);
                exit(0);
                #endif
            }
        }
        
        /// Deallocate memory and reset pointers.
        void deallocate()
        {
            #ifdef _GPU_
            unpin_memory();
            #endif
            unique_ptr_.reset(nullptr);
            ptr_ = nullptr;
        }
        
        inline void zero()
        {
            if (size() > 0)
            {
                assert(ptr_ != nullptr);
                memset(ptr_, 0, size() * sizeof(T));
            }
        }

        /// Set raw pointer.
        inline void set_ptr(T* ptr__)
        {
            ptr_ = ptr__;
        }
        
        /// Return raw pointer.
        inline T* ptr()
        {
            return ptr_;
        }
        
        /// Compute hash of the array
        /** Example: printf("hash(h) : %16llX\n", h.hash()); */
        uint64_t hash() const
        {
            uint64_t h = 5381;

            for (size_t i = 0; i < size() * sizeof(T); i++) h = ((h << 5) + h) + ((unsigned char*)ptr_)[i];

            return h;
        }
        
        /// Copy the content of the array to dest
        void operator>>(mdarray_base<T, N>& dest__) const
        {
            for (int i = 0; i < N; i++) 
            {
                if (dest__.dims_[i].begin() != dims_[i].begin() || dest__.dims_[i].end() != dims_[i].end())
                {
                    printf("error at line %i of file %s: array dimensions don't match\n", __LINE__, __FILE__);
                    raise(SIGTERM);
                    exit(-1);
                }
            }
            memcpy(dest__.ptr(), ptr_, size() * sizeof(T));
        }

        #ifdef _GPU_
        void allocate_on_device()
        {
            size_t sz = size();

            ptr_device_ = (T*)cuda_malloc(sz * sizeof(T));

            unique_ptr_device_ = std::unique_ptr<T[], mdarray_mem_mgr<T> >(ptr_device_, mdarray_mem_mgr<T>(sz, 2));
        }

        void deallocate_on_device()
        {
            unique_ptr_device_.reset(nullptr);
            ptr_device_ = nullptr;
        }

        void allocate_page_locked()
        {
            allocate(1);
        }

        void copy_to_device() 
        {
            assert(ptr_ != nullptr);
            assert(ptr_device_ != nullptr);

            cuda_copy_to_device(ptr_device_, ptr_, size() * sizeof(T));
        }

        void copy_to_host() 
        {
            assert(ptr_ != nullptr);
            assert(ptr_device_ != nullptr);
            
            cuda_copy_to_host(ptr_, ptr_device_, size() * sizeof(T));
        }

        void async_copy_to_device(int stream_id__ = -1) 
        {
            cuda_async_copy_to_device(ptr_device_, ptr_, size() * sizeof(T), stream_id__);
        }
        
        void async_copy_to_host(int stream_id__ = -1) 
        {
            cuda_async_copy_to_host(ptr_, ptr_device_, size() * sizeof(T), stream_id__);
        }

        void zero_on_device()
        {
            cuda_memset(ptr_device_, 0, size() * sizeof(T));
        }

        void pin_memory()
        {
            if (pinned_)
            {
                printf("error at line %i of file %s: array is already pinned\n", __LINE__, __FILE__);
                raise(SIGTERM);                                                                                            \
                exit(-1);
            }
            cuda_host_register(ptr_, size() * sizeof(T));
            pinned_ = true;
        }
        
        void unpin_memory()
        {
            if (pinned_)
            {
                cuda_host_unregister(ptr_);
                pinned_ = false;
            }
        }

        /// Set raw device pointer.
        inline void set_ptr_device(T* ptr_device__)
        {
            ptr_device_ = ptr_device__;
        }
        #endif
};


template <typename T, int N> 
class mdarray: public mdarray_base<T, N>
{
    public:

        mdarray()
        {
        }
        
        mdarray(mdarray_index_descriptor const& d0)
        {
            this->init_dimensions({d0});
            this->allocate();
        }

        mdarray(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            this->init_dimensions({d0, d1});
            this->allocate();
        }

        mdarray(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1,
                mdarray_index_descriptor const& d2)
        {
            this->init_dimensions({d0, d1, d2});
            this->allocate();
        }

        mdarray(mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1,
                mdarray_index_descriptor const& d2, mdarray_index_descriptor const& d3)
        {
            this->init_dimensions({d0, d1, d2, d3});
            this->allocate();
        }

        mdarray(T* ptr__, mdarray_index_descriptor const& d0)
        {
            this->init_dimensions({d0});
            this->set_ptr(ptr__);
        }

        mdarray(T* ptr__, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            this->init_dimensions({d0, d1});
            this->set_ptr(ptr__);
        }

        mdarray(T* ptr__, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1, 
                mdarray_index_descriptor const& d2)
        {
            this->init_dimensions({d0, d1, d2});
            this->set_ptr(ptr__);
        }

        mdarray(T* ptr__, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1, 
                mdarray_index_descriptor const& d2, mdarray_index_descriptor const& d3)
        {
            this->init_dimensions({d0, d1, d2, d3});
            this->set_ptr(ptr__);
        }

        mdarray(T* ptr__, T* ptr_device__, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1)
        {
            this->init_dimensions({d0, d1});
            this->set_ptr(ptr__);
            #ifdef _GPU_
            this->set_ptr_device(ptr_device__);
            #endif
        }
        mdarray(T* ptr__, T* ptr_device__, mdarray_index_descriptor const& d0, mdarray_index_descriptor const& d1, 
                mdarray_index_descriptor const& d2)
        {
            this->init_dimensions({d0, d1, d2});
            this->set_ptr(ptr__);
            #ifdef _GPU_
            this->set_ptr_device(ptr_device__);
            #endif
        }

        mdarray<T, 2> submatrix(int idx)
        {
            T* ptr = &(*this)(0, 0, idx);
            return mdarray<T, 2>(ptr, this->dims_[0], this->dims_[1]);
        }
};

// Alias for matrix
template <typename T> using matrix = mdarray<T, 2>;

#endif // __MDARRAY_H__


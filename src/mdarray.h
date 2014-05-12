#ifndef __MDARRAY_H__
#define __MDARRAY_H__

#include <memory>
#include <string.h>
#include <vector>
#include "error_handling.h"

class dimension 
{
    private:

        int64_t start_;
        int64_t end_;
        size_t size_;

    public:
  
        dimension() : start_(0), end_(-1), size_(0) 
        {
        }
        
        dimension(size_t size__) : size_(size__)
        {
            start_ = 0;
            end_ = size_ - 1;
        }
    
        dimension(int64_t start_, int64_t end_) : start_(start_), end_(end_) 
        {
            assert(end_ >= start_);
            size_ = end_ - start_ + 1;
        };

        inline int64_t start() 
        {
            return start_;
        }
        
        inline int64_t end() 
        {
            return end_;
        }
        
        inline size_t size() 
        {
            return size_;
        }
        
};

template <typename T, int ND> class mdarray_base
{
    private:

        // forbid copy constructor
        //mdarray_base(const mdarray_base<T, ND>& src);
        
        // forbid assignment operator
        //mdarray_base<T, ND>& operator=(const mdarray_base<T, ND>& src); 

    protected:
    
        std::shared_ptr<T> mdarray_shared_ptr_;
        
        T* mdarray_ptr_;
        
        bool allocated_;
       
        #ifdef _GPU_
        T* mdarray_ptr_device;  
        
        bool allocated_on_device;

        bool pinned_;
        #endif
        
        dimension d[ND];
        
        int64_t offset[ND];

    public:
    
        mdarray_base() 
            : mdarray_ptr_(NULL), 
              allocated_(false)
              #ifdef _GPU_
              ,mdarray_ptr_device(NULL), 
              allocated_on_device(false), 
              pinned_(false)
              #endif
        { 
        }
        
        ~mdarray_base()
        {
            deallocate();
        }
        
        /// Copy constructor
        mdarray_base(const mdarray_base<T, ND>& src)
        {
            this->mdarray_shared_ptr_ = src.mdarray_shared_ptr_; 
            this->mdarray_ptr_ = src.mdarray_ptr_;
            for (int i = 0; i < ND; i++)
            {
                this->d[i] = src.d[i];
                this->offset[i] = src.offset[i];
            }
        }

        /// Assigment operator
        inline mdarray_base<T, ND>& operator=(const mdarray_base<T, ND>& src)
        {
            mdarray_shared_ptr_ = src.mdarray_shared_ptr_;
            mdarray_ptr_ = src.mdarray_ptr_;
            for (int i = 0; i < ND; i++)
            {
                d[i] = src.d[i];
                offset[i] = src.offset[i];
            }
            return *this;
        }

        void init_dimensions(const std::vector<dimension>& vd) 
        {
            assert(vd.size() == ND);
            
            for (int i = 0; i < ND; i++) d[i] = vd[i];
            
            offset[0] = -d[0].start();
            size_t n = 1;
            for (int i = 1; i < ND; i++) 
            {
                n *= d[i - 1].size();
                offset[i] = n;
                offset[0] -= offset[i] * d[i].start();
            }
        }
 
        inline size_t size()
        {
            size_t size_ = 1;

            for (int i = 0; i < ND; i++) size_ *= d[i].size();

            return size_;
        }

        inline size_t size(int i)
        {
           assert(i < ND);
           return d[i].size();
        }
    
        inline uint32_t ld()
        {
            assert(d[0].size() < size_t(1 << 31));

            return (int32_t)d[0].size();
        }

        void allocate()
        {
            deallocate();
            
            size_t sz = size();

            mdarray_shared_ptr_ = std::shared_ptr<T>(new T[sz], [](T* ptr){delete[] ptr;});
            mdarray_ptr_ = mdarray_shared_ptr_.get();



            //== if (type_wrapper<T>::is_primitive())
            //== {
            //==     size_t num_bytes = size() * sizeof(T);

            //==     if (num_bytes)
            //==     {
            //==         mdarray_ptr = (T*)malloc(num_bytes);
            //==         if (mdarray_ptr == NULL)
            //==         {
            //==             std::stringstream s;
            //==             s << "Error allocating " << ND << "-dimensional array of size " << num_bytes << " bytes";
            //==             error_local(__FILE__, __LINE__, s);
            //==         }
            //==         allocated_ = true;
            //==     }
            //== }
            //== else
            //== {
            //==     size_t sz = size();
            //==      
            //==     if (sz && (!mdarray_ptr)) 
            //==     {
            //==         try
            //==         {
            //==             mdarray_ptr = new T[sz];
            //==         }
            //==         catch(...)
            //==         {
            //==             std::stringstream s;
            //==             s << "Error allocating " << ND << "-dimensional array of size " << sz * sizeof(T) << " bytes";
            //==             error_local(__FILE__, __LINE__, s);
            //==         }
            //==         allocated_ = true;
            //==     }
            //== }
        }

        void deallocate()
        {
            mdarray_shared_ptr_.reset();
            mdarray_ptr_ = NULL;

            //== if (allocated_)
            //== {
            //==     #ifdef _GPU_
            //==     unpin_memory();
            //==     #endif
            //==     if (type_wrapper<T>::is_primitive())
            //==     {
            //==         free(mdarray_ptr);
            //==     }
            //==     else
            //==     {
            //==         delete[] mdarray_ptr;
            //==     }
            //==     mdarray_ptr = NULL;
            //==     allocated_ = false;
            //== }
            //== #ifdef _GPU_
            //== deallocate_on_device();
            //== #endif
        }
        
        void zero()
        {
            if (size() != 0)
            {
                assert(mdarray_ptr_);
                memset(mdarray_ptr_, 0, size() * sizeof(T));
            }
        }
        
        void set_ptr(T* ptr)
        {
            mdarray_ptr_ = ptr;
        }
        
        T* ptr()
        {
            return mdarray_ptr_;
        }
        
        bool allocated()
        {
            return allocated_;
        }

        /// Compute hash of the array
        /** Example: printf("hash(h) : %16llX\n", h.hash()); */
        uint64_t hash()
        {
            uint64_t h = 5381;

            for(size_t i = 0; i < size() * sizeof(T); i++) h = ((h << 5) + h) + ((unsigned char*)mdarray_ptr_)[i];

            return h;
        }
        
        /// Copy the content of the array to dest
        void operator>>(mdarray_base<T, ND>& dest)
        {
            for (int i = 0; i < ND; i++) 
            {
                if (dest.d[i].start() != d[i].start() || dest.d[i].end() != d[i].end())
                    error_local(__FILE__, __LINE__, "array dimensions don't match");
            }
            memcpy(dest.ptr(), ptr(), size() * sizeof(T));
        }

        #ifdef _GPU_
        void allocate_on_device()
        {
            deallocate_on_device();
            
            size_t sz = size();
            if (sz == 0) 
            {
                std::stringstream s;
                s <<  "can't allocate a zero sized array" << std::endl
                  <<  "  array dimensions : ";
                for (int i = 0; i < ND; i++) s << d[i].size() << " ";

                error_local(__FILE__, __LINE__, s);
            }
             
            cuda_malloc((void**)(&mdarray_ptr_device), sz * sizeof(T));
            allocated_on_device = true;
        }

        void deallocate_on_device()
        {
            if (allocated_on_device)
            {
                cuda_free(mdarray_ptr_device);
                mdarray_ptr_device = NULL;
                allocated_on_device = false;
            }
        }

        void allocate_page_locked()
        {
            cuda_malloc_host((void**)(&mdarray_ptr_), size() * sizeof(T));
        }

        void deallocate_page_locked()
        {
            cuda_free_host((void**)(&mdarray_ptr_));
        }

        void copy_to_device() 
        {
            assert(mdarray_ptr_ != NULL);
            assert(mdarray_ptr_device != NULL);

            cuda_copy_to_device(mdarray_ptr_device, mdarray_ptr_, size() * sizeof(T));
        }

        void copy_to_host() 
        {
            assert(mdarray_ptr_ != NULL);
            assert(mdarray_ptr_device != NULL);
            
            cuda_copy_to_host(mdarray_ptr_, mdarray_ptr_device, size() * sizeof(T));
        }

        void async_copy_to_device(int stream_id = -1) 
        {
            cuda_async_copy_to_device(mdarray_ptr_device, mdarray_ptr_, size() * sizeof(T), stream_id);
        }
        
        void async_copy_to_host(int stream_id = -1) 
        {
            cuda_async_copy_to_host(mdarray_ptr_, mdarray_ptr_device, size() * sizeof(T), stream_id);
        }

        void zero_on_device()
        {
            cuda_memset(mdarray_ptr_device, 0, size() * sizeof(T));
        }

        void pin_memory()
        {
            if (pinned_) error_local(__FILE__, __LINE__, "Memory is already pinned");
            cuda_host_register(mdarray_ptr_, size() * sizeof(T));
            pinned_ = true;
        }
        
        void unpin_memory()
        {
            if (pinned_)
            {
                cuda_host_unregister(mdarray_ptr_);
                pinned_ = false;
            }
        }
        #endif
 
};

template <typename T, int ND> class mdarray : public mdarray_base<T, ND>
{
};

// 1d specialization
template <typename T> class mdarray<T, 1> : public mdarray_base<T, 1> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const dimension& d0)
        {
            set_dimensions(d0);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const dimension& d0)
        {
            set_dimensions(d0);
            this->allocate();
        }

        void set_dimensions(const dimension& d0)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int64_t i0) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(this->mdarray_ptr_);
            
            int64_t i = this->offset[0] + i0;
            return this->mdarray_ptr_[i];
        }

        #ifdef _GPU_
        inline T* ptr_device()
        {
            return this->mdarray_ptr_device;
        }

        inline T* ptr_device(const int64_t i0)
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(this->mdarray_ptr_device);
            
            int64_t i = this->offset[0] + i0;
            return &this->mdarray_ptr_device[i];
        }
        #endif
};

// 2d specialization
template <typename T> class mdarray<T, 2> : public mdarray_base<T, 2> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const dimension& d0, const dimension& d1)
        {
            set_dimensions(d0, d1);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const dimension& d0, const dimension& d1)
        {
            set_dimensions(d0, d1);
            this->allocate();
        }
        
        void set_dimensions(const dimension& d0, const dimension& d1)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            vd.push_back(d1);
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int64_t i0, const int64_t i1) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(this->mdarray_ptr_);
            
            int64_t i = this->offset[0] + i0 + i1 * this->offset[1];
            return this->mdarray_ptr_[i];
        }
    
        #ifdef _GPU_
        inline T* ptr_device()
        {
            return this->mdarray_ptr_device;
        }

        inline T* ptr_device(const int64_t i0, const int64_t i1) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(this->mdarray_ptr_device);
            
            int64_t i = this->offset[0] + i0 + i1 * this->offset[1];
            return &this->mdarray_ptr_device[i];
        }
        #endif
};

// 3d specialization
template <typename T> class mdarray<T, 3> : public mdarray_base<T, 3> 
{
    private:

        mdarray<T, 2> submatrix_;

    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const dimension& d0, const dimension& d1, const dimension& d2)
        {
            set_dimensions(d0, d1, d2);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const dimension& d0, const dimension& d1, const dimension& d2)
        {
            set_dimensions(d0, d1, d2);
            this->allocate();
        }
        
        void set_dimensions(const dimension& d0, const dimension& d1, const dimension& d2)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            vd.push_back(d1);
            vd.push_back(d2);
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int64_t i0, const int64_t i1, const int64_t i2) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(this->mdarray_ptr_);
            
            int64_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2];
            return this->mdarray_ptr_[i];
        }

        #ifdef _GPU_
        inline T* ptr_device()
        {
            return this->mdarray_ptr_device;
        }

        inline T* ptr_device(const int64_t i0, const int64_t i1, const int64_t i2) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(this->mdarray_ptr_device);
            
            int64_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2];
            return &this->mdarray_ptr_device[i];
        }
        #endif

        mdarray<T, 2>& submatrix(int idx)
        {
            submatrix_.set_dimensions(this->d[0], this->d[1]);
            submatrix_.set_ptr(&(*this)(0, 0, idx));
            return submatrix_;
        }
};

// 4d specialization
template <typename T> class mdarray<T, 4> : public mdarray_base<T, 4> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3)
        {
            set_dimensions(d0, d1, d2, d3);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3)
        {
            set_dimensions(d0, d1, d2, d3);
            this->allocate();
        }
         
        void set_dimensions(const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            vd.push_back(d1);
            vd.push_back(d2);
            vd.push_back(d3);            
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(i3 >= this->d[3].start() && i3 <= this->d[3].end());
            assert(this->mdarray_ptr_);
            
            int64_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3];
            return this->mdarray_ptr_[i];
        }

        #ifdef _GPU_
        inline T* ptr_device()
        {
            return this->mdarray_ptr_device;
        }

        inline T* ptr_device(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(i3 >= this->d[3].start() && i3 <= this->d[3].end());
            assert(this->mdarray_ptr_device);
            
            int64_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3];
            return &this->mdarray_ptr_device[i];
        }
        #endif
};

#endif // __MDARRAY_H__


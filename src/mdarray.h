#ifndef _MDARRAY_H_
#define _MDARRAY_H_

class dimension 
{
    public:
  
        dimension() : start_(0), end_(-1), size_(0) 
        {
        }
        
        dimension(unsigned int size__) : size_(size__)
        {
            start_ = 0;
            end_ = size_ - 1;
        }
    
        dimension(int start_, int end_) : start_(start_), end_(end_) 
        {
            assert(end_ >= start_);
            size_ = end_ - start_ + 1;
        };

        inline int start() 
        {
            return start_;
        }
        
        inline int end() 
        {
            return end_;
        }
        
        inline unsigned int size() 
        {
            return size_;
        }
        
    private:

        int start_;
        int end_;
        unsigned int size_;
};

template <typename T, int ND> class mdarray_base
{
    public:
    
        mdarray_base() : mdarray_ptr(NULL), allocated_(false)
                         #ifdef _GPU_
                         ,mdarray_ptr_device(NULL), allocated_on_device(false), pinned_(false)
                         #endif
        { 
        }
        
        ~mdarray_base()
        {
            deallocate();
        }
        
        void init_dimensions(const std::vector<dimension>& vd) 
        {
            assert(vd.size() == ND);
            
            for (int i = 0; i < ND; i++) d[i] = vd[i];
            
            offset[0] = -d[0].start();
            size_t n = 1;
            for (int i = 1; i < ND; i++) 
            {
                n *= d[i-1].size();
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

        inline int size(int i)
        {
           assert(i < ND);
           return d[i].size();
        }

        inline int ld()
        {
            return d[0].size();
        }

        inline std::vector<int> dimensions()
        {
            std::vector<int> vd(ND);
            for (int i = 0; i < ND; i++) vd[i] = d[i].size();
            return vd;
        }

        void allocate()
        {
            deallocate();
            
            size_t sz = size();
             
            if (sz && (!mdarray_ptr)) 
            {
                try
                {
                    mdarray_ptr = new T[sz];
                }
                catch(...)
                {
                    std::stringstream s;
                    s << "Error allocating " << ND << "-dimensional array of size " << sz * sizeof(T);
                    error(__FILE__, __LINE__, s, fatal_err);
                }
                allocated_ = true;
                Platform::adjust_heap_allocated(sz * sizeof(T));
            }
        }

        void deallocate()
        {
            if (allocated_)
            {
                #ifdef _GPU_
                unpin_memory();
                #endif
                delete[] mdarray_ptr;
                mdarray_ptr = NULL;
                allocated_ = false;
                Platform::adjust_heap_allocated(-size() * sizeof(T));
            }
            #ifdef _GPU_
            deallocate_on_device();
            #endif
        }
        
        void zero()
        {
            if (size() != 0)
            {
                assert(mdarray_ptr);
                memset(mdarray_ptr, 0, size() * sizeof(T));
            }
        }
        
        void set_ptr(T* ptr)
        {
            mdarray_ptr = ptr;
        }
        
        T* get_ptr()
        {
            return mdarray_ptr;
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

            for(size_t i = 0; i < size() * sizeof(T); i++) h = ((h << 5) + h) + ((unsigned char*)mdarray_ptr)[i];

            return h;
        }

        #ifdef _GPU_
        void allocate_on_device()
        {
            deallocate_on_device();
            
            size_t sz = size();
            if (sz == 0) throw std::runtime_error("can't allocate a zero sized array");
             
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

        void copy_to_device() 
        {
            cuda_copy_to_device(mdarray_ptr_device, mdarray_ptr, size() * sizeof(T));
        }
        
        void copy_to_host() 
        {
            cuda_copy_to_host(mdarray_ptr, mdarray_ptr_device, size() * sizeof(T));
        }
        
        void async_copy_to_device(int stream_id = -1) 
        {
            cuda_async_copy_to_device(mdarray_ptr_device, mdarray_ptr, size() * sizeof(T), stream_id);
        }
        
        void async_copy_to_host(int stream_id = -1) 
        {
            cuda_async_copy_to_host(mdarray_ptr, mdarray_ptr_device, size() * sizeof(T), stream_id);
        }

        inline T* get_ptr_device()
        {
            return mdarray_ptr_device;
        }

        void zero_on_device()
        {
            cuda_memset(mdarray_ptr_device, 0, size() * sizeof(T));
        }

        void pin_memory()
        {
            if (pinned_) error(__FILE__, __LINE__, "Memory is already pinned");
            cuda_host_register(mdarray_ptr, size() * sizeof(T));
            pinned_ = true;
        }
        
        void unpin_memory()
        {
            if (pinned_)
            {
                cuda_host_unregister(mdarray_ptr);
                pinned_ = false;
            }
        }
        #endif
 
    protected:
    
        T* mdarray_ptr;
        
        bool allocated_;
       
        #ifdef _GPU_
        T* mdarray_ptr_device;  
        
        bool allocated_on_device;

        bool pinned_;
        #endif
        
        dimension d[ND];
        
        size_t offset[ND];

    private:

        // forbid copy constructor
        mdarray_base(const mdarray_base<T, ND>& src);
        
        // forbid assignment operator
        mdarray_base<T, ND>& operator=(const mdarray_base<T, ND>& src); 
        
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
    
        inline T& operator()(const int i0) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            size_t i = this->offset[0] + i0;
            
            assert(this->mdarray_ptr);
            return this->mdarray_ptr[i];
        }
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
    
        inline T& operator()(const int i0, const int i1) 
        {
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            size_t i = this->offset[0] + i0 + i1 * this->offset[1];
            
            assert(this->mdarray_ptr);
            return this->mdarray_ptr[i];
        }
};

// 3d specialization
template <typename T> class mdarray<T, 3> : public mdarray_base<T, 3> 
{
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
    
        inline T& operator()(const int i0, const int i1, const int i2) 
        {
            assert(this->mdarray_ptr);
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            
            size_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2];
            return this->mdarray_ptr[i];
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
    
        inline T& operator()(const int i0, const int i1, const int i2, const int i3) 
        {
            assert(this->mdarray_ptr);
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(i3 >= this->d[3].start() && i3 <= this->d[3].end());
            
            size_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3];
            return this->mdarray_ptr[i];
        }
};

#if 0
// 5d specialization
template <typename T> class mdarray<T, 5> : public mdarray_base<T, 5> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4)
        {
            set_dimensions(d0, d1, d2, d3, d4);
            this->set_ptr(data_ptr);
        }
        
        mdarray(const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4)
        {
            set_dimensions(d0, d1, d2, d3, d4);
            this->allocate();
        }
        
        void set_dimensions(const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            vd.push_back(d1);
            vd.push_back(d2);
            vd.push_back(d3);    
            vd.push_back(d4);    
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int i0, const int i1, const int i2, const int i3, const int i4) 
        {
            assert(this->mdarray_ptr);
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(i3 >= this->d[3].start() && i3 <= this->d[3].end());
            assert(i4 >= this->d[4].start() && i4 <= this->d[4].end());
            
            size_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3] + i4 * this->offset[4];
            return this->mdarray_ptr[i];
        }
};
#endif

#if 0
// 6d specialization
template <typename T> class mdarray<T, 6> : public mdarray_base<T, 6> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T* data_ptr, const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4, const dimension& d5)
        {
            set_dimensions(d0, d1, d2, d3, d4, d5);
            this->set_ptr(data_ptr);
        }
        
        void set_dimensions(const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4, const dimension& d5)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            vd.push_back(d1);
            vd.push_back(d2);
            vd.push_back(d3);
            vd.push_back(d4);
            vd.push_back(d5);
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5) 
        {
            assert(this->mdarray_ptr);
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            assert(i2 >= this->d[2].start() && i2 <= this->d[2].end());
            assert(i3 >= this->d[3].start() && i3 <= this->d[3].end());
            assert(i4 >= this->d[4].start() && i4 <= this->d[4].end());
            assert(i5 >= this->d[5].start() && i5 <= this->d[5].end());
            
            size_t i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3] + i4 * this->offset[4] + i5 * this->offset[5];
            return this->mdarray_ptr[i];
        }
};
#endif

#endif // _MDARRAY_H_


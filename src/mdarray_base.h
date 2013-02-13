#ifndef _MDARRAY_BASE_H_
#define _MDARRAY_BASE_H_

class dimension 
{
    public:
  
        dimension() : start_(0), end_(-1), size_(0) 
        {
        }
        
        dimension(unsigned int size_) : size_(size_)
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
    
        mdarray_base() : mdarray_ptr(NULL), 
                         allocated_(false), 
                         mdarray_ptr_device(NULL), 
                         allocated_on_device(false) 
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

            for (int i = 0; i < ND; i++) 
                size_ *= d[i].size();

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
                delete[] mdarray_ptr;
                mdarray_ptr = NULL;
                allocated_ = false;
                Platform::adjust_heap_allocated(-size() * sizeof(T));
            }
        }
        
        void zero()
        {
            assert(mdarray_ptr);
            memset(mdarray_ptr, 0, size() * sizeof(T));
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

            for(size_t i = 0; i < size() * sizeof(T); i++)
                h = ((h << 5) + h) + ((unsigned char*)mdarray_ptr)[i];

            return h;
        }

        
        /*void copy_members(const mdarray_base<impl,T,ND>& src) 
        {
            for (int i = 0; i < ND; i++) 
            { 
                offset[i] = src.offset[i];
                d[i] = src.d[i];
            }
        }*/
 
    protected:
    
        T* mdarray_ptr;
        
        bool allocated_;
        
        T* mdarray_ptr_device;  
        
        bool allocated_on_device;
        
        dimension d[ND];
        
        size_t offset[ND];

    private:

        // forbid copy constructor
        mdarray_base(const mdarray_base<T,ND>& src);
        
        // forbid assignment operator
        mdarray_base<T,ND>& operator=(const mdarray_base<T,ND>& src); 
        
};

#endif // _MDARRAY_BASE_H_



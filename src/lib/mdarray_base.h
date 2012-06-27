#ifndef _MDARRAY_BASE_H_
#define _MDARRAY_BASE_H_

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <assert.h>

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
    
        mdarray_base() : mdarray_ptr(0), 
                         allocated(false), 
                         mdarray_ptr_device(0), 
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
            unsigned int n = 1;
            for (int i = 1; i < ND; i++) 
            {
                n *= d[i-1].size();
                offset[i] = n;
                offset[0] -= offset[i] * d[i].start();
            }
        }
 
        inline int size()
        {
            int n = 1;
            for (int i = 0; i < ND; i++) n *= d[i].size();
            return n;
        }

        inline int size(int i)
        {
           assert(i < ND);
           return d[i].size();
        }

        void allocate()
        {
            deallocate();
            
            int sz = size();
            if (sz == 0) throw std::runtime_error("can't allocate a zero size array");
             
            mdarray_ptr = new T[sz];
            allocated = true;
        }

        void deallocate()
        {
            if (allocated)
            {
                delete[] mdarray_ptr;
                allocated = false;
            }
        }
        
        void zero()
        {
            assert(mdarray_ptr);
            memset(mdarray_ptr, 0, size() * sizeof(T));
        }
        
        void set_ptr(T *ptr)
        {
            mdarray_ptr = ptr;
        }
        
        T* get_ptr()
        {
            return mdarray_ptr;
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
    
        T *mdarray_ptr;
        bool allocated;
        T *mdarray_ptr_device;  
        bool allocated_on_device;
        dimension d[ND];
        int offset[ND];

    private:

        // forbid copy constructor
        mdarray_base(const mdarray_base<T,ND>& src);
        
        // forbid assign operator
        mdarray_base<T,ND>& operator=(const mdarray_base<T,ND>& src); 
        
};

#endif // _MDARRAY_BASE_H_



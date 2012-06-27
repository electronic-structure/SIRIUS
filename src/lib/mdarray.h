#ifndef _MDARRAY_H_
#define _MDARRAY_H_

#include "mdarray_base.h"
#ifdef _GPU_
#include "mdarray_base_gpu.h"
#else
#include "mdarray_base_cpu.h"
#endif

template <typename T, int ND> class mdarray : public mdarray_base_impl<T,ND> 
{
};

// 1d specialization
template <typename T> class mdarray<T,1> : public mdarray_base_impl<T,1> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T *data_ptr, const dimension& d0)
        {
            set_dimensions(d0);
            this->set_ptr(data_ptr);
        }

        void set_dimensions(const dimension& d0)
        {
            std::vector<dimension> vd;
            vd.push_back(d0);
            this->init_dimensions(vd);
        }
    
        inline T& operator()(const int i0) 
        {
            assert(this->mdarray_ptr);
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            
            int i = this->offset[0] + i0;
            return this->mdarray_ptr[i];
        }
};

// 2d specialization
template <typename T> class mdarray<T,2> : public mdarray_base_impl<T,2> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T *data_ptr, const dimension& d0, const dimension& d1)
        {
            set_dimensions(d0, d1);
            this->set_ptr(data_ptr);
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
            assert(this->mdarray_ptr);
            assert(i0 >= this->d[0].start() && i0 <= this->d[0].end());
            assert(i1 >= this->d[1].start() && i1 <= this->d[1].end());
            
            int i = this->offset[0] + i0 + i1 * this->offset[1];
            return this->mdarray_ptr[i];
        }
};

// 3d specialization
template <typename T> class mdarray<T,3> : public mdarray_base_impl<T,3> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T *data_ptr, const dimension& d0, const dimension& d1, const dimension& d2)
        {
            set_dimensions(d0, d1, d2);
            this->set_ptr(data_ptr);
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
            
            int i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2];
            return this->mdarray_ptr[i];
        }
};

// 4d specialization
template <typename T> class mdarray<T,4> : public mdarray_base_impl<T,4> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T *data_ptr, const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3)
        {
            set_dimensions(d0, d1, d2, d3);
            this->set_ptr(data_ptr);
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
            
            int i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3];
            return this->mdarray_ptr[i];
        }
};

// 5d specialization
template <typename T> class mdarray<T,5> : public mdarray_base_impl<T,5> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T *data_ptr, const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4)
        {
            set_dimensions(d0, d1, d2, d3, d4);
            this->set_ptr(data_ptr);
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
            
            int i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3] + i4 * this->offset[4];
            return this->mdarray_ptr[i];
        }
};

// 6d specialization
template <typename T> class mdarray<T,6> : public mdarray_base_impl<T,6> 
{
    public:
  
        mdarray() 
        {
        }

        mdarray(T *data_ptr, const dimension& d0, const dimension& d1, const dimension& d2, const dimension& d3, const dimension& d4, const dimension& d5)
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
            
            int i = this->offset[0] + i0 + i1 * this->offset[1] + i2 * this->offset[2] + i3 * this->offset[3] + i4 * this->offset[4] + i5 * this->offset[5];
            return this->mdarray_ptr[i];
        }
};

#endif // _MDARRAY_H_


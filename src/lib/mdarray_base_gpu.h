#ifndef _MDARRAY_BASE_GPU_H_
#define _MDARRAY_BASE_GPU_H_

#include <iostream>
#include "gpu_interface.h"

template <typename T, int ND> class mdarray_base_impl : public mdarray_base<T,ND> 
{
    public:
        ~mdarray_base_impl()
        {
            deallocate_on_device();
        }

        void allocate_on_device()
        {
            deallocate_on_device();
            
            int sz = this->size();
            if (sz == 0) throw std::runtime_error("can't allocate a zero sized array");
             
            gpu_malloc((void**)(&this->mdarray_ptr_device), sz * sizeof(T));
            this->allocated_on_device = true;
        }

        void deallocate_on_device()
        {
            if (this->allocated_on_device)
            {
                gpu_free(this->mdarray_ptr_device);
                this->allocated_on_device = false;
            }
        }
        
        void copy_to_device() 
        {
            gpu_copy_to_device(this->mdarray_ptr_device, this->mdarray_ptr, this->size() * sizeof(T));
        }
        
        void copy_to_host() 
        {
            gpu_copy_to_host(this->mdarray_ptr, this->mdarray_ptr_device, this->size() * sizeof(T));
        }

        inline T *get_ptr_device()
        {
            return this->mdarray_ptr_device;
        }

        void zero_on_device()
        {
            gpu_mem_zero(this->mdarray_ptr_device, this->size() * sizeof(T));
        }
};

#endif // _MDARRAY_BASE_GPU_H_



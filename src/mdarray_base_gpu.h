#ifndef _MDARRAY_BASE_GPU_H_
#define _MDARRAY_BASE_GPU_H_

template <typename T, int ND> class mdarray_base_impl : public mdarray_base<T, ND> 
{
    public:
        ~mdarray_base_impl()
        {
            deallocate_on_device();
        }

        void allocate_on_device()
        {
            deallocate_on_device();
            
            size_t sz = this->size();
            if (sz == 0) throw std::runtime_error("can't allocate a zero sized array");
             
            cuda_malloc((void**)(&this->mdarray_ptr_device), sz * sizeof(T));
            this->allocated_on_device = true;
        }

        void deallocate_on_device()
        {
            if (this->allocated_on_device)
            {
                cuda_free(this->mdarray_ptr_device);
                this->allocated_on_device = false;
            }
        }

        void allocate_page_locked()
        {
            deallocate_page_locked();

            size_t sz = this->size();
             
            if (sz && (!this->mdarray_ptr)) 
            {
                cuda_malloc_host((void**)&this->mdarray_ptr, sz * sizeof(T));
                this->allocated_ = true;
                Platform::adjust_heap_allocated(sz * sizeof(T));
            }
        }

        void deallocate_page_locked()
        {
            if (this->allocated_)
            {
                cuda_free_host((void**)&this->mdarray_ptr);
                this->mdarray_ptr = NULL;
                this->allocated_ = false;
                Platform::adjust_heap_allocated(-this->size() * sizeof(T));
            }
        }
        
        void copy_to_device() 
        {
            cuda_copy_to_device(this->mdarray_ptr_device, this->mdarray_ptr, this->size() * sizeof(T));
        }
        
        void copy_to_host() 
        {
            cuda_copy_to_host(this->mdarray_ptr, this->mdarray_ptr_device, this->size() * sizeof(T));
        }

        inline T *get_ptr_device()
        {
            return this->mdarray_ptr_device;
        }

        void zero_on_device()
        {
            cuda_memset(this->mdarray_ptr_device, 0, this->size() * sizeof(T));
        }
};

#endif // _MDARRAY_BASE_GPU_H_



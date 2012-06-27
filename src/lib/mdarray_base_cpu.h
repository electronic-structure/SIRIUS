#ifndef _MDARRAY_BASE_CPU_H_
#define _MDARRAY_BASE_CPU_H_

template <typename T, int ND> class mdarray_base_impl : public mdarray_base<T,ND> 
{
    public:
    
        void allocate_on_device();
        
        void deallocate_on_device();
        
        void copy_to_device();
        
        void copy_to_host();
        
        void zero_on_device();
        
        T *get_ptr_device();
};

#endif // _MDARRAY_BASE_CPU_H_



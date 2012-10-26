#ifndef _GPU_INTERFACE_H_
#define _GPU_INTERFACE_H_

extern "C" void gpu_malloc(void **ptr, int size);

extern "C" void gpu_free(void *ptr);

extern "C" void gpu_copy_to_device(void *target, void *source, int size);

extern "C" void gpu_copy_to_host(void *target, void *source, int size);

extern "C" void gpu_mem_zero(void *ptr, int size);

#endif // _GPU_INTERFACE_H_


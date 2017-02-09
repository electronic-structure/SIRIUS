#ifndef __CUDA_INTERFACE_H__
#define __CUDA_INTERFACE_H__

extern "C" {

void cuda_initialize();

void cuda_device_info();

void* cuda_malloc(size_t size);

void cuda_free(void* ptr);

void* cuda_malloc_host(size_t size);

void cuda_free_host(void* ptr);

void cuda_memset(void *ptr, int value, size_t size);

void cuda_host_register(void* ptr, size_t size);

void cuda_host_unregister(void* ptr);

void cuda_device_synchronize();

void cuda_create_streams(int num_streams);

void cuda_destroy_streams();

int get_num_cuda_streams();

void cuda_stream_synchronize(int stream_id);

void cuda_copy_to_device(void* target, void const* source, size_t size);
void cuda_async_copy_to_device(void* target, void const* source, size_t size, int stream_id);
void cuda_copy2d_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__,
                           size_t ncol__, int elem_size__);
void cuda_async_copy2d_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                                 size_t nrow__, size_t ncol__, int elem_size__, int stream_id__);

void cuda_copy_to_host(void* target, void const* source, size_t size);
void cuda_async_copy_to_host(void* target, void const* source, size_t size, int stream_id);
void cuda_copy2d_to_host(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__, 
                         size_t ncol__, int elem_size__);
void cuda_async_copy2d_to_host(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                               size_t nrow__, size_t ncol__, int elem_size__, int stream_id__);

void cuda_copy_device_to_device(void* target, void const* source, size_t size);
void cuda_copy2d_device_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__,
                                  size_t nrow__, size_t ncol__, int elem_size__);
void cuda_async_copy2d_device_to_device(void* dst__, size_t ld1__, const void* src__, size_t ld2__, size_t nrow__,
                                        size_t ncol__, int elem_size__, int stream_id__);

size_t cuda_get_free_mem();

void cuda_device_reset();

void cuda_check_last_error();

bool cuda_check_device_ptr(void const* ptr__);

void cuda_memset2d(void* ptr__, int ld__, int nrow__, int ncol__, int elem_size__, int value__);

#ifdef __GPU_NVTX
void cuda_begin_range_marker(const char* label__);

void cuda_end_range_marker();
#endif

}

#endif

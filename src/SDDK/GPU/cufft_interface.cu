#include <unistd.h>
#include <cufft.h>
#include <cuda.h>
#include "cuda_common.h"

extern "C" void cuda_memset(void* ptr, int value, size_t size);

//=================
// CUFFT functions
//=================

void cufft_error_message(cufftResult result)
{
    switch (result) {
        case CUFFT_INVALID_PLAN: {
            printf("CUFFT_INVALID_PLAN\n");
            break;
        }
        case CUFFT_ALLOC_FAILED: {
            printf("CUFFT_ALLOC_FAILED\n");
            break;
        }
        case CUFFT_INVALID_VALUE: {
            printf("CUFFT_INVALID_VALUE\n");
            break;
        }
        case CUFFT_INTERNAL_ERROR: {
            printf("CUFFT_INTERNAL_ERROR\n");
            break;
        }
        case CUFFT_SETUP_FAILED: {
            printf("CUFFT_SETUP_FAILED\n");
            break;
        }
        case CUFFT_INVALID_SIZE: {
            printf("CUFFT_INVALID_SIZE\n");
            break;
        }
        default: {
            printf("unknown error code %i\n", result);
            break;
        }
    }
}

#define CALL_CUFFT(func__, args__)                                                  \
{                                                                                   \
    cufftResult result;                                                             \
    if ((result = func__ args__) != CUFFT_SUCCESS) {                                \
        char nm[1024];                                                              \
        gethostname(nm, 1024);                                                      \
        printf("hostname: %s\n", nm);                                               \
        printf("Error in %s at line %i of file %s: ", #func__, __LINE__, __FILE__); \
        cufft_error_message(result);                                                \
        exit(-100);                                                                 \
    }                                                                               \
}

extern "C" void cufft_create_plan_handle(cufftHandle* plan)
{
    CALL_CUFFT(cufftCreate, (plan));
}

extern "C" void cufft_destroy_plan_handle(cufftHandle plan)
{
    CALL_CUFFT(cufftDestroy, (plan));
}

// Size of work buffer in bytes
extern "C" size_t cufft_get_work_size(int ndim, int* dims, int nfft)
{
    int fft_size = 1;
    for (int i = 0; i < ndim; i++) {
        fft_size *= dims[i];
    }
    size_t work_size;

    CALL_CUFFT(cufftEstimateMany, (ndim, dims, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, nfft, &work_size));
    
    return work_size;
}

extern "C" size_t cufft_create_batch_plan(cufftHandle plan, int rank, int* dims, int* embed, int stride, int dist, int nfft, int auto_alloc)
{
    int fft_size = 1;
    for (int i = 0; i < rank; i++) fft_size *= dims[i];
    
    if (auto_alloc)
    {
        CALL_CUFFT(cufftSetAutoAllocation, (plan, true));
    }
    else
    {
        CALL_CUFFT(cufftSetAutoAllocation, (plan, false));
    }
    size_t work_size;

    /* 1D
       input[ b * idist + x * istride]
       output[ b * odist + x * ostride]
       
       2D
       input[b * idist + (x * inembed[1] + y) * istride]
       output[b * odist + (x * onembed[1] + y) * ostride]
       
       3D
       input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
       output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]

       - See more at: http://docs.nvidia.com/cuda/cufft/index.html#advanced-data-layout
     */
    CALL_CUFFT(cufftMakePlanMany, (plan, rank, dims, embed, stride, dist, embed, stride, dist, CUFFT_Z2Z, nfft, &work_size));

    return work_size;
}

extern "C" void cufft_set_work_area(cufftHandle plan, void* work_area)
{
    CALL_CUFFT(cufftSetWorkArea, (plan, work_area));
}

extern "C" void cufft_set_stream(cufftHandle plan__, int stream_id__)
{
    CALL_CUFFT(cufftSetStream, (plan__, cuda_stream_by_id(stream_id__)));
}

__global__ void cufft_batch_load_gpu_kernel
(
    int fft_size, 
    int num_pw_components, 
    int const* map, 
    cuDoubleComplex const* data, 
    cuDoubleComplex* fft_buffer
)
{
    int i = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pw_components) {
        fft_buffer[array2D_offset(map[idx], i, fft_size)] = data[array2D_offset(idx, i, num_pw_components)];
    }
}

extern "C" void cufft_batch_load_gpu(int fft_size,
                                     int num_pw_components, 
                                     int num_fft,
                                     int const* map, 
                                     cuDoubleComplex const* data, 
                                     cuDoubleComplex* fft_buffer)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);
    
    cuda_memset(fft_buffer, 0, fft_size * num_fft * sizeof(cuDoubleComplex));

    cufft_batch_load_gpu_kernel <<<grid_b, grid_t>>>
    (
        fft_size,
        num_pw_components,
        map,
        data, 
        fft_buffer
    );
}

__global__ void cufft_load_x0y0_col_gpu_kernel(int z_col_size,
                                               int const* map,
                                               cuDoubleComplex const* data,
                                               cuDoubleComplex* fft_buffer)

{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < z_col_size) {
        fft_buffer[map[idx]] = cuConj(data[idx]);
    }
}

extern "C" void cufft_load_x0y0_col_gpu(int z_col_size,
                                        int const* map,
                                        cuDoubleComplex const* data,
                                        cuDoubleComplex* fft_buffer)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(z_col_size, grid_t.x));

    cufft_load_x0y0_col_gpu_kernel <<<grid_b, grid_t>>>
    (
        z_col_size,
        map,
        data,
        fft_buffer
    );
}

__global__ void cufft_batch_unload_gpu_kernel
(
    int fft_size, 
    int num_pw_components, 
    int const* map, 
    cuDoubleComplex const* fft_buffer,
    cuDoubleComplex* data,
    double alpha,
    double beta
)
{
    int i = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pw_components) {
        data[array2D_offset(idx, i, num_pw_components)] = cuCadd(
            cuCmul(make_cuDoubleComplex(alpha, 0), data[array2D_offset(idx, i, num_pw_components)]),
            cuCmul(make_cuDoubleComplex(beta, 0),  fft_buffer[array2D_offset(map[idx], i, fft_size)]));
    }
}

/// Unload data from FFT buffer.
/** The following operation is executed:
 *  data[ig] = alpha * data[ig] + beta * fft_buffer[map[ig]] */
extern "C" void cufft_batch_unload_gpu(int fft_size,
                                       int num_pw_components,
                                       int num_fft,
                                       int const* map, 
                                       cuDoubleComplex const* fft_buffer, 
                                       cuDoubleComplex* data,
                                       double alpha,
                                       double beta)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);

    cufft_batch_unload_gpu_kernel <<<grid_b, grid_t>>>
    (
        fft_size, 
        num_pw_components, 
        map, 
        fft_buffer,
        data,
        alpha,
        beta
    );
}

extern "C" void cufft_forward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer)
{
    //CUDA_timer t("cufft_forward_transform");
    CALL_CUFFT(cufftExecZ2Z, (plan, fft_buffer, fft_buffer, CUFFT_FORWARD));
}

extern "C" void cufft_backward_transform(cufftHandle plan, cuDoubleComplex* fft_buffer)
{
    //CUDA_timer t("cufft_backward_transform");
    CALL_CUFFT(cufftExecZ2Z, (plan, fft_buffer, fft_buffer, CUFFT_INVERSE));
}

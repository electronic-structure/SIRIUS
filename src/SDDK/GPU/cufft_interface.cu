#include "cuda.hpp"
#include "cuda_common.h"

__global__ void cufft_batch_load_gpu_kernel(int                    fft_size, 
                                            int                    num_pw_components, 
                                            int const*             map, 
                                            cuDoubleComplex const* data, 
                                            cuDoubleComplex*       fft_buffer)
{
    int i = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pw_components) {
        fft_buffer[array2D_offset(map[idx], i, fft_size)] = data[array2D_offset(idx, i, num_pw_components)];
    }
}

extern "C" void cufft_batch_load_gpu(int                    fft_size,
                                     int                    num_pw_components, 
                                     int                    num_fft,
                                     int const*             map, 
                                     cuDoubleComplex const* data, 
                                     cuDoubleComplex*       fft_buffer,
                                     int                    stream_id)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);

    cudaStream_t stream = acc::stream(stream_id);

    acc::zero(fft_buffer, fft_size * num_fft);

    cufft_batch_load_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
    (
        fft_size,
        num_pw_components,
        map,
        data, 
        fft_buffer
    );
}

__global__ void cufft_load_x0y0_col_gpu_kernel(int                    z_col_size,
                                               int const*             map,
                                               cuDoubleComplex const* data,
                                               cuDoubleComplex*       fft_buffer)

{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < z_col_size) {
        fft_buffer[map[idx]] = make_cuDoubleComplex(data[idx].x, -data[idx].y);
    }
}

extern "C" void cufft_load_x0y0_col_gpu(int                    z_col_size,
                                        int const*             map,
                                        cuDoubleComplex const* data,
                                        cuDoubleComplex*       fft_buffer,
                                        int                    stream_id)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(z_col_size, grid_t.x));

    cudaStream_t stream = acc::stream(stream_id);

    cufft_load_x0y0_col_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
    (
        z_col_size,
        map,
        data,
        fft_buffer
    );
}

__global__ void cufft_batch_unload_gpu_kernel(int                    fft_size, 
                                              int                    num_pw_components, 
                                              int const*             map, 
                                              cuDoubleComplex const* fft_buffer,
                                              cuDoubleComplex*       data,
                                              double                 alpha,
                                              double                 beta)
{
    int i = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_pw_components) {
        data[array2D_offset(idx, i, num_pw_components)] = cuCadd(
            cuCmul(make_cuDoubleComplex(alpha, 0), data[array2D_offset(idx, i, num_pw_components)]),
            cuCmul(make_cuDoubleComplex(beta, 0), fft_buffer[array2D_offset(map[idx], i, fft_size)]));
    }
}

/// Unload data from FFT buffer.
/** The following operation is executed:
 *  data[ig] = alpha * data[ig] + beta * fft_buffer[map[ig]] */
extern "C" void cufft_batch_unload_gpu(int                    fft_size,
                                       int                    num_pw_components,
                                       int                    num_fft,
                                       int const*             map, 
                                       cuDoubleComplex const* fft_buffer, 
                                       cuDoubleComplex*       data,
                                       double                 alpha,
                                       double                 beta,
                                       int                    stream_id)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);

    cudaStream_t stream = acc::stream(stream_id);
    
    cufft_batch_unload_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
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

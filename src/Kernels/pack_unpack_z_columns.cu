#include "kernels_common.h"

template <int direction>
__global__ void pack_unpack_z_cols_gpu_kernel
(
    cuDoubleComplex* z_cols_packed__,
    cuDoubleComplex* fft_buf__,
    int size_x__,
    int size_y__,
    int size_z__,
    int num_z_cols__,
    int const* z_columns_pos__
)
{
    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    if (icol < num_z_cols__)
    {
        int x = z_columns_pos__[array2D_offset(0, icol, 2)];
        int y = z_columns_pos__[array2D_offset(1, icol, 2)];
        
        for (int z = 0; z < size_z__; z++)
        {
            if (direction == 1)
            {
                fft_buf__[array3D_offset(x, y, z, size_x__, size_y__)] = z_cols_packed__[array2D_offset(z, icol, size_z__)];
            }
            if (direction == -1)
            {
                z_cols_packed__[array2D_offset(z, icol, size_z__)] = fft_buf__[array3D_offset(x, y, z, size_x__, size_y__)];
            }
        }
    }
}

extern "C" void unpack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                                  cuDoubleComplex* fft_buf__,
                                  int size_x__,
                                  int size_y__,
                                  int size_z__,
                                  int num_z_cols__,
                                  int const* z_columns_pos__,
                                  int stream_id__)
{
    cudaStream_t stream = (stream_id__ == -1) ? NULL : streams[stream_id__];

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x));

    cudaMemsetAsync(fft_buf__, 0, size_x__ * size_y__ * size_z__ * sizeof(cuDoubleComplex), stream);

    pack_unpack_z_cols_gpu_kernel<1> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_columns_pos__
    );
}

extern "C" void pack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                                cuDoubleComplex* fft_buf__,
                                int size_x__,
                                int size_y__,
                                int size_z__,
                                int num_z_cols__,
                                int const* z_columns_pos__,
                                int stream_id__)
{
    cudaStream_t stream = (stream_id__ == -1) ? NULL : streams[stream_id__];

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x));

    pack_unpack_z_cols_gpu_kernel<-1> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_columns_pos__
    );
}




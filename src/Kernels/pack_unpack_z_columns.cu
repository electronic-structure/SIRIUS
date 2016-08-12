#include "kernels_common.h"

template <int direction, bool conjugate>
__global__ void pack_unpack_z_cols_gpu_kernel
(
    cuDoubleComplex* z_cols_packed__,
    cuDoubleComplex* fft_buf__,
    int size_x__,
    int size_y__,
    int size_z__,
    int num_z_cols__,
    int const* z_col_pos__
)
{
    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y;
    int size_xy = size_x__ * size_y__;
    if (icol < num_z_cols__) {
        int ipos = z_col_pos__[icol];
        /* load into buffer */
        if (direction == 1) {
            if (conjugate) {
                fft_buf__[array2D_offset(ipos, iz, size_xy)] = cuConj(z_cols_packed__[array2D_offset(iz, icol, size_z__)]);
            }
            else {
                fft_buf__[array2D_offset(ipos, iz, size_xy)] = z_cols_packed__[array2D_offset(iz, icol, size_z__)];
            }
        }
        if (direction == -1) {
            z_cols_packed__[array2D_offset(iz, icol, size_z__)] = fft_buf__[array2D_offset(ipos, iz, size_xy)];
        }
    }
}

extern "C" void unpack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                                  cuDoubleComplex* fft_buf__,
                                  int size_x__,
                                  int size_y__,
                                  int size_z__,
                                  int num_z_cols__,
                                  int const* z_col_pos__,
                                  bool use_reduction__, 
                                  int stream_id__)
{
    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    cudaMemsetAsync(fft_buf__, 0, size_x__ * size_y__ * size_z__ * sizeof(cuDoubleComplex), stream);

    pack_unpack_z_cols_gpu_kernel<1, false> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_col_pos__
    );
    if (use_reduction__) {
        pack_unpack_z_cols_gpu_kernel<1, true> <<<grid_b, grid_t, 0, stream>>>
        (
            &z_cols_packed__[size_z__], // skip first column for {-x, -y} coordinates
            fft_buf__,
            size_x__,
            size_y__,
            size_z__,
            num_z_cols__ - 1,
            &z_col_pos__[num_z_cols__ + 1] // skip first column for {-x, -y} coordinates
        );
    }
}

extern "C" void pack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                                cuDoubleComplex* fft_buf__,
                                int size_x__,
                                int size_y__,
                                int size_z__,
                                int num_z_cols__,
                                int const* z_col_pos__,
                                int stream_id__)
{
    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    pack_unpack_z_cols_gpu_kernel<-1, false> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_col_pos__
    );
}

template <int direction>
__global__ void pack_unpack_z_cols_2_gpu_kernel
(
    cuDoubleComplex* z_cols_packed1__,
    cuDoubleComplex* z_cols_packed2__,
    cuDoubleComplex* fft_buf__,
    int size_x__,
    int size_y__,
    int size_z__,
    int num_z_cols__,
    int const* z_columns_pos__
)
{
    int icol = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y;
    if (icol < num_z_cols__)
    {
        int x = (z_columns_pos__[array2D_offset(0, icol, 2)] + size_x__) % size_x__;
        int y = (z_columns_pos__[array2D_offset(1, icol, 2)] + size_y__) % size_y__;
        int mx = (-z_columns_pos__[array2D_offset(0, icol, 2)] + size_x__) % size_x__;
        int my = (-z_columns_pos__[array2D_offset(1, icol, 2)] + size_y__) % size_y__;

        /* load into buffer */
        if (direction == 1)
        {
            fft_buf__[array3D_offset(x, y, iz, size_x__, size_y__)] = cuCadd(z_cols_packed1__[array2D_offset(iz, icol, size_z__)],
                cuCmul(make_cuDoubleComplex(0, 1), z_cols_packed2__[array2D_offset(iz, icol, size_z__)]));
            
            fft_buf__[array3D_offset(mx, my, iz, size_x__, size_y__)] = cuCadd(cuConj(z_cols_packed1__[array2D_offset(iz, icol, size_z__)]),
                cuCmul(make_cuDoubleComplex(0, 1), cuConj(z_cols_packed2__[array2D_offset(iz, icol, size_z__)])));
        }
        if (direction == -1)
        {
            z_cols_packed1__[array2D_offset(iz, icol, size_z__)] = cuCmul(make_cuDoubleComplex(0.5, 0),
                cuCadd(fft_buf__[array3D_offset(x, y, iz, size_x__, size_y__)], cuConj(fft_buf__[array3D_offset(mx, my, iz, size_x__, size_y__)])));

            z_cols_packed2__[array2D_offset(iz, icol, size_z__)] = cuCmul(make_cuDoubleComplex(0, -0.5),
                cuCsub(fft_buf__[array3D_offset(x, y, iz, size_x__, size_y__)], cuConj(fft_buf__[array3D_offset(mx, my, iz, size_x__, size_y__)])));
        }
    }
}

extern "C" void unpack_z_cols_2_gpu(cuDoubleComplex* z_cols_packed1__,
                                    cuDoubleComplex* z_cols_packed2__,
                                    cuDoubleComplex* fft_buf__,
                                    int size_x__,
                                    int size_y__,
                                    int size_z__,
                                    int num_z_cols__,
                                    int const* z_columns_pos__,
                                    int stream_id__)
{
    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    cudaMemsetAsync(fft_buf__, 0, size_x__ * size_y__ * size_z__ * sizeof(cuDoubleComplex), stream);

    pack_unpack_z_cols_2_gpu_kernel<1> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed1__,
        z_cols_packed2__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_columns_pos__
    );
}

extern "C" void pack_z_cols_2_gpu(cuDoubleComplex* z_cols_packed1__,
                                  cuDoubleComplex* z_cols_packed2__,
                                  cuDoubleComplex* fft_buf__,
                                  int size_x__,
                                  int size_y__,
                                  int size_z__,
                                  int num_z_cols__,
                                  int const* z_columns_pos__,
                                  int stream_id__)
{
    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    pack_unpack_z_cols_2_gpu_kernel<-1> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed1__,
        z_cols_packed2__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_columns_pos__
    );
}


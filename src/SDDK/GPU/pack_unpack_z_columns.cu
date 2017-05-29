#include "cuda_common.h"
#include "cuda.hpp"

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
    cudaStream_t stream = acc::stream(stream_id__);

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
    cudaStream_t stream = acc::stream(stream_id__);

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

template <int direction, bool conjugate>
__global__ void pack_unpack_two_z_cols_gpu_kernel
(
    cuDoubleComplex* z_cols_packed1__,
    cuDoubleComplex* z_cols_packed2__,
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
        /* load into buffer */
        if (direction == 1) {
            int ipos = z_col_pos__[icol];
            cuDoubleComplex z1 = z_cols_packed1__[array2D_offset(iz, icol, size_z__)];
            cuDoubleComplex z2 = z_cols_packed2__[array2D_offset(iz, icol, size_z__)];
            if (conjugate) {
                /* conj(z1) + I * conj(z2) */
                fft_buf__[array2D_offset(ipos, iz, size_xy)] = make_cuDoubleComplex(z1.x + z2.y, z2.x - z1.y);
            }
            else {
                /* z1 + I * z2 */
                fft_buf__[array2D_offset(ipos, iz, size_xy)] = make_cuDoubleComplex(z1.x - z2.y, z1.y + z2.x);
            }
        }
        if (direction == -1) {
            int ipos1 = z_col_pos__[icol];
            int ipos2 = z_col_pos__[num_z_cols__ + icol];
            cuDoubleComplex z1 = fft_buf__[array2D_offset(ipos1, iz, size_xy)];
            cuDoubleComplex z2 = fft_buf__[array2D_offset(ipos2, iz, size_xy)];

            z_cols_packed1__[array2D_offset(iz, icol, size_z__)] = make_cuDoubleComplex(0.5 * (z1.x + z2.x), 0.5 * (z1.y - z2.y));
            z_cols_packed2__[array2D_offset(iz, icol, size_z__)] = make_cuDoubleComplex(0.5 * (z1.y + z2.y), 0.5 * (z2.x - z1.x));
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
                                    int const* z_col_pos__,
                                    int stream_id__)
{
    cudaStream_t stream = acc::stream(stream_id__);

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    cudaMemsetAsync(fft_buf__, 0, size_x__ * size_y__ * size_z__ * sizeof(cuDoubleComplex), stream);

    pack_unpack_two_z_cols_gpu_kernel<1, false> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed1__,
        z_cols_packed2__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_col_pos__
    );
    pack_unpack_two_z_cols_gpu_kernel<1, true> <<<grid_b, grid_t, 0, stream>>>
    (
        &z_cols_packed1__[size_z__], // skip first column for {-x, -y} coordinates
        &z_cols_packed2__[size_z__], // skip first column for {-x, -y} coordinates
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__ - 1,
        &z_col_pos__[num_z_cols__ + 1] // skip first column for {-x, -y} coordinates
    );
}

extern "C" void pack_z_cols_2_gpu(cuDoubleComplex* z_cols_packed1__,
                                  cuDoubleComplex* z_cols_packed2__,
                                  cuDoubleComplex* fft_buf__,
                                  int size_x__,
                                  int size_y__,
                                  int size_z__,
                                  int num_z_cols__,
                                  int const* z_col_pos__,
                                  int stream_id__)
{
    cudaStream_t stream = acc::stream(stream_id__);

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    pack_unpack_two_z_cols_gpu_kernel<-1, false> <<<grid_b, grid_t, 0, stream>>>
    (
        z_cols_packed1__,
        z_cols_packed2__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_col_pos__
    );
}


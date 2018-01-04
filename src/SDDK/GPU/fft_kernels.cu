#include "cuda.hpp"
#include "cuda_common.hpp"

#include <stdio.h>


__global__ void cufft_repack_z_buffer_back_kernel(int dimz,
                                             int num_zcol_loc, 
                                             int const* local_z_offsets,
                                             int const* local_z_sizes,
                                             cuDoubleComplex const* old_buffer,
                                             cuDoubleComplex* new_buffer)
{
    int iz = blockDim.x * blockIdx.x + threadIdx.x;
    int izcol = blockIdx.y;
    int rank = blockIdx.z;
    
    int local_zsize = local_z_sizes[rank];
    if (iz < local_zsize) {
        int offs = local_z_offsets[rank];
        new_buffer[offs + iz + izcol * dimz] = old_buffer[offs * num_zcol_loc + izcol * local_zsize + iz];
    }
}


__global__ void cufft_repack_z_buffer_kernel(int dimz,
                                             int num_zcol_loc, 
                                             int const* local_z_offsets,
                                             int const* local_z_sizes,
                                             cuDoubleComplex* old_buffer,
                                             cuDoubleComplex* new_buffer)
{
    int iz = blockDim.x * blockIdx.x + threadIdx.x;
    int izcol = blockIdx.y;
    int rank = blockIdx.z;
    
    int local_zsize = local_z_sizes[rank];
    if (iz < local_zsize) {
        int offs = local_z_offsets[rank];
        new_buffer[offs * num_zcol_loc + izcol * local_zsize + iz] = old_buffer[offs + iz + izcol * dimz];
    }
}


extern "C" void cufft_repack_z_buffer(int direction,
                                      int num_ranks,
                                      int dimz,
                                      int num_zcol_loc,
                                      int zcol_max_size,
                                      int const* local_z_offsets,
                                      int const* local_z_sizes,
                                      cuDoubleComplex* serial_buffer,
                                      cuDoubleComplex* parallel_buffer)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(zcol_max_size, grid_t.x), num_zcol_loc, num_ranks);

    if (direction == 1) {
        cufft_repack_z_buffer_kernel<<<grid_b, grid_t, 0, 0>>>
                                                   (dimz,
                                                    num_zcol_loc,
                                                    local_z_offsets,
                                                    local_z_sizes,
                                                    serial_buffer,
                                                    parallel_buffer);
    } else { 
        cufft_repack_z_buffer_back_kernel<<<grid_b, grid_t, 0, 0>>>
                                                   (dimz,
                                                    num_zcol_loc,
                                                    local_z_offsets,
                                                    local_z_sizes,
                                                    parallel_buffer,
                                                    serial_buffer);
    }

}



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
        cuDoubleComplex z1 = data[array2D_offset(idx, i, num_pw_components)];
        cuDoubleComplex z2 = fft_buffer[array2D_offset(map[idx], i, fft_size)];
        data[array2D_offset(idx, i, num_pw_components)] = make_cuDoubleComplex(alpha * z1.x + beta * z2.x, alpha * z1.y + beta * z2.y);

        //data[array2D_offset(idx, i, num_pw_components)] = cuCadd(
        //    cuCmul(make_cuDoubleComplex(alpha, 0), data[array2D_offset(idx, i, num_pw_components)]),
        //    cuCmul(make_cuDoubleComplex(beta, 0), fft_buffer[array2D_offset(map[idx], i, fft_size)]));
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

    if (alpha == 0) {
        acc::zero(data, num_pw_components);
    }
    
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

template <int direction, bool conjugate>
__global__ void pack_unpack_z_cols_gpu_kernel(cuDoubleComplex* z_cols_packed__,
                                              cuDoubleComplex* fft_buf__,
                                              int              size_x__,
                                              int              size_y__,
                                              int              size_z__,
                                              int              num_z_cols__,
                                              int const*       z_col_pos__)
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
                                  int              size_x__,
                                  int              size_y__,
                                  int              size_z__,
                                  int              num_z_cols__,
                                  int const*       z_col_pos__,
                                  bool             use_reduction__, 
                                  int              stream_id__)
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
                                int              size_x__,
                                int              size_y__,
                                int              size_z__,
                                int              num_z_cols__,
                                int const*       z_col_pos__,
                                int              stream_id__)
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
__global__ void pack_unpack_two_z_cols_gpu_kernel(cuDoubleComplex* z_cols_packed1__,
                                                  cuDoubleComplex* z_cols_packed2__,
                                                  cuDoubleComplex* fft_buf__,
                                                  int              size_x__,
                                                  int              size_y__,
                                                  int              size_z__,
                                                  int              num_z_cols__,
                                                  int const*       z_col_pos__)
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
                                    int              size_x__,
                                    int              size_y__,
                                    int              size_z__,
                                    int              num_z_cols__,
                                    int const*       z_col_pos__,
                                    int              stream_id__)
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
                                  int              size_x__,
                                  int              size_y__,
                                  int              size_z__,
                                  int              num_z_cols__,
                                  int const*       z_col_pos__,
                                  int              stream_id__)
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


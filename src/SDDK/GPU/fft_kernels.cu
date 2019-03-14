// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file fft_kernels.cu
 *
 *  \brief Contains implementaiton of CUDA and ROCM kernels necessary for a FFT driver.
 */

#include <stdio.h>
#include "cuda_common.hpp"
#include "acc_runtime.hpp"

//NOTE: HIP will call the corresponding CUDA function if compiled with CUDA support


template <int direction>
__global__ void repack_z_buffer_gpu_kernel(int size_z,
                                           int num_zcol_loc,
                                           int const* local_z_offsets,
                                           int const* local_z_sizes,
                                           acc_complex_double_t* z_sticks_local,
                                           acc_complex_double_t* a2a_buffer)
{
    int iz = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int izcol = hipBlockIdx_y;
    int rank = hipBlockIdx_z;

    int local_zsize = local_z_sizes[rank];
    if (iz < local_zsize) {
        int offs = local_z_offsets[rank];
        if (direction == -1) {
            z_sticks_local[offs + iz + izcol * size_z] = a2a_buffer[offs * num_zcol_loc + izcol * local_zsize + iz];
        }
        if (direction == 1) {
            a2a_buffer[offs * num_zcol_loc + izcol * local_zsize + iz] = z_sticks_local[offs + iz + izcol * size_z];
        }
    }
}

extern "C" void repack_z_buffer_gpu(int direction,
                                    int num_ranks,
                                    int size_z,
                                    int num_zcol_loc,
                                    int zcol_max_size,
                                    int const* local_z_offsets,
                                    int const* local_z_sizes,
                                    acc_complex_double_t* z_sticks_local,
                                    acc_complex_double_t* a2a_buffer)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(zcol_max_size, grid_t.x), num_zcol_loc, num_ranks);

    if (direction == 1) {
        accLaunchKernel((repack_z_buffer_gpu_kernel<1>), dim3(grid_b), dim3(grid_t), 0, 0, 
            size_z,
            num_zcol_loc,
            local_z_offsets,
            local_z_sizes,
            z_sticks_local,
            a2a_buffer
        );
    } else {
        accLaunchKernel((repack_z_buffer_gpu_kernel<-1>), dim3(grid_b), dim3(grid_t), 0, 0, 
            size_z,
            num_zcol_loc,
            local_z_offsets,
            local_z_sizes,
            z_sticks_local,
            a2a_buffer
        );
    }
}



__global__ void batch_load_gpu_kernel(int                    fft_size, 
                                      int                    num_pw_components, 
                                      int const*             map, 
                                      acc_complex_double_t const* data, 
                                      acc_complex_double_t*       fft_buffer)
{
    int i = hipBlockIdx_y;
    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (idx < num_pw_components) {
        fft_buffer[array2D_offset(map[idx], i, fft_size)] = data[array2D_offset(idx, i, num_pw_components)];
    }
}

extern "C" void batch_load_gpu(int                    fft_size,
                               int                    num_pw_components, 
                               int                    num_fft,
                               int const*             map, 
                               acc_complex_double_t const* data, 
                               acc_complex_double_t*       fft_buffer,
                               int                    stream_id__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);

    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    acc::zero(fft_buffer, fft_size*num_fft, stream_id(stream_id__));

    accLaunchKernel((batch_load_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, stream, 
        fft_size,
        num_pw_components,
        map,
        data, 
        fft_buffer
    );
}

__global__ void batch_unload_gpu_kernel(int                    fft_size, 
                                        int                    num_pw_components, 
                                        int const*             map, 
                                        acc_complex_double_t const* fft_buffer,
                                        acc_complex_double_t*       data,
                                        double                 alpha,
                                        double                 beta)
{
    int i = hipBlockIdx_y;
    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (idx < num_pw_components) {
        acc_complex_double_t z1 = data[array2D_offset(idx, i, num_pw_components)];
        acc_complex_double_t z2 = fft_buffer[array2D_offset(map[idx], i, fft_size)];
        data[array2D_offset(idx, i, num_pw_components)] = make_accDoubleComplex(alpha * z1.x + beta * z2.x, alpha * z1.y + beta * z2.y);

        //data[array2D_offset(idx, i, num_pw_components)] = cuCadd(
        //    cuCmul(make_cuDoubleComplex(alpha, 0), data[array2D_offset(idx, i, num_pw_components)]),
        //    cuCmul(make_cuDoubleComplex(beta, 0), fft_buffer[array2D_offset(map[idx], i, fft_size)]));
    }
}

/// Unload data from FFT buffer.
/** The following operation is executed:
 *  data[ig] = alpha * data[ig] + beta * fft_buffer[map[ig]] */
extern "C" void batch_unload_gpu(int                    fft_size,
                                 int                    num_pw_components,
                                 int                    num_fft,
                                 int const*             map, 
                                 acc_complex_double_t const* fft_buffer, 
                                 acc_complex_double_t*       data,
                                 double                 alpha,
                                 double                 beta,
                                 int                    stream_id__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_pw_components, grid_t.x), num_fft);

    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    if (alpha == 0) {
        acc::zero(data, num_pw_components, stream_id(stream_id__));
    }

    accLaunchKernel((batch_unload_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, stream, 
        fft_size, 
        num_pw_components, 
        map, 
        fft_buffer,
        data,
        alpha,
        beta
    );
}

__global__ void load_x0y0_col_gpu_kernel(int                    z_col_size,
                                         int const*             map,
                                         acc_complex_double_t const* data,
                                         acc_complex_double_t*       fft_buffer)

{
    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (idx < z_col_size) {
        fft_buffer[map[idx]] = make_accDoubleComplex(data[idx].x, -data[idx].y);
    }
}

extern "C" void load_x0y0_col_gpu(int                    z_col_size,
                                  int const*             map,
                                  acc_complex_double_t const* data,
                                  acc_complex_double_t*       fft_buffer,
                                  int                    stream_id__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(z_col_size, grid_t.x));

    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    accLaunchKernel((load_x0y0_col_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, stream, 
        z_col_size,
        map,
        data,
        fft_buffer
    );
}

template <int direction, bool conjugate>
__global__ void pack_unpack_z_cols_gpu_kernel(acc_complex_double_t* z_cols_packed__,
                                              acc_complex_double_t* fft_buf__,
                                              int              size_x__,
                                              int              size_y__,
                                              int              size_z__,
                                              int              num_z_cols__,
                                              int const*       z_col_pos__)
{
    int icol = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int iz = hipBlockIdx_y;
    int size_xy = size_x__ * size_y__;
    if (icol < num_z_cols__) {
        int ipos = z_col_pos__[icol];
        /* load into buffer */
        if (direction == 1) {
            if (conjugate) {
                fft_buf__[array2D_offset(ipos, iz, size_xy)].x = z_cols_packed__[array2D_offset(iz, icol, size_z__)].x;
                fft_buf__[array2D_offset(ipos, iz, size_xy)].y = -z_cols_packed__[array2D_offset(iz, icol, size_z__)].y;
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

extern "C" void unpack_z_cols_gpu(acc_complex_double_t* z_cols_packed__,
                                  acc_complex_double_t* fft_buf__,
                                  int              size_x__,
                                  int              size_y__,
                                  int              size_z__,
                                  int              num_z_cols__,
                                  int const*       z_col_pos__,
                                  bool             use_reduction__, 
                                  int              stream_id__)
{
    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    acc::zero(fft_buf__, size_x__ * size_y__ * size_z__, stream_id(stream_id__));

    accLaunchKernel((pack_unpack_z_cols_gpu_kernel<1, false>), dim3(grid_b), dim3(grid_t), 0, stream, 
        z_cols_packed__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_col_pos__
    );
    if (use_reduction__) {
        accLaunchKernel((pack_unpack_z_cols_gpu_kernel<1, true>), dim3(grid_b), dim3(grid_t), 0, stream, 
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

extern "C" void pack_z_cols_gpu(acc_complex_double_t* z_cols_packed__,
                                acc_complex_double_t* fft_buf__,
                                int              size_x__,
                                int              size_y__,
                                int              size_z__,
                                int              num_z_cols__,
                                int const*       z_col_pos__,
                                int              stream_id__)
{
    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    accLaunchKernel((pack_unpack_z_cols_gpu_kernel<-1, false>), dim3(grid_b), dim3(grid_t), 0, stream, 
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
__global__ void pack_unpack_two_z_cols_gpu_kernel(acc_complex_double_t* z_cols_packed1__,
                                                  acc_complex_double_t* z_cols_packed2__,
                                                  acc_complex_double_t* fft_buf__,
                                                  int              size_x__,
                                                  int              size_y__,
                                                  int              size_z__,
                                                  int              num_z_cols__,
                                                  int const*       z_col_pos__)
{
    int icol = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int iz = hipBlockIdx_y;
    int size_xy = size_x__ * size_y__;
    if (icol < num_z_cols__) {
        /* load into buffer */
        if (direction == 1) {
            int ipos = z_col_pos__[icol];
            acc_complex_double_t z1 = z_cols_packed1__[array2D_offset(iz, icol, size_z__)];
            acc_complex_double_t z2 = z_cols_packed2__[array2D_offset(iz, icol, size_z__)];
            if (conjugate) {
                /* conj(z1) + I * conj(z2) */
                fft_buf__[array2D_offset(ipos, iz, size_xy)] = make_accDoubleComplex(z1.x + z2.y, z2.x - z1.y);
            }
            else {
                /* z1 + I * z2 */
                fft_buf__[array2D_offset(ipos, iz, size_xy)] = make_accDoubleComplex(z1.x - z2.y, z1.y + z2.x);
            }
        }
        if (direction == -1) {
            int ipos1 = z_col_pos__[icol];
            int ipos2 = z_col_pos__[num_z_cols__ + icol];
            acc_complex_double_t z1 = fft_buf__[array2D_offset(ipos1, iz, size_xy)];
            acc_complex_double_t z2 = fft_buf__[array2D_offset(ipos2, iz, size_xy)];

            z_cols_packed1__[array2D_offset(iz, icol, size_z__)] = make_accDoubleComplex(0.5 * (z1.x + z2.x), 0.5 * (z1.y - z2.y));
            z_cols_packed2__[array2D_offset(iz, icol, size_z__)] = make_accDoubleComplex(0.5 * (z1.y + z2.y), 0.5 * (z2.x - z1.x));
        }
    }
}

extern "C" void unpack_z_cols_2_gpu(acc_complex_double_t* z_cols_packed1__,
                                    acc_complex_double_t* z_cols_packed2__,
                                    acc_complex_double_t* fft_buf__,
                                    int              size_x__,
                                    int              size_y__,
                                    int              size_z__,
                                    int              num_z_cols__,
                                    int const*       z_col_pos__,
                                    int              stream_id__)
{
    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    acc::zero(fft_buf__, size_x__ * size_y__ * size_z__, stream_id(stream_id__));

    accLaunchKernel((pack_unpack_two_z_cols_gpu_kernel<1, false>), dim3(grid_b), dim3(grid_t), 0, stream, 
        z_cols_packed1__,
        z_cols_packed2__,
        fft_buf__,
        size_x__,
        size_y__,
        size_z__,
        num_z_cols__,
        z_col_pos__
    );
    accLaunchKernel((pack_unpack_two_z_cols_gpu_kernel<1, true>), dim3(grid_b), dim3(grid_t), 0, stream, 
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

extern "C" void pack_z_cols_2_gpu(acc_complex_double_t* z_cols_packed1__,
                                  acc_complex_double_t* z_cols_packed2__,
                                  acc_complex_double_t* fft_buf__,
                                  int              size_x__,
                                  int              size_y__,
                                  int              size_z__,
                                  int              num_z_cols__,
                                  int const*       z_col_pos__,
                                  int              stream_id__)
{
    acc_stream_t stream = (acc_stream_t) acc::stream(stream_id(stream_id__));

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_z_cols__, grid_t.x), size_z__);

    accLaunchKernel((pack_unpack_two_z_cols_gpu_kernel<-1, false>), dim3(grid_b), dim3(grid_t), 0, stream, 
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


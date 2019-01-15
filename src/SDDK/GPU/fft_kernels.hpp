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

/** \file fft_kernels.hpp
 *
 *  \brief Contains definition of CUDA kernels necessary for a FFT driver.
 */

#include "acc.hpp"

extern "C" {

void cufft_repack_z_buffer(int direction,
                           int num_ranks,
                                      int dimz,
                                      int num_zcol_loc,
                                      int zcol_max_size,
                                      int const* local_z_offsets,
                                      int const* local_z_sizes,
                                      cuDoubleComplex* serial_buffer,
                                      cuDoubleComplex* parallel_buffer);


void cufft_batch_load_gpu(int                    fft_size,
                          int                    num_pw_components, 
                          int                    num_fft,
                          int const*             map, 
                          cuDoubleComplex const* data, 
                          cuDoubleComplex*       fft_buffer,
                          int                    stream_id);

void cufft_load_x0y0_col_gpu(int                    z_col_size,
                             int const*             map,
                             cuDoubleComplex const* data,
                             cuDoubleComplex*       fft_buffer,
                             int                    stream_id);

void cufft_batch_unload_gpu(int                    fft_size,
                            int                    num_pw_components,
                            int                    num_fft,
                            int const*             map, 
                            cuDoubleComplex const* fft_buffer, 
                            cuDoubleComplex*       data,
                            double                 alpha,
                            double                 beta,
                            int                    stream_id);

void unpack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                       cuDoubleComplex* fft_buf__,
                       int              size_x__,
                       int              size_y__,
                       int              size_z__,
                       int              num_z_cols__,
                       int const*       z_col_pos__,
                       bool             use_reduction__, 
                       int              stream_id__);

void pack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                     cuDoubleComplex* fft_buf__,
                     int              size_x__,
                     int              size_y__,
                     int              size_z__,
                     int              num_z_cols__,
                     int const*       z_col_pos__,
                     int              stream_id__);

void unpack_z_cols_2_gpu(cuDoubleComplex* z_cols_packed1__,
                         cuDoubleComplex* z_cols_packed2__,
                         cuDoubleComplex* fft_buf__,
                         int              size_x__,
                         int              size_y__,
                         int              size_z__,
                         int              num_z_cols__,
                         int const*       z_col_pos__,
                         int              stream_id__);

void pack_z_cols_2_gpu(cuDoubleComplex* z_cols_packed1__,
                       cuDoubleComplex* z_cols_packed2__,
                       cuDoubleComplex* fft_buf__,
                       int              size_x__,
                       int              size_y__,
                       int              size_z__,
                       int              num_z_cols__,
                       int const*       z_col_pos__,
                       int              stream_id__);
}

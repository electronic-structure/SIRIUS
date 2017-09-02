#include "cuda.hpp"

extern "C" {


void cufft_repack_z_buffer(int num_ranks,
                                      int dimz,
                                      int num_zcol_loc,
                                      int zcol_max_size,
                                      int const* local_z_offsets,
                                      int const* local_z_sizes,
                                      cuDoubleComplex const* old_buffer,
                                      cuDoubleComplex* new_buffer);

     
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

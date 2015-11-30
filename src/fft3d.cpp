// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file fft3d_cpu.cpp
 *   
 *  \brief Implementation
 */

#include "fft3d.h"

#ifdef __GPU
extern "C" void unpack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                                  cuDoubleComplex* fft_buf__,
                                  int size_x__,
                                  int size_y__,
                                  int size_z__,
                                  int num_z_cols__,
                                  int const* z_columns_pos__,
                                  int stream_id__);

extern "C" void pack_z_cols_gpu(cuDoubleComplex* z_cols_packed__,
                                cuDoubleComplex* fft_buf__,
                                int size_x__,
                                int size_y__,
                                int size_z__,
                                int num_z_cols__,
                                int const* z_columns_pos__,
                                int stream_id__);
#endif

namespace sirius {

FFT3D::FFT3D(vector3d<int> dims__,
             int num_fft_workers__,
             Communicator const& comm__,
             processing_unit_t pu__)
    : num_fft_workers_(num_fft_workers__),
      comm_(comm__),
      pu_(pu__),
      cufft3d_(false)
      #ifdef __GPU
      ,cufft_nbatch_(0),
      allocated_on_device_(false)
      #endif
      
{
    PROFILE();

    grid_ = FFT3D_grid(dims__);

    /* split z-direction */
    spl_z_ = splindex<block>(grid_.size(2), comm_.size(), comm_.rank());
    local_size_z_ = spl_z_.local_size();
    offset_z_ = spl_z_.global_offset();

    /* allocate main buffer */
    fft_buffer_ = mdarray<double_complex, 1>(local_size(), "fft_buffer_");
    
    /* allocate 1d and 2d buffers */
    for (int i = 0; i < num_fft_workers_; i++)
    {
        fftw_buffer_z_.push_back((double_complex*)fftw_malloc(grid_.size(2) * sizeof(double_complex)));
        fftw_buffer_xy_.push_back((double_complex*)fftw_malloc(grid_.size(0) * grid_.size(1) * sizeof(double_complex)));
    }

    plan_forward_z_   = std::vector<fftw_plan>(num_fft_workers_);
    plan_forward_xy_  = std::vector<fftw_plan>(num_fft_workers_);
    plan_backward_z_  = std::vector<fftw_plan>(num_fft_workers_);
    plan_backward_xy_ = std::vector<fftw_plan>(num_fft_workers_);

    for (int i = 0; i < num_fft_workers_; i++)
    {
        plan_forward_z_[i] = fftw_plan_dft_1d(grid_.size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                              (fftw_complex*)fftw_buffer_z_[i], FFTW_FORWARD, FFTW_ESTIMATE);

        plan_backward_z_[i] = fftw_plan_dft_1d(grid_.size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                               (fftw_complex*)fftw_buffer_z_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
        
        plan_forward_xy_[i] = fftw_plan_dft_2d(grid_.size(1), grid_.size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                               (fftw_complex*)fftw_buffer_xy_[i], FFTW_FORWARD, FFTW_ESTIMATE);

        plan_backward_xy_[i] = fftw_plan_dft_2d(grid_.size(1), grid_.size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                                (fftw_complex*)fftw_buffer_xy_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    
    #ifdef __GPU
    if (pu_ == GPU)
    {
        int auto_alloc = 0;
        if (comm_.size() == 1 && cufft3d_)
        {
            cufft_nbatch_ = 1;

            int dims[] = {grid_.size(2), grid_.size(1), grid_.size(0)};
            cufft_create_plan_handle(&cufft_plan_);
            cufft_create_batch_plan(cufft_plan_, 3, dims, dims, 1, 1, cufft_nbatch_, auto_alloc);
        }
        else
        {
            /* GPU will take care of this number of xy-planes */
            cufft_nbatch_ = static_cast<int>(0.5 * local_size_z_);

            int dim_xy[] = {grid_.size(1), grid_.size(0)};
            int embed_xy[] = {grid_.size(1), grid_.size(0)};

            cufft_create_plan_handle(&cufft_plan_xy_);
            cufft_create_batch_plan(cufft_plan_xy_, 2, dim_xy, embed_xy, 1, grid_.size(0) * grid_.size(1), cufft_nbatch_, auto_alloc);
            cufft_set_stream(cufft_plan_xy_, 0);
        }
    }
    #endif
}

FFT3D::~FFT3D()
{
    for (int i = 0; i < num_fft_workers_; i++)
    {
        fftw_free(fftw_buffer_z_[i]);
        fftw_free(fftw_buffer_xy_[i]);
    }

    for (int i = 0; i < num_fft_workers_; i++)
    {
        fftw_destroy_plan(plan_forward_z_[i]);
        fftw_destroy_plan(plan_forward_xy_[i]);
        fftw_destroy_plan(plan_backward_z_[i]);
        fftw_destroy_plan(plan_backward_xy_[i]);
    }
    #ifdef __GPU
    if (pu_ == GPU)
    {
        if (comm_.size() == 1 && cufft3d_)
        {
            cufft_destroy_plan_handle(cufft_plan_);
        }
        else
        {
            cufft_destroy_plan_handle(cufft_plan_xy_);
        }
    }
    #endif
}

template <int direction>
void FFT3D::transform_xy(Gvec const& gvec__)
{
    int size_xy = grid_.size(0) * grid_.size(1);
    int first_z = 0;

    #ifdef __GPU
    if (pu_ == GPU)
    {
        /* stream #0 will be doing cuFFT */
        switch (direction)
        {
            case 1:
            {
                /* srteam #0 copies packed columns to GPU */
                acc::copyin(fft_buffer_aux_.at<GPU>(), cufft_nbatch_, fft_buffer_aux_.at<CPU>(), local_size_z_,
                            cufft_nbatch_, static_cast<int>(gvec__.z_columns().size()), 0);

                /* srteam #0 unpacks z-columns into proper position of FFT buffer */
                unpack_z_cols_gpu(fft_buffer_aux_.at<GPU>(), fft_buffer_.at<GPU>(), grid_.size(0), grid_.size(1), 
                                  cufft_nbatch_, static_cast<int>(gvec__.z_columns().size()), gvec__.z_columns_pos().at<GPU>(), 0);
                /* stream #0 executes FFT */
                cufft_backward_transform(cufft_plan_xy_, fft_buffer_.at<GPU>());
                break;
            }
            case -1:
            {
                /* stream #1 copies part of FFT buffer to CPU */
                acc::copyout(fft_buffer_.at<CPU>(cufft_nbatch_ * size_xy), fft_buffer_.at<GPU>(cufft_nbatch_ * size_xy),
                             size_xy * (local_size_z_ - cufft_nbatch_), 1);
                /* stream #0 executes FFT */
                cufft_forward_transform(cufft_plan_xy_, fft_buffer_.at<GPU>());
                /* stream #0 packs z-columns */
                pack_z_cols_gpu(fft_buffer_aux_.at<GPU>(), fft_buffer_.at<GPU>(), grid_.size(0), grid_.size(1), 
                                cufft_nbatch_, static_cast<int>(gvec__.z_columns().size()), gvec__.z_columns_pos().at<GPU>(), 0);

                /* srteam #0 copies packed columns to CPU */
                acc::copyout(fft_buffer_aux_.at<CPU>(), local_size_z_, fft_buffer_aux_.at<GPU>(), cufft_nbatch_, 
                             cufft_nbatch_, static_cast<int>(gvec__.z_columns().size()), 0);
                /* stream #1 waits to complete memory copy */
                cuda_stream_synchronize(1);
                break;
            }
        }
        first_z = cufft_nbatch_;
    }
    #endif

    #pragma omp parallel num_threads(num_fft_workers_)
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int iz = first_z; iz < local_size_z_; iz++)
        {
            switch (direction)
            {
                case 1:
                {
                    /* clear xy-buffer */
                    std::memset(fftw_buffer_xy_[tid], 0, sizeof(double_complex) * size_xy);
                    /* load z-columns into proper location */
                    for (size_t i = 0; i < gvec__.z_columns().size(); i++)
                    {
                        int x = gvec__.z_column(i).x;
                        int y = gvec__.z_column(i).y;
                        if (x < 0) x += grid_.size(0);
                        if (y < 0) y += grid_.size(1);

                        fftw_buffer_xy_[tid][x + y * grid_.size(0)] = fft_buffer_aux_[iz + local_size_z_ * i];
                    }

                    /* execute local FFT transform */
                    fftw_execute(plan_backward_xy_[tid]);

                    /* copy xy plane to the FFT buffer */
                    std::memcpy(&fft_buffer_[iz * size_xy], fftw_buffer_xy_[tid], sizeof(fftw_complex) * size_xy);
                    break;
                }
                case -1:
                {
                    /* copy xy plane from the fft buffer */
                    std::memcpy(fftw_buffer_xy_[tid], &fft_buffer_[iz * size_xy], sizeof(fftw_complex) * size_xy);

                    /* execute local FFT transform */
                    fftw_execute(plan_forward_xy_[tid]);

                    /* get z-columns */
                    for (size_t i = 0; i < gvec__.z_columns().size(); i++)
                    {
                        int x = gvec__.z_column(i).x;
                        int y = gvec__.z_column(i).y;
                        if (x < 0) x += grid_.size(0);
                        if (y < 0) y += grid_.size(1);

                        fft_buffer_aux_(iz + local_size_z_ * i) = fftw_buffer_xy_[tid][x + y * grid_.size(0)];
                    }
                    break;
                }
                default:
                {
                    TERMINATE("wrong direction");
                }
            }
        }
    }
        
    #ifdef __GPU
    if (pu_ == GPU)
    {
        if (direction == 1)
        {
            /* stream #1 copies data to GPU */
            acc::copyin(fft_buffer_.at<GPU>(cufft_nbatch_ * size_xy), fft_buffer_.at<CPU>(cufft_nbatch_ * size_xy),
                        size_xy * (local_size_z_ - cufft_nbatch_), 1);
            cuda_stream_synchronize(1);
        }
        /* wait for stram #0 */
        cuda_stream_synchronize(0);
    }
    #endif
}

template <int direction, bool use_reduction>
void FFT3D::transform_z_serial(Gvec const& gvec__, double_complex* data__)
{
    double norm = 1.0 / size();
    #pragma omp parallel num_threads(num_fft_workers_)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < gvec__.z_columns().size(); i++)
        {
            int offset = gvec__.z_column(i).offset;

            switch (direction)
            {
                case 1:
                {
                    /* zero input FFT buffer */
                    std::memset(fftw_buffer_z_[tid], 0, grid_.size(2) * sizeof(double_complex));
                    /* load column into local FFT buffer */
                    for (size_t j = 0; j < gvec__.z_column(i).z.size(); j++)
                    {
                        /* z-coordinate of G-vector */
                        int z = gvec__.z_column(i).z[j];
                        /* coordinate inside FFT grid */
                        if (z < 0) z += grid_.size(2);
                        fftw_buffer_z_[tid][z] = data__[offset + j];
                    }
                    /* column with {x,y} = {0,0} has only non-negative z components */
                    if (use_reduction && !gvec__.z_column(i).x && !gvec__.z_column(i).y)
                    {
                        /* load remaining part of {0,0,z} column */
                        for (size_t j = 0; j < gvec__.z_column(i).z.size(); j++)
                        {
                            int z = -gvec__.z_column(i).z[j];
                            if (z < 0) z += grid_.size(2);
                            fftw_buffer_z_[tid][z] = std::conj(data__[offset + j]);
                        }
                    }

                    /* execute 1D transform of a z-column */
                    fftw_execute(plan_backward_z_[tid]);

                    /* load full column into auxiliary buffer */
                    std::memcpy(&fft_buffer_aux_[i * grid_.size(2)], fftw_buffer_z_[tid],
                                grid_.size(2) * sizeof(double_complex));

                    if (use_reduction && (gvec__.z_columns()[i].x || gvec__.z_columns()[i].y))
                    {
                        STOP();
                        ///* x,y coordinates of inverse G-vectors */
                        //int x = -gvec__.z_columns()[i].x;
                        //int y = -gvec__.z_columns()[i].y;
                        ///* coordinates inside FFT grid */
                        //if (x < 0) x += grid_.size(0);
                        //if (y < 0) y += grid_.size(1);
                        ///* load conjugated z-column into main FFT buffer */
                        //for (int z = 0; z < grid_.size(2); z++)
                        //{
                        //    fft_buffer_[x + y * grid_.size(0) + z * size_xy] = std::conj(fftw_buffer_z_[tid][z]);
                        //}
                    }
                    break;
                }
                case -1:
                {
                    /* load full column from auxiliary buffer */
                    std::memcpy(fftw_buffer_z_[tid], &fft_buffer_aux_[i * grid_.size(2)],
                                grid_.size(2) * sizeof(double_complex));

                    /* execute 1D transform of a z-column */
                    fftw_execute(plan_forward_z_[tid]);

                    /* store PW coefficients */
                    for (size_t j = 0; j < gvec__.z_column(i).z.size(); j++)
                    {
                        int z = gvec__.z_column(i).z[j];
                        if (z < 0) z += grid_.size(2);
                        data__[offset + j] = fftw_buffer_z_[tid][z] * norm;
                    }
                    break;
                }
                default:
                {
                    TERMINATE("wrong direction");
                }
            }
        }
    }
}

template <int direction>
void FFT3D::transform_z_parallel(Gvec const& gvec__, double_complex* data__)
{
    int rank = comm_.rank();
    int num_zcol_local = gvec__.zcol_fft_distr().counts[rank];
    double norm = 1.0 / size();

    if (direction == -1)
    {
        block_data_descriptor send(comm_.size());
        block_data_descriptor recv(comm_.size());
        for (int r = 0; r < comm_.size(); r++)
        {
            send.counts[r] = spl_z_.local_size(rank) * gvec__.zcol_fft_distr().counts[r];
            recv.counts[r] = spl_z_.local_size(r) * gvec__.zcol_fft_distr().counts[rank];
        }
        send.calc_offsets();
        recv.calc_offsets();

        std::memcpy(&fft_buffer_[0], &fft_buffer_aux_[0], gvec__.z_columns().size() * local_size_z_ * sizeof(double_complex)); 

        comm_.alltoall(&fft_buffer_[0], &send.counts[0], &send.offsets[0], &fft_buffer_aux_(0), &recv.counts[0], &recv.offsets[0]);
    }

    #pragma omp parallel num_threads(num_fft_workers_)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < num_zcol_local; i++)
        {
            int icol = gvec__.zcol_fft_distr().offsets[rank] + i;
            int offset = gvec__.z_column(icol).offset;

            switch (direction)
            {
                case 1:
                {
                    /* clear z buffer */
                    std::memset(fftw_buffer_z_[tid], 0, grid_.size(2) * sizeof(double_complex));
                    /* load z column into buffer */
                    for (size_t j = 0; j < gvec__.z_column(icol).z.size(); j++)
                    {
                        int z = gvec__.z_column(icol).z[j];
                        if (z < 0) z += grid_.size(2);

                        fftw_buffer_z_[tid][z] = data__[offset + j];
                    }

                    /* perform local FFT transform of a column */
                    fftw_execute(plan_backward_z_[tid]);

                    /* redistribute z-column for a forthcoming all-to-all */ 
                    for (int r = 0; r < comm_.size(); r++)
                    {
                        int lsz = spl_z_.local_size(r);
                        int offs = spl_z_.global_offset(r);
                    
                        std::memcpy(&fft_buffer_aux_[offs * num_zcol_local + i * lsz], &fftw_buffer_z_[tid][offs], 
                                    lsz * sizeof(double_complex));
                    }
                    break;

                }
                case -1:
                {
                    /* collect full z-column */ 
                    for (int r = 0; r < comm_.size(); r++)
                    {
                        int lsz = spl_z_.local_size(r);
                        int offs = spl_z_.global_offset(r);
                    
                        std::memcpy(&fftw_buffer_z_[tid][offs], &fft_buffer_aux_[offs * num_zcol_local + i * lsz],
                                    lsz * sizeof(double_complex));
                    }

                    /* perform local FFT transform of a column */
                    fftw_execute(plan_forward_z_[tid]);

                    /* save z column of PW coefficients*/
                    for (size_t j = 0; j < gvec__.z_column(icol).z.size(); j++)
                    {
                        int z = gvec__.z_column(icol).z[j];
                        if (z < 0) z += grid_.size(2);

                        data__[offset + j] = fftw_buffer_z_[tid][z] * norm;
                    }
                    break;

                }
                default:
                {
                    TERMINATE("wrong direction");
                }
            }
        }
    }

    /* scatter z-columns between slabs of FFT buffer */
    if (direction == 1)
    {
        block_data_descriptor send(comm_.size());
        block_data_descriptor recv(comm_.size());
        for (int r = 0; r < comm_.size(); r++)
        {
            send.counts[r] = spl_z_.local_size(r) * gvec__.zcol_fft_distr().counts[rank];
            recv.counts[r] = spl_z_.local_size(rank) * gvec__.zcol_fft_distr().counts[r];
        }
        send.calc_offsets();
        recv.calc_offsets();

        /* scatter z-columns */
        comm_.alltoall(&fft_buffer_aux_[0], &send.counts[0], &send.offsets[0], &fft_buffer_[0], &recv.counts[0], &recv.offsets[0]);
        /* copy local fractions of z-columns to auxiliary buffer;
           now we can populate fft_buffer with the results of xy-transform */
        std::memcpy(&fft_buffer_aux_[0], &fft_buffer_[0], gvec__.z_columns().size() * local_size_z_ * sizeof(double_complex)); 
    }
}

template <int direction>
void FFT3D::transform(Gvec const& gvec__, double_complex* data__)
{
    /* reallocate auxiliary buffer if needed */
    size_t sz_max;
    if (comm_.size() > 1)
    {
        int rank = comm_.rank();
        int num_zcol_local = gvec__.zcol_fft_distr().counts[rank];
        /* we need this buffer for mpi_alltoall */
        sz_max = std::max(grid_.size(2) * num_zcol_local, local_size());
    }
    else
    {
        sz_max = grid_.size(2) * gvec__.z_columns().size();
    }
    if (sz_max > fft_buffer_aux_.size())
    {
        fft_buffer_aux_ = mdarray<double_complex, 1>(sz_max);
        #ifdef __GPU
        if (pu_ == GPU)
        {
            fft_buffer_aux_.pin_memory();
            fft_buffer_aux_.allocate_on_device();
        }
        #endif
    }

    /* single node FFT */
    if (comm_.size() == 1)
    {
        if (pu_ == CPU || !cufft3d_)
        {
            switch (direction)
            {
                case 1:
                {
                    fft_buffer_.zero();
                    if (gvec__.reduced())
                    {
                        transform_z_serial<1, true>(gvec__, data__);
                    }
                    else
                    {
                        transform_z_serial<1, false>(gvec__, data__);
                    }
                    transform_xy<1>(gvec__);
                    break;
                }
                case -1:
                {
                    transform_xy<-1>(gvec__);
                    transform_z_serial<-1, false>(gvec__, data__);
                    break;
                }
                default:
                {
                    TERMINATE("wrong direction");
                }
            }
        }
        #ifdef __GPU
        if (pu_ == GPU && cufft3d_)
        {
            switch (direction)
            {
                case 1:
                {
                    cufft_backward_transform(cufft_plan_, fft_buffer_.at<GPU>());
                    break;
                }
                case -1:
                {
                    cufft_forward_transform(cufft_plan_, fft_buffer_.at<GPU>());
                    break;
                }
            }
        }
        #endif
    }
    else
    {
        switch (direction)
        {
            case 1:
            {
                transform_z_parallel<1>(gvec__, data__);
                transform_xy<1>(gvec__);
                break;
            }
            case -1:
            {
                transform_xy<-1>(gvec__);
                transform_z_parallel<-1>(gvec__, data__);
                break;
            }
            default:
            {
                TERMINATE("wrong direction");
            }
        }   
    }
}

template void FFT3D::transform<1>(Gvec const& gvec__, double_complex* data__);

template void FFT3D::transform<-1>(Gvec const& gvec__, double_complex* data__);
        
};

// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

namespace sirius {

template <int direction, bool use_reduction>
void FFT3D::transform_z_parallel(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__, mdarray<double_complex, 1>& fft_buffer_aux__)
{
    comm_.barrier();
    TIMER("sirius::FFT3D::transform_z_parallel");

    auto& gvec = gvec_fft_distr__.gvec();

    int rank = comm_.rank();
    int num_zcol_local = gvec_fft_distr__.zcol_fft_distr().counts[rank];
    double norm = 1.0 / size();

    if (direction == -1)
    {
        runtime::Timer t("sirius::FFT3D::transform_z_parallel|comm");

        block_data_descriptor send(comm_.size());
        block_data_descriptor recv(comm_.size());
        for (int r = 0; r < comm_.size(); r++)
        {
            send.counts[r] = spl_z_.local_size(rank) * gvec_fft_distr__.zcol_fft_distr().counts[r];
            recv.counts[r] = spl_z_.local_size(r)    * gvec_fft_distr__.zcol_fft_distr().counts[rank];
        }
        send.calc_offsets();
        recv.calc_offsets();

        std::copy(&fft_buffer_aux__[0], &fft_buffer_aux__[0] + gvec.num_z_cols() * local_size_z_,
                  &fft_buffer_[0]);

        comm_.alltoall(&fft_buffer_[0], &send.counts[0], &send.offsets[0], &fft_buffer_aux__[0], &recv.counts[0], &recv.offsets[0]);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < num_zcol_local; i++)
        {
            /* global index of column */
            int icol = gvec_fft_distr__.zcol_fft_distr().offsets[rank] + i;
            int data_offset = gvec_fft_distr__.zcol_offset(i);

            switch (direction)
            {
                case 1:
                {
                    /* clear z buffer */
                    std::fill(fftw_buffer_z_[tid], fftw_buffer_z_[tid] + grid_.size(2), 0);
                    /* load z column  of PW coefficients into buffer */
                    for (size_t j = 0; j < gvec.z_column(icol).z.size(); j++)
                    {
                        int z = (gvec.z_column(icol).z[j] + grid_.size(2)) % grid_.size(2);
                        fftw_buffer_z_[tid][z] = data__[data_offset + j];
                    }

                    /* column with {x,y} = {0,0} has only non-negative z components */
                    if (use_reduction && !icol)
                    {
                        /* load remaining part of {0,0,z} column */
                        for (size_t j = 0; j < gvec.z_column(icol).z.size(); j++)
                        {
                            int z = (-gvec.z_column(icol).z[j] + grid_.size(2)) % grid_.size(2);
                            fftw_buffer_z_[tid][z] = std::conj(data__[data_offset + j]);
                        }
                    }

                    /* perform local FFT transform of a column */
                    fftw_execute(plan_backward_z_[tid]);

                    /* redistribute z-column for a forthcoming all-to-all */ 
                    for (int r = 0; r < comm_.size(); r++)
                    {
                        int lsz = spl_z_.local_size(r);
                        int offs = spl_z_.global_offset(r);

                        std::copy(&fftw_buffer_z_[tid][offs], &fftw_buffer_z_[tid][offs + lsz], 
                                  &fft_buffer_aux__[offs * num_zcol_local + i * lsz]);
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
                        std::copy(&fft_buffer_aux__[offs * num_zcol_local + i * lsz],
                                  &fft_buffer_aux__[offs * num_zcol_local + i * lsz + lsz],
                                  &fftw_buffer_z_[tid][offs]);
                    }

                    /* perform local FFT transform of a column */
                    fftw_execute(plan_forward_z_[tid]);

                    /* save z column of PW coefficients*/
                    for (size_t j = 0; j < gvec.z_column(icol).z.size(); j++)
                    {
                        int z = (gvec.z_column(icol).z[j] + grid_.size(2)) % grid_.size(2);
                        data__[data_offset + j] = fftw_buffer_z_[tid][z] * norm;
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
        runtime::Timer t("sirius::FFT3D::transform_z_parallel|comm");

        block_data_descriptor send(comm_.size());
        block_data_descriptor recv(comm_.size());
        for (int r = 0; r < comm_.size(); r++)
        {
            send.counts[r] = spl_z_.local_size(r)    * gvec_fft_distr__.zcol_fft_distr().counts[rank];
            recv.counts[r] = spl_z_.local_size(rank) * gvec_fft_distr__.zcol_fft_distr().counts[r];
        }
        send.calc_offsets();
        recv.calc_offsets();

        /* scatter z-columns */
        comm_.alltoall(&fft_buffer_aux__[0], &send.counts[0], &send.offsets[0], &fft_buffer_[0], &recv.counts[0], &recv.offsets[0]);

        /* copy local fractions of z-columns into auxiliary buffer */
        std::copy(&fft_buffer_[0], &fft_buffer_[0] + gvec.num_z_cols() * local_size_z_,
                  &fft_buffer_aux__[0]);
    }
    comm_.barrier();
}

template <int direction>
void FFT3D::transform(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__)
{
    TIMER("sirius::FFT3D::transform");

    if (!prepared_) {
        TERMINATE("FFT3D is not ready");
    }

    /* reallocate auxiliary buffer if needed */
    size_t sz_max;
    if (comm_.size() > 1) {
        int rank = comm_.rank();
        int num_zcol_local = gvec_fft_distr__.zcol_fft_distr().counts[rank];
        /* we need this buffer for mpi_alltoall */
        sz_max = std::max(grid_.size(2) * num_zcol_local, local_size());
    } else {
        sz_max = grid_.size(2) * gvec_fft_distr__.gvec().num_z_cols();
    }
    if (sz_max > fft_buffer_aux1_.size()) {
        fft_buffer_aux1_ = mdarray<double_complex, 1>(sz_max, "fft_buffer_aux1_");
        #ifdef __GPU
        if (pu_ == GPU) {
            fft_buffer_aux1_.pin_memory();
            fft_buffer_aux1_.allocate_on_device();
        }
        #endif
    }

    /* single node FFT */
    if (comm_.size() == 1) {
        /* special case when FFT is fully on GPU */
        if (full_gpu_impl_) {
            #ifdef __GPU
            if (gvec_fft_distr__.gvec().reduced()) {
                TERMINATE_NOT_IMPLEMENTED
            } else {
               transform_3d_serial_gpu<direction, false>(gvec_fft_distr__, data__);
            }
            #else
            TERMINATE_NO_GPU
            #endif
        } else {
            switch (direction) {
                case 1: {
                    if (gvec_fft_distr__.gvec().reduced()) {
                        transform_z_serial<1, true>(gvec_fft_distr__, data__, fft_buffer_aux1_);
                        transform_xy<1, true>(gvec_fft_distr__, fft_buffer_aux1_);
                    } else {
                        transform_z_serial<1, false>(gvec_fft_distr__, data__, fft_buffer_aux1_);
                        transform_xy<1, false>(gvec_fft_distr__, fft_buffer_aux1_);
                    }
                    break;
                }
                case -1: {
                    transform_xy<-1, false>(gvec_fft_distr__, fft_buffer_aux1_);
                    transform_z_serial<-1, false>(gvec_fft_distr__, data__, fft_buffer_aux1_);
                    break;
                }
                default: {
                    TERMINATE("wrong direction");
                }
            }
        }
    } else {
        switch (direction) {
            case 1: {
                if (gvec_fft_distr__.gvec().reduced()) {
                    transform_z_parallel<1, true>(gvec_fft_distr__, data__, fft_buffer_aux1_);
                    transform_xy<1, true>(gvec_fft_distr__, fft_buffer_aux1_);
                } else {
                    transform_z_parallel<1, false>(gvec_fft_distr__, data__, fft_buffer_aux1_);
                    transform_xy<1, false>(gvec_fft_distr__, fft_buffer_aux1_);
                }
                break;
            }
            case -1: {
                transform_xy<-1, false>(gvec_fft_distr__, fft_buffer_aux1_);
                transform_z_parallel<-1, false>(gvec_fft_distr__, data__, fft_buffer_aux1_);
                break;
            }
            default: {
                TERMINATE("wrong direction");
            }
        }   
    }
}

template <int direction>
void FFT3D::transform(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data1__, double_complex* data2__)
{
    TIMER("sirius::FFT3D::transform");

    if (!prepared_) {
        TERMINATE("FFT3D is not ready");
    }

    if (!gvec_fft_distr__.gvec().reduced()) {
        TERMINATE("reduced set of G-vectors is required");
    }

    /* reallocate auxiliary buffers if needed */
    size_t sz_max;
    if (comm_.size() > 1) {
        int rank = comm_.rank();
        int num_zcol_local = gvec_fft_distr__.zcol_fft_distr().counts[rank];
        /* we need this buffer for mpi_alltoall */
        sz_max = std::max(grid_.size(2) * num_zcol_local, local_size());
    } else {
        sz_max = grid_.size(2) * gvec_fft_distr__.gvec().num_z_cols();
    }
    
    if (sz_max > fft_buffer_aux1_.size()) {
        fft_buffer_aux1_ = mdarray<double_complex, 1>(sz_max, "fft_buffer_aux1_");
        #ifdef __GPU
        if (pu_ == GPU) {
            fft_buffer_aux1_.pin_memory();
            fft_buffer_aux1_.allocate_on_device();
        }
        #endif
    }
    if (sz_max > fft_buffer_aux2_.size()) {
        fft_buffer_aux2_ = mdarray<double_complex, 1>(sz_max, "fft_buffer_aux2_");
        #ifdef __GPU
        if (pu_ == GPU) {
            fft_buffer_aux2_.pin_memory();
            fft_buffer_aux2_.allocate_on_device();
        }
        #endif
    }

    /* single node FFT */
    if (comm_.size() == 1)
    {
        switch (direction)
        {
            case 1:
            {
                transform_z_serial<1, true>(gvec_fft_distr__, data1__, fft_buffer_aux1_);
                transform_z_serial<1, true>(gvec_fft_distr__, data2__, fft_buffer_aux2_);
                transform_xy<1>(gvec_fft_distr__, fft_buffer_aux1_, fft_buffer_aux2_);
                break;
            }
            case -1:
            {
                transform_xy<-1>(gvec_fft_distr__, fft_buffer_aux1_, fft_buffer_aux2_);
                transform_z_serial<-1, false>(gvec_fft_distr__, data1__, fft_buffer_aux1_);
                transform_z_serial<-1, false>(gvec_fft_distr__, data2__, fft_buffer_aux2_);
                break;
            }
            default:
            {
                TERMINATE("wrong direction");
            }
        }
    }
    else
    {
        switch (direction)
        {
            case 1:
            {
                transform_z_parallel<1, true>(gvec_fft_distr__, data1__, fft_buffer_aux1_);
                transform_z_parallel<1, true>(gvec_fft_distr__, data2__, fft_buffer_aux2_);
                transform_xy<1>(gvec_fft_distr__, fft_buffer_aux1_, fft_buffer_aux2_);
                break;
            }
            case -1:
            {
                transform_xy<-1>(gvec_fft_distr__, fft_buffer_aux1_, fft_buffer_aux2_);
                transform_z_parallel<-1, false>(gvec_fft_distr__, data1__, fft_buffer_aux1_);
                transform_z_parallel<-1, false>(gvec_fft_distr__, data2__, fft_buffer_aux2_);
                break;
            }
            default:
            {
                TERMINATE("wrong direction");
            }
        }   
    }
}

template void FFT3D::transform<1>(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__);
template void FFT3D::transform<-1>(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__);

template void FFT3D::transform<1>(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data1__, double_complex* data2__);
template void FFT3D::transform<-1>(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data1__, double_complex* data2__);
        
};

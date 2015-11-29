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

    grid_ = FFT_grid(dims__);

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

    fftw_plan_with_nthreads(1);

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

};

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
      ,allocated_on_device_(false)
      #endif
      
{
    PROFILE();

    fft_grid_ = FFT_grid(dims__);

    size_t alloc_local_size = 0;
    if (comm_.size() > 1)
    {
        #ifdef __FFTW_MPI
        ptrdiff_t sz, offs;
        alloc_local_size = fftw_mpi_local_size_3d(fft_grid_.size(2), fft_grid_.size(1), fft_grid_.size(0), comm__.mpi_comm(), &sz, &offs);

        local_size_z_ = (int)sz;
        offset_z_ = (int)offs;
        #else
        TERMINATE("not compiled with MPI support");
        #endif
    }
    else
    {
        alloc_local_size = size();
        local_size_z_ = fft_grid_.size(2);
        offset_z_ = 0;
    }

    /* split z-direction */
    spl_z_ = splindex<block>(fft_grid_.size(2), comm_.size(), comm_.rank());
    assert((int)spl_z_.local_size() == local_size_z_);
    
    if (comm_.size() > 1)
    {
        /* we need this buffer for mpi_alltoall */
        int sz_max = std::max(fft_grid_.size(2) * splindex_base<int>::block_size(fft_grid_.size(0) * fft_grid_.size(1), comm_.size()),
                              local_size());
        fft_buffer_aux_ = mdarray<double_complex, 1>(sz_max);
    }
    
    /* allocate main buffer */
    fftw_buffer_ = (double_complex*)fftw_malloc(alloc_local_size * sizeof(double_complex));
    
    /* allocate 1d and 2d buffers */
    for (int i = 0; i < num_fft_workers_; i++)
    {
        fftw_buffer_z_.push_back((double_complex*)fftw_malloc(fft_grid_.size(2) * sizeof(double_complex)));
        fftw_buffer_xy_.push_back((double_complex*)fftw_malloc(fft_grid_.size(0) * fft_grid_.size(1) * sizeof(double_complex)));
    }

    //fftw_plan_with_nthreads(num_fft_workers_);

    //if (comm_.size() > 1)
    //{
    //    #ifdef __FFTW_MPI
    //    plan_backward_ = fftw_mpi_plan_dft_3d(size(2), size(1), size(0), 
    //                                          (fftw_complex*)fftw_buffer_[i], 
    //                                          (fftw_complex*)fftw_buffer_[i],
    //                                          comm_.mpi_comm(), FFTW_BACKWARD, FFTW_ESTIMATE);

    //    plan_forward_ = fftw_mpi_plan_dft_3d(size(2), size(1), size(0), 
    //                                         (fftw_complex*)fftw_buffer_[i], 
    //                                         (fftw_complex*)fftw_buffer_[i],
    //                                         comm_.mpi_comm(), FFTW_FORWARD, FFTW_ESTIMATE);

    //    #else
    //    TERMINATE("not compiled with MPI support");
    //    #endif
    //}
    //else
    //{
    //    plan_backward_ = fftw_plan_dft_3d(size(2), size(1), size(0), 
    //                                      (fftw_complex*)fftw_buffer_[i], 
    //                                      (fftw_complex*)fftw_buffer_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
    //    
    //    plan_forward_ = fftw_plan_dft_3d(size(2), size(1), size(0), 
    //                                     (fftw_complex*)fftw_buffer_[i], 
    //                                     (fftw_complex*)fftw_buffer_[i], FFTW_FORWARD, FFTW_ESTIMATE);
    //}

    fftw_plan_with_nthreads(1);

    plan_forward_z_   = std::vector<fftw_plan>(num_fft_workers_);
    plan_forward_xy_  = std::vector<fftw_plan>(num_fft_workers_);
    plan_backward_z_  = std::vector<fftw_plan>(num_fft_workers_);
    plan_backward_xy_ = std::vector<fftw_plan>(num_fft_workers_);

    for (int i = 0; i < num_fft_workers_; i++)
    {
        plan_forward_z_[i] = fftw_plan_dft_1d(fft_grid_.size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                              (fftw_complex*)fftw_buffer_z_[i], FFTW_FORWARD, FFTW_ESTIMATE);

        plan_backward_z_[i] = fftw_plan_dft_1d(fft_grid_.size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                               (fftw_complex*)fftw_buffer_z_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
        
        plan_forward_xy_[i] = fftw_plan_dft_2d(fft_grid_.size(1), fft_grid_.size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                               (fftw_complex*)fftw_buffer_xy_[i], FFTW_FORWARD, FFTW_ESTIMATE);

        plan_backward_xy_[i] = fftw_plan_dft_2d(fft_grid_.size(1), fft_grid_.size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                                (fftw_complex*)fftw_buffer_xy_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    
    #ifdef __GPU
    if (pu_ == GPU)
    {
        int auto_alloc = 0;
        if (comm_.size() == 1 && cufft3d_)
        {
            cufft_nbatch_ = 1;

            int dims[] = {fft_grid_.size(2), fft_grid_.size(1), fft_grid_.size(0)};
            cufft_create_plan_handle(&cufft_plan_);
            cufft_create_batch_plan(cufft_plan_, 3, dims, dims, 1, 1, cufft_nbatch_, auto_alloc);
        }
        else
        {
            cufft_nbatch_ = 1; //local_size_z_;

            int dim_xy[] = {fft_grid_.size(1), fft_grid_.size(0)};
            int embed_xy[] = {fft_grid_.size(1), fft_grid_.size(0)};

            cufft_create_plan_handle(&cufft_plan_xy_);
            cufft_create_batch_plan(cufft_plan_xy_, 2, dim_xy, embed_xy, 1, fft_grid_.size(0) * fft_grid_.size(1), cufft_nbatch_, auto_alloc);
            cufft_set_stream(cufft_plan_xy_, 0);
        }
    }
    #endif
}

FFT3D::~FFT3D()
{
    fftw_free(fftw_buffer_);

    for (int i = 0; i < num_fft_workers_; i++)
    {
        fftw_free(fftw_buffer_z_[i]);
        fftw_free(fftw_buffer_xy_[i]);
    }

    //fftw_destroy_plan(plan_backward_);
    //fftw_destroy_plan(plan_forward_);

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

//void FFT3D::backward_custom(std::vector< std::pair<int, int> > const& z_sticks_coord__)
//{
//    splindex<block> spl_n((int)z_sticks_coord__.size(), comm_.size(), comm_.rank());
//
//    std::vector<int> sendcounts(comm_.size());
//    std::vector<int> sdispls(comm_.size());
//    std::vector<int> recvcounts(comm_.size());
//    std::vector<int> rdispls(comm_.size());
//
//    for (int rank = 0; rank < comm_.size(); rank++)
//    {
//        sendcounts[rank] = (int)spl_z_.local_size() * (int)spl_n.local_size(rank);
//        sdispls[rank]    = (int)spl_z_.local_size() * (int)spl_n.global_offset(rank);
//
//        recvcounts[rank] = (int)spl_n.local_size() * (int)spl_z_.local_size(rank);
//        rdispls[rank]    = (int)spl_n.local_size() * (int)spl_z_.global_offset(rank);
//    }
//
//    if (comm_.size() > 1)
//    {
//        /* transform f(Gx,Gy,Gz) -> f(Gx,Gy,z) */ 
//        transform_z_parallel<1>(sendcounts, sdispls, recvcounts, rdispls, (int)spl_n.local_size());
//        memcpy(&fft_buffer_aux_(0), fftw_buffer_, local_size() * sizeof(double_complex));
//        /* transform f(Gx,Gy,z) -> f(x,y,z) */
//        transform_xy_parallel<1>(z_sticks_coord__);
//    }
//    else
//    {
//        /* transform f(Gx,Gy,Gz) -> f(Gx,Gy,z) */ 
//        transform_z_serial<1>(z_sticks_coord__);
//        /* transform f(Gx,Gy,z) -> f(x,y,z) */
//        transform_xy_serial<1>();
//    }
//}
//
//void FFT3D::backward_custom(std::vector<z_column_descriptor> const& z_cols__, double_complex* data__)
//{
//    std::memset(fftw_buffer_, 0, size() * sizeof(double_complex));
//    transform_z_serial<1>(z_cols__, data__);
//    transform_xy_serial<1>();
//}
//
//void FFT3D::forward_custom(std::vector< std::pair<int, int> > const& z_sticks_coord__)
//{
//    splindex<block> spl_n((int)z_sticks_coord__.size(), comm_.size(), comm_.rank());
//
//    std::vector<int> sendcounts(comm_.size());
//    std::vector<int> sdispls(comm_.size());
//    std::vector<int> recvcounts(comm_.size());
//    std::vector<int> rdispls(comm_.size());
//
//    for (int rank = 0; rank < comm_.size(); rank++)
//    {
//        sendcounts[rank] = (int)spl_z_.local_size() * (int)spl_n.local_size(rank);
//        sdispls[rank]    = (int)spl_z_.local_size() * (int)spl_n.global_offset(rank);
//
//        recvcounts[rank] = (int)spl_n.local_size() * (int)spl_z_.local_size(rank);
//        rdispls[rank]    = (int)spl_n.local_size() * (int)spl_z_.global_offset(rank);
//    }
//
//    if (comm_.size() > 1)
//    {
//        /* transform f(x,y,z) -> f(Gx,Gy,z) */
//        transform_xy_parallel<-1>(z_sticks_coord__);
//        memcpy(fftw_buffer_, &fft_buffer_aux_(0), local_size_z_ * z_sticks_coord__.size() * sizeof(double_complex));
//        /* transform f(Gx,Gy,z) -> f(Gx,Gy,Gz) */
//        transform_z_parallel<-1>(sendcounts, sdispls, recvcounts, rdispls, (int)spl_n.local_size());
//    }
//    else
//    {
//        /* transform f(x,y,z) -> f(Gx,Gy,z) */
//        transform_xy_serial<-1>();
//        /* transform f(Gx,Gy,z) -> f(Gx,Gy,Gz) */
//        transform_z_serial<-1>(z_sticks_coord__);
//    }
//
//    double norm = 1.0 / size();
//    #pragma omp parallel for schedule(static) num_threads(num_fft_workers_)
//    for (int i = 0; i < local_size(); i++) fftw_buffer_[i] *= norm;
//}

};

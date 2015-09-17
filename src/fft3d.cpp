// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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
      pu_(pu__)
{
    PROFILE();

    for (int i = 0; i < 3; i++)
    {
        grid_size_[i] = find_grid_size(dims__[i]);
        
        grid_limits_[i].second = grid_size_[i] / 2;
        grid_limits_[i].first = grid_limits_[i].second - grid_size_[i] + 1;
    }

    size_t alloc_local_size = 0;
    if (comm_.size() > 1)
    {
        #ifdef __FFTW_MPI
        ptrdiff_t sz, offs;
        alloc_local_size = fftw_mpi_local_size_3d(size(2), size(1), size(0), comm__.mpi_comm(), &sz, &offs);

        local_size_z_ = (int)sz;
        offset_z_ = (int)offs;
        #else
        TERMINATE("not compiled with MPI support");
        #endif
    }
    else
    {
        alloc_local_size = size();
        local_size_z_ = size(2);
        offset_z_ = 0;
    }

    /* split z-direction */
    spl_z_ = splindex<block>(size(2), comm_.size(), comm_.rank());
    assert((int)spl_z_.local_size() == local_size_z_);
    
    if (comm_.size() > 1)
    {
        /* we need this buffer for mpi_alltoall */
        int sz_max = std::max(int(size(2) * splindex_base::block_size(size(0) * size(1), comm_.size())), local_size());
        fft_buffer_aux_ = mdarray<double_complex, 1>(sz_max);
    }
    
    /* allocate main buffer */
    fftw_buffer_ = (double_complex*)fftw_malloc(alloc_local_size * sizeof(double_complex));
    
    /* allocate 1d and 2d buffers */
    for (int i = 0; i < num_fft_workers_; i++)
    {
        fftw_buffer_z_.push_back((double_complex*)fftw_malloc(size(2) * sizeof(double_complex)));
        fftw_buffer_xy_.push_back((double_complex*)fftw_malloc(size(0) * size(1) * sizeof(double_complex)));
    }

    #ifdef __GPU
    if (pu_ == GPU)
    {
        buf_z_ = std::vector< mdarray<double_complex, 1> >(num_fft_workers_);
        buf_xy_ = std::vector< mdarray<double_complex, 1> >(num_fft_workers_);
        for (int i = 0; i < num_fft_workers_; i++)
        {
            buf_z_[i] = mdarray<double_complex, 1>(fftw_buffer_z_[i], size(2));
            buf_z_[i].allocate_on_device();
            buf_xy_[i] = mdarray<double_complex, 1>(fftw_buffer_xy_[i], size(0) * size(1));
            buf_xy_[i].allocate_on_device();
        }
    }
    #endif

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
        plan_forward_z_[i] = fftw_plan_dft_1d(size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                              (fftw_complex*)fftw_buffer_z_[i], FFTW_FORWARD, FFTW_ESTIMATE);

        plan_backward_z_[i] = fftw_plan_dft_1d(size(2), (fftw_complex*)fftw_buffer_z_[i], 
                                               (fftw_complex*)fftw_buffer_z_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
        
        plan_forward_xy_[i] = fftw_plan_dft_2d(size(1), size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                               (fftw_complex*)fftw_buffer_xy_[i], FFTW_FORWARD, FFTW_ESTIMATE);

        plan_backward_xy_[i] = fftw_plan_dft_2d(size(1), size(0), (fftw_complex*)fftw_buffer_xy_[i], 
                                                (fftw_complex*)fftw_buffer_xy_[i], FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    
    #ifdef __GPU
    if (pu_ == GPU)
    {
        plan_z_  = std::vector<cufftHandle>(num_fft_workers_);
        plan_xy_ = std::vector<cufftHandle>(num_fft_workers_);

        int dim_z[] = {size(2)};
        int dim_xy[] = {size(1), size(0)};

        for (int i = 0; i < num_fft_workers_; i++)
        {
            cufft_create_plan_handle(&plan_z_[i]);
            cufft_create_batch_plan(plan_z_[i], 1, dim_z, 1, 1);
            cufft_set_stream(plan_z_[i], i);

            cufft_create_plan_handle(&plan_xy_[i]);
            cufft_create_batch_plan(plan_xy_[i], 2, dim_xy, 1, 1);
            cufft_set_stream(plan_xy_[i], i);
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
        for (int i = 0; i < num_fft_workers_; i++)
        {
            cufft_destroy_plan_handle(plan_z_[i]); 
            cufft_destroy_plan_handle(plan_xy_[i]); 
        }
    }
    #endif
}

void FFT3D::backward_custom(std::vector< std::pair<int, int> > const& z_sticks_coord__)
{
    int size_xy = size(0) * size(1);

    splindex<block> spl_n(z_sticks_coord__.size(), comm_.size(), comm_.rank());

    std::vector<int> sendcounts(comm_.size());
    std::vector<int> sdispls(comm_.size());
    std::vector<int> recvcounts(comm_.size());
    std::vector<int> rdispls(comm_.size());

    for (int rank = 0; rank < comm_.size(); rank++)
    {
        sendcounts[rank] = (int)spl_z_.local_size() * (int)spl_n.local_size(rank);
        sdispls[rank]    = (int)spl_z_.local_size() * (int)spl_n.global_offset(rank);

        recvcounts[rank] = (int)spl_n.local_size() * (int)spl_z_.local_size(rank);
        rdispls[rank]    = (int)spl_n.local_size() * (int)spl_z_.global_offset(rank);
    }
    
    if (comm_.size() > 1)
    {
        STOP();

        //Timer t1("fft|comm_inner");
        //comm_.alltoall(fftw_buffer_[thread_id__], &sendcounts[0], &sdispls[0], 
        //               &fft_buffer_aux_(0, thread_id__), &recvcounts[0], &rdispls[0]);
        //t1.stop();

        //#pragma omp parallel num_threads(num_fft_workers_)
        //{
        //    int tid = omp_get_thread_num();

        //    double_complex* src = (comm_.size() == 1) ? fftw_buffer_[thread_id__] : &z_sticks_(0, thread_id__);
        //    double_complex* dest = &z_sticks_(0, thread_id__);
        //    
        //    #pragma omp for
        //    for (int i = 0; i < (int)spl_n.local_size(); i++)
        //    {
        //        /* collect z-column from temporary data_slice_ buffer */
        //        for (int rank = 0; rank < comm_.size(); rank++)
        //        {
        //            int lsz = (int)spl_z_.local_size(rank);

        //            memcpy(&fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
        //                   &src[spl_z_.global_offset(rank) * spl_n.local_size() + i * lsz],
        //                   lsz * sizeof(double_complex));
        //        }

        //        fftw_execute(plan_backward_z_[tid]);

        //        for (int rank = 0; rank < comm_.size(); rank++)
        //        {
        //            int lsz = (int)spl_z_.local_size(rank);

        //            memcpy(&dest[spl_z_.global_offset(rank) * spl_n.local_size() + i * lsz],
        //                   &fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
        //                   lsz * sizeof(double_complex));
        //        }
        //    }
        //}

        //t1.start();
        //comm_.alltoall(&z_sticks_(0, thread_id__), &recvcounts[0], &rdispls[0], 
        //               fftw_buffer_[thread_id__], &sendcounts[0], &sdispls[0]);
    }
    else
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < (int)z_sticks_coord__.size(); i++)
            {
                int x = z_sticks_coord__[i].first;
                int y = z_sticks_coord__[i].second;
                for (int z = 0; z < size(2); z++)
                {
                    fftw_buffer_z_[tid][z] = fftw_buffer_[x + y * size(0) + z * size_xy];
                }
                //switch (pu_)
                //{
                //    case CPU:
                //    {
                        fftw_execute(plan_backward_z_[tid]);
                //        break;
                //    }
                //    case GPU:
                //    {
                //        #ifdef __GPU
                //        buf_z_[tid].async_copy_to_device(tid);
                //        cufft_backward_transform(plan_z_[tid], buf_z_[tid].at<GPU>());
                //        buf_z_[tid].async_copy_to_host(tid);
                //        cuda_stream_synchronize(tid);
                //        #endif
                //        break;
                //    }
                //}
                for (int z = 0; z < size(2); z++)
                {
                    fftw_buffer_[x + y * size(0) + z * size_xy] = fftw_buffer_z_[tid][z];
                }
            }
        }
    }

    if (comm_.size() > 1)
    {
        STOP();
        //#pragma omp parallel num_threads(num_fft_workers_)
        //{
        //    int tid = omp_get_thread_num();
        //    #pragma omp for
        //    for (int iz = 0; iz < local_size_z_; iz++)
        //    {
        //        memset(fftw_buffer_xy_[tid], 0, sizeof(double_complex) * size_xy);
        //        for (int n = 0; n < (int)z_sticks_coord__.size(); n++)
        //        {
        //            int x = z_sticks_coord__[n].first;
        //            int y = z_sticks_coord__[n].second;

        //            fftw_buffer_xy_[tid][x + y * size(0)] = data_slab_(iz, n);
        //        }
        //        fftw_execute(plan_backward_xy_[tid]);
        //        memcpy(&fftw_buffer_[0][iz * size_xy], fftw_buffer_xy_[tid], sizeof(fftw_complex) * size_xy);
        //    }
        //}
    }
    else
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int z = 0; z < size(2); z++)
            {
                switch (pu_)
                {
                    case CPU:
                    {
                        memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[z * size_xy], sizeof(double_complex) * size_xy);
                        fftw_execute(plan_backward_xy_[tid]);
                        memcpy(&fftw_buffer_[z * size_xy], fftw_buffer_xy_[tid], sizeof(double_complex) * size_xy);
                        break;
                    }
                    case GPU:
                    {
                        #ifdef __GPU
                        mdarray<double_complex, 1>(&fftw_buffer_[z * size_xy], buf_xy_[tid].at<GPU>(), size(0) * size(1)).async_copy_to_device(tid);
                        cufft_backward_transform(plan_xy_[tid], buf_xy_[tid].at<GPU>());
                        mdarray<double_complex, 1>(&fftw_buffer_[z * size_xy], buf_xy_[tid].at<GPU>(), size(0) * size(1)).async_copy_to_host(tid);
                        #endif
                        break;
                    }
                }
            }
            cuda_stream_synchronize(tid);
        }
    }

    //if (Platform::rank() == 0)
    //{
    //    std::cout << "----------------------------------------------------------------" << std::endl;
    //    std::cout << "thread_id  | fft_z (num, throughput) | fft_xy (num, throughput) " << std::endl;
    //    std::cout << "----------------------------------------------------------------" << std::endl;
    //    for (int i = 0; i < num_fft_workers_; i++)
    //    {
    //        double d1 = (z_counts[i] == 0) ? 0 : z_counts[i] / z_times[i];
    //        double d2 = (xy_counts[i] == 0) ? 0 : xy_counts[i] / xy_times[i];
    //        printf("   %2i      | %5i  %10.4e       |  %5i   %10.4e \n", i, z_counts[i], d1, xy_counts[i], d2);
    //    }
    //    std::cout << "----------------------------------------------------------------" << std::endl;
    //}
}
        
void FFT3D::forward_custom(std::vector< std::pair<int, int> > const& z_sticks_coord__)
{
    int size_xy = size(0) * size(1);

    splindex<block> spl_n(z_sticks_coord__.size(), comm_.size(), comm_.rank());

    std::vector<int> sendcounts(comm_.size());
    std::vector<int> sdispls(comm_.size());
    std::vector<int> recvcounts(comm_.size());
    std::vector<int> rdispls(comm_.size());

    for (int rank = 0; rank < comm_.size(); rank++)
    {
        sendcounts[rank] = (int)spl_z_.local_size() * (int)spl_n.local_size(rank);
        sdispls[rank]    = (int)spl_z_.local_size() * (int)spl_n.global_offset(rank);

        recvcounts[rank] = (int)spl_n.local_size() * (int)spl_z_.local_size(rank);
        rdispls[rank]    = (int)spl_n.local_size() * (int)spl_z_.global_offset(rank);
    }
    
    if (comm_.size() > 1)
    {
        STOP();
        //#pragma omp parallel num_threads(num_fft_workers_)
        //{
        //    int tid = omp_get_thread_num();
        //    
        //    #pragma omp for
        //    for (int iz = 0; iz < local_size_z_; iz++)
        //    {
        //        memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[0][iz * size_xy], sizeof(fftw_complex) * size_xy);
        //        fftw_execute(plan_forward_xy_[tid]);
        //        for (int n = 0; n < num_xy_packed__; n++)
        //        {
        //            int x = xy_packed_idx__(0, n);
        //            int y = xy_packed_idx__(1, n);
        //            data_slab_(iz, n) = fftw_buffer_xy_[tid][x + y * size(0)];
        //        }
        //    }
        //}
    }
    else
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();
            
            #pragma omp for
            for (int z = 0; z < local_size_z_; z++)
            {
                switch (pu_)
                {
                    case CPU:
                    {
                        memcpy(fftw_buffer_xy_[tid], &fftw_buffer_[z * size_xy], sizeof(double_complex) * size_xy);
                        fftw_execute(plan_forward_xy_[tid]);
                        memcpy(&fftw_buffer_[z * size_xy], fftw_buffer_xy_[tid], sizeof(double_complex) * size_xy);
                        break;
                    }
                    case GPU:
                    {
                        #ifdef __GPU
                        mdarray<double_complex, 1>(&fftw_buffer_[z * size_xy], buf_xy_[tid].at<GPU>(), size(0) * size(1)).async_copy_to_device(tid);
                        cufft_forward_transform(plan_xy_[tid], buf_xy_[tid].at<GPU>());
                        mdarray<double_complex, 1>(&fftw_buffer_[z * size_xy], buf_xy_[tid].at<GPU>(), size(0) * size(1)).async_copy_to_host(tid);
                        #endif
                        break;
                    }
                }
            }
            cuda_stream_synchronize(tid);
        }
    }

    if (comm_.size() > 1)
    {
        STOP();
        //Timer t2("fft|comm_inner");
        //comm_.alltoall(data_slab_.at<CPU>(), &sendcounts[0], &sdispls[0], 
        //               data_slice_.at<CPU>(), &recvcounts[0], &rdispls[0]);
        //t2.stop();

        //#pragma omp parallel num_threads(num_fft_workers_)
        //{
        //    int tid = omp_get_thread_num();
        //    
        //    #pragma omp for
        //    for (int i = 0; i < (int)spl_n.local_size(); i++)
        //    {
        //        for (int rank = 0; rank < comm_.size(); rank++)
        //        {
        //            int lsz = (int)spl_z_.local_size(rank);

        //            memcpy(&fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
        //                   &data_slice_(spl_z_.global_offset(rank) * spl_n.local_size() + i * lsz),
        //                   lsz * sizeof(double_complex));
        //        }

        //        fftw_execute(plan_forward_z_[tid]);

        //        for (int rank = 0; rank < comm_.size(); rank++)
        //        {
        //            int lsz = (int)spl_z_.local_size(rank);

        //            memcpy(&data_slice_(spl_z_.global_offset(rank) * spl_n.local_size() + i * lsz),
        //                   &fftw_buffer_z_[tid][spl_z_.global_offset(rank)], 
        //                   lsz * sizeof(double_complex));
        //        }
        //    }
        //}

        //t2.start();
        //comm_.alltoall(data_slice_.at<CPU>(), &recvcounts[0], &rdispls[0], 
        //               data_slab_.at<CPU>(), &sendcounts[0], &sdispls[0]);
        //t2.stop();
    }
    else
    {
        #pragma omp parallel num_threads(num_fft_workers_)
        {
            int tid = omp_get_thread_num();

            #pragma omp for
            for (int i = 0; i < (int)z_sticks_coord__.size(); i++)
            {
                int x = z_sticks_coord__[i].first;
                int y = z_sticks_coord__[i].second;

                for (int z = 0; z < size(2); z++)
                {
                    fftw_buffer_z_[tid][z] = fftw_buffer_[x + y * size(0) + z * size_xy];
                }

                fftw_execute(plan_forward_z_[tid]);

                for (int z = 0; z < size(2); z++)
                {
                    fftw_buffer_[x + y * size(0) + z * size_xy] = fftw_buffer_z_[tid][z];
                }
            }
        }
    }

    double norm = 1.0 / size();
    #pragma omp parallel for schedule(static) num_threads(num_fft_workers_)
    for (int i = 0; i < local_size(); i++) fftw_buffer_[i] *= norm;
}

};

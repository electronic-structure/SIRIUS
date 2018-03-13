// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file fft3d.hpp
 *
 *  \brief Contains declaration and partial implementation of FFT3D class.
 */

#ifndef __FFT3D_HPP__
#define __FFT3D_HPP__

#include <fftw3.h>
#include "geometry3d.hpp"
#include "fft3d_grid.hpp"
#include "gvec.hpp"
#ifdef __GPU
#include "GPU/cufft.hpp"
#include "GPU/fft_kernels.hpp"
#endif

namespace sddk {

/// Implementation of FFT3D.
/** FFT convention:
 *  \f[
 *      f({\bf r}) = \sum_{{\bf G}} e^{i{\bf G}{\bf r}} f({\bf G})
 *  \f]
 *  is a \em backward transformation from a set of pw coefficients to a function.
 *
 *  \f[
 *      f({\bf G}) = \frac{1}{\Omega} \int e^{-i{\bf G}{\bf r}} f({\bf r}) d {\bf r} =
 *          \frac{1}{N} \sum_{{\bf r}_j} e^{-i{\bf G}{\bf r}_j} f({\bf r}_j)
 *  \f]
 *  is a \em forward transformation from a function to a set of coefficients.
 *
 *  The following cases are handeled by the FFT driver:
 *    - transformation of a single real / complex function (serial / parallel, cpu / gpu)
 *    - transformation of two real functions (serial / parallel, cpu / gpu)
 *    - input / ouput data buffer pointer (cpu / gpu). GPU input pointer works only in serial.
 *
 *  The transformation of two real functions is done as one transformation of complex function:
 *  \f[
 *    \Psi({\bf r}) = \psi_1({\bf r}) + i \psi_2({\bf r})
 *  \f]
 *  For each of the real wave functions the following condition is fulfilled:
 *  \f[
 *    \psi_{1,2}({\bf r}) = \psi_{1,2}^{*}({\bf r})
 *  \f]
 *  from which it follows that
 *  \f[
 *    \psi_{1,2}({\bf G}) = \psi_{1,2}^{*}(-{\bf G})
 *  \f]
 *  When z-direction is transformed
 *  \f[
 *    \psi_{1,2}(G_x, G_y, z) = \sum_{G_z} e^{izG_z} \psi_{1,2}(G_x, G_y, G_z)
 *  \f]
 *  it leads to the following symmetry
 *  \f[
 *   \psi_{1,2}^{*}(G_x, G_y, z) = \sum_{G_z} e^{-izG_z} \psi_{1,2}^{*}(G_x, G_y, G_z) =
 *      \sum_{G_z} e^{-izG_z} \psi_{1,2}(-G_x, -G_y, -G_z) = \psi_{1,2}(-G_x, -G_y, z)
 *  \f]
 *
 *  \todo add += operation for (-1) transform, i.e. accumulate in the output buffer.
 *  \todo GPU input ponter for parallel FFT
 */
class FFT3D
{
    protected:

        /// Communicator for the parallel FFT.
        Communicator const& comm_;

        /// Main processing unit of this FFT.
        device_t pu_;

        /// Split z-direction.
        splindex<block> spl_z_;

        /// Definition of the FFT grid.
        FFT3D_grid grid_;

        /// Local size of z-dimension of FFT buffer.
        int local_size_z_;

        /// Offset in the global z-dimension.
        int offset_z_;

        /// Main input/output buffer.
        mdarray<double_complex, 1> fft_buffer_;

        /// Auxiliary array to store z-sticks for all-to-all or GPU.
        mdarray<double_complex, 1> fft_buffer_aux1_;

        /// Auxiliary array in case of simultaneous transformation of two wave-functions.
        mdarray<double_complex, 1> fft_buffer_aux2_;

        /// Internal buffer for independent z-transforms.
        std::vector<double_complex*> fftw_buffer_z_;

        /// Internal buffer for independent {xy}-transforms.
        std::vector<double_complex*> fftw_buffer_xy_;

        std::vector<fftw_plan> plan_backward_z_;

        std::vector<fftw_plan> plan_backward_xy_;

        std::vector<fftw_plan> plan_forward_z_;

        std::vector<fftw_plan> plan_forward_xy_;

        bool is_gpu_direct_{false};

        #ifdef __GPU
        /// Handler for xy-transform cuFFT plan.
        cufftHandle cufft_plan_xy_;

        /// Handler for z-transform cuFFT plan.
        cufftHandle cufft_plan_z_;
        
        /// True if the cufft_plan_z handler was created and has to be destroyed.
        bool cufft_plan_z_created_{false};

        /// offsets  for z-buffer
        mdarray<int,1> z_offsets_;

        /// local sizes for z-buffer
        mdarray<int,1> z_sizes_;

        /// max local z size
        int max_zloc_size_{0};

        /// Work buffer, required by cuFFT.
        mdarray<char, 1> cufft_work_buf_;

        /// Mapping of the G-vectors to the FFT buffer for batched 1D transform.
        mdarray<int, 1> map_gvec_to_fft_buffer_;

        /// Mapping of the {0,0,z} G-vectors to the FFT buffer for batched 1D transform in case of reduced G-vector list.
        mdarray<int, 1> map_gvec_to_fft_buffer_x0y0_;

        int const cufft_stream_id{0};
        #endif

        /// Position of z-columns inside 2D FFT buffer.
        mdarray<int, 2> z_col_pos_;

        memory_t host_memory_type_;

        /// Maximum number of z-columns ever transformed.
        /** This is used to recreate the cuFFT plan only when the number of columns has increased */
        int zcol_count_max_{0};

        /// Defines the distribution of G-vectors between the MPI ranks of FFT communicator.
        Gvec_partition const* gvec_partition_{nullptr};

        /// Serial part of 1D transformation of columns.
        template <int direction, device_t data_ptr_type>
        void transform_z_serial(double_complex* data__,
                                mdarray<double_complex, 1>& fft_buffer_aux__) // TODO: prepare data for mpi_a2a in case of gpu_ptr
        {
            PROFILE("sddk::FFT3D::transform_z_serial");

            int num_zcol_local = gvec_partition_->zcol_count_fft();
            double norm = 1.0 / size();

            bool is_reduced = gvec_partition_->gvec().reduced();

            assert(static_cast<int>(fft_buffer_aux__.size()) >= gvec_partition_->zcol_count_fft() * grid_.size(2));

            /* input/output data buffer is on GPU */
            if (data_ptr_type == GPU) {
                sddk::timer t("sddk::FFT3D::transform_z_serial|gpu");
                #ifdef __GPU
                switch (direction) {
                    case 1: {
                        /* load all columns into FFT buffer */
                        cufft_batch_load_gpu(num_zcol_local * grid_.size(2),
                                             gvec_partition_->gvec_count_fft(),
                                             1,
                                             map_gvec_to_fft_buffer_.at<GPU>(),
                                             (cuDoubleComplex*)data__,
                                             (cuDoubleComplex*)fft_buffer_aux__.at<GPU>(),
                                             cufft_stream_id);
                        if (is_reduced && comm_.rank() == 0) {
                            cufft_load_x0y0_col_gpu(static_cast<int>(gvec_partition_->gvec().zcol(0).z.size()),
                                                    map_gvec_to_fft_buffer_x0y0_.at<GPU>(),
                                                    (cuDoubleComplex*)data__,
                                                    (cuDoubleComplex*)fft_buffer_aux__.at<GPU>(),
                                                    cufft_stream_id);
                        }
                        /* transform all columns */
                        cufft::backward_transform(cufft_plan_z_, (cuDoubleComplex*)fft_buffer_aux__.at<GPU>());

                        /* copy to temp buffer*/
                        acc::copy<char>((char*)cufft_work_buf_.at<GPU>(), (char*)fft_buffer_aux__.at<GPU>(), sizeof(double_complex)*fft_buffer_aux__.size());

                        /* repack the buffer */
                        cufft_repack_z_buffer(direction,
                                              comm_.size(),
                                              grid_.size(2),
                                              num_zcol_local,
                                              max_zloc_size_,
                                              z_offsets_.at<GPU>(),
                                              z_sizes_.at<GPU>(),
                                              (cuDoubleComplex*)cufft_work_buf_.at<GPU>(),
                                              (cuDoubleComplex*)fft_buffer_aux__.at<GPU>());

                        break;
                    }
                    case -1: {
                        /* copy to temp buffer*/
                        acc::copy<char>((char*)cufft_work_buf_.at<GPU>(), (char*)fft_buffer_aux__.at<GPU>(), sizeof(double_complex)*fft_buffer_aux__.size());

                        /* repack the buffer back*/
                        cufft_repack_z_buffer(direction,
                                              comm_.size(),
                                              grid_.size(2),
                                              num_zcol_local,
                                              max_zloc_size_,
                                              z_offsets_.at<GPU>(),
                                              z_sizes_.at<GPU>(),
                                              (cuDoubleComplex*)fft_buffer_aux__.at<GPU>(),
                                              (cuDoubleComplex*)cufft_work_buf_.at<GPU>());

                        /* transform all columns */
                        cufft::forward_transform(cufft_plan_z_, (cuDoubleComplex*)fft_buffer_aux__.at<GPU>());
                        /* get all columns from FFT buffer */
                        cufft_batch_unload_gpu(gvec_partition_->zcol_count_fft() * grid_.size(2),
                                               gvec_partition_->gvec_count_fft(),
                                               1,
                                               map_gvec_to_fft_buffer_.at<GPU>(),
                                               (cuDoubleComplex*)fft_buffer_aux__.at<GPU>(),
                                               (cuDoubleComplex*)data__,
                                               0.0,
                                               norm,
                                               cufft_stream_id);
                        break;
                    }
                    default: {
                        TERMINATE("wrong direction");
                    }
                }
                acc::sync_stream(cufft_stream_id);
                #endif
            }

            if (data_ptr_type == CPU) {
                sddk::timer t("sddk::FFT3D::transform_z_serial|cpu");
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    #pragma omp for schedule(dynamic, 1)
                    for (int i = 0; i < num_zcol_local; i++) {
                        /* global index of column */
                        int icol = gvec_partition_->idx_zcol<index_domain_t::local>(i);
                        /* offset of the PW coeffs in the input/output data buffer */
                        int data_offset = gvec_partition_->zcol_offs(icol);

                        switch (direction) {
                            case 1: {
                                /* clear z buffer */
                                std::fill(fftw_buffer_z_[tid], fftw_buffer_z_[tid] + grid_.size(2), 0);
                                /* load z column  of PW coefficients into buffer */
                                for (size_t j = 0; j < gvec_partition_->gvec().zcol(icol).z.size(); j++) {
                                    int z = grid().coord_by_gvec(gvec_partition_->gvec().zcol(icol).z[j], 2);
                                    fftw_buffer_z_[tid][z] = data__[data_offset + j];
                                }

                                /* column with {x,y} = {0,0} has only non-negative z components */
                                if (is_reduced && !icol) {
                                    /* load remaining part of {0,0,z} column */
                                    for (size_t j = 0; j < gvec_partition_->gvec().zcol(icol).z.size(); j++) {
                                        int z = grid().coord_by_gvec(-gvec_partition_->gvec().zcol(icol).z[j], 2);
                                        fftw_buffer_z_[tid][z] = std::conj(data__[data_offset + j]);
                                    }
                                }

                                /* perform local FFT transform of a column */
                                fftw_execute(plan_backward_z_[tid]);

                                /* redistribute z-column for a forthcoming all-to-all or just load the
                                 * full column into auxiliary buffer in serial case */
                                for (int r = 0; r < comm_.size(); r++) {
                                    int lsz  = spl_z_.local_size(r);
                                    int offs = spl_z_.global_offset(r);

                                    std::copy(&fftw_buffer_z_[tid][offs],
                                              &fftw_buffer_z_[tid][offs] + lsz,
                                              &fft_buffer_aux__[offs * num_zcol_local + i * lsz]);
                                }
                                break;

                            }
                            case -1: {
                                /* collect full z-column or just load it from the auxiliary buffer is serial case */
                                for (int r = 0; r < comm_.size(); r++) {
                                    int lsz  = spl_z_.local_size(r);
                                    int offs = spl_z_.global_offset(r);

                                    std::copy(&fft_buffer_aux__[offs * num_zcol_local + i * lsz],
                                              &fft_buffer_aux__[offs * num_zcol_local + i * lsz] + lsz,
                                              &fftw_buffer_z_[tid][offs]);
                                }

                                /* perform local FFT transform of a column */
                                fftw_execute(plan_forward_z_[tid]);

                                /* save z column of PW coefficients */
                                for (size_t j = 0; j < gvec_partition_->gvec().zcol(icol).z.size(); j++) {
                                    int z = grid().coord_by_gvec(gvec_partition_->gvec().zcol(icol).z[j], 2);
                                    data__[data_offset + j] = fftw_buffer_z_[tid][z] * norm;
                                }

                                break;
                            }
                            default: {
                                TERMINATE("wrong direction");
                            }
                        }
                    }
                }
            }
        }

        /// Transformation of z-columns.
        template <int direction, device_t data_ptr_type>
        void transform_z(double_complex* data__,
                         mdarray<double_complex, 1>& fft_buffer_aux__)
        {
            PROFILE("sddk::FFT3D::transform_z");

            int rank = comm_.rank();

            if (direction == -1) {
                /* copy z-sticks to CPU */
                if ((data_ptr_type == CPU && pu_ == GPU) || (data_ptr_type == GPU && !is_gpu_direct_ && comm_.size() > 1)) {
                    sddk::timer t("sddk::FFT3D::transform_z|comm|d-1|DtoH");
                    fft_buffer_aux__.copy<memory_t::device, memory_t::host>(local_size_z_ * gvec_partition_->gvec().num_zcol());
                }

                /* collect full sticks */
                if (comm_.size() > 1) {
                    sddk::timer t("sddk::FFT3D::transform_z|d-1|comm");

                    block_data_descriptor send(comm_.size());
                    block_data_descriptor recv(comm_.size());
                    for (int r = 0; r < comm_.size(); r++) {
                        send.counts[r] = spl_z_.local_size(rank) * gvec_partition_->zcol_count_fft(r);
                        recv.counts[r] = spl_z_.local_size(r)    * gvec_partition_->zcol_count_fft(rank);
                    }
                    send.calc_offsets();
                    recv.calc_offsets();

                    if (data_ptr_type == CPU || !is_gpu_direct_) {
                        /* copy auxiliary buffer because it will be use as the output buffer in the following mpi_a2a */
                        std::copy(&fft_buffer_aux__[0], &fft_buffer_aux__[0] + gvec_partition_->gvec().num_zcol() * local_size_z_,
                                  &fft_buffer_[0]);

                        comm_.barrier();
                        sddk::timer t("sddk::FFT3D::transform_z|comm|d-1|a2a_cpu");
                        comm_.alltoall(&fft_buffer_[0], &send.counts[0], &send.offsets[0], &fft_buffer_aux__[0], &recv.counts[0], &recv.offsets[0]);
                        comm_.barrier();
                    }

                    #ifdef __GPU
                    if (data_ptr_type == GPU && is_gpu_direct_) {
                        /* copy auxiliary buffer because it will be use as the output buffer in the following mpi_a2a */
                        acc::copy<double_complex>(fft_buffer_.at<GPU>(), fft_buffer_aux__.at<GPU>(), gvec_partition_->gvec().num_zcol() * local_size_z_);

                        comm_.barrier();
                        sddk::timer t("sddk::FFT3D::transform_z|comm|d-1|a2a_gpu");
                        comm_.alltoall(fft_buffer_.at<GPU>(), &send.counts[0], &send.offsets[0], fft_buffer_aux__.at<GPU>(), &recv.counts[0], &recv.offsets[0]);
                        comm_.barrier();
                    }

                    /* buffer is on CPU after mpi_a2a and has to be copied to GPU */
                    if (data_ptr_type == GPU && !is_gpu_direct_) {
//                        comm_.barrier();
                        sddk::timer t("sddk::FFT3D::transform_z|comm|d-1|HtoD");
                        fft_buffer_aux__.copy<memory_t::host, memory_t::device>(gvec_partition_->zcol_count_fft() * grid_.size(2));
//                        comm_.barrier();
                    }
                    #endif
                }
            }

            transform_z_serial<direction, data_ptr_type>(data__, fft_buffer_aux__);

            if (direction == 1) {
                /* scatter z-columns between slabs of FFT buffer */
                if (comm_.size() > 1) {
                    #ifdef __GPU
                    if (data_ptr_type == GPU && !is_gpu_direct_) {
                        sddk::timer t("sddk::FFT3D::transform_z|comm|d1|DtoH");
                        fft_buffer_aux__.copy<memory_t::device, memory_t::host>(gvec_partition_->zcol_count_fft() * grid_.size(2));
                    }
                    #endif

                    sddk::timer t("sddk::FFT3D::transform_z|d1|comm");

                    block_data_descriptor send(comm_.size());
                    block_data_descriptor recv(comm_.size());
                    for (int r = 0; r < comm_.size(); r++) {
                        send.counts[r] = spl_z_.local_size(r)    * gvec_partition_->zcol_count_fft(rank);
                        recv.counts[r] = spl_z_.local_size(rank) * gvec_partition_->zcol_count_fft(r);
                    }
                    send.calc_offsets();
                    recv.calc_offsets();


                    if (data_ptr_type == CPU || !is_gpu_direct_){
                        comm_.barrier();
                        sddk::timer t("sddk::FFT3D::transform_z|comm|d1|a2a_cpu");
                        /* scatter z-columns */
                        comm_.alltoall(&fft_buffer_aux__[0], &send.counts[0], &send.offsets[0], &fft_buffer_[0], &recv.counts[0], &recv.offsets[0]);
                        comm_.barrier();
                        t.stop();
                        /* copy local fractions of z-columns into auxiliary buffer */
                        std::copy(&fft_buffer_[0], &fft_buffer_[0] + gvec_partition_->gvec().num_zcol() * local_size_z_,
                                  &fft_buffer_aux__[0]);
                    }

                    #ifdef __GPU
                    if (data_ptr_type == GPU && is_gpu_direct_) {
                        comm_.barrier();
                        sddk::timer t("sddk::FFT3D::transform_z|comm|d1|a2a_gpu");
                        /* scatter z-columns */
                        comm_.alltoall(fft_buffer_aux__.at<GPU>(), &send.counts[0], &send.offsets[0], fft_buffer_.at<GPU>(), &recv.counts[0], &recv.offsets[0]);
                        comm_.barrier();
                        t.stop();
                        /* copy local fractions of z-columns into auxiliary buffer */
                        acc::copy<double_complex>(fft_buffer_aux__.at<GPU>(), fft_buffer_.at<GPU>(), gvec_partition_->gvec().num_zcol() * local_size_z_);
                    }
                    #endif
                }
                #ifdef __GPU
                if ((data_ptr_type == CPU && pu_ == GPU) || (comm_.size() > 1 && data_ptr_type == GPU && !is_gpu_direct_ ) ) {
                    sddk::timer t("sddk::FFT3D::transform_z|comm|d1|HtoD");
                    fft_buffer_aux__.copy<memory_t::host, memory_t::device>(local_size_z_ * gvec_partition_->gvec().num_zcol());
                }
                #endif
            }
        }

        /// Apply 2D FFT transformation to z-columns of one complex function.
        template <int direction>
        void transform_xy(mdarray<double_complex, 1>& fft_buffer_aux__)
        {
            PROFILE("sddk::FFT3D::transform_xy");

            int size_xy = grid_.size(0) * grid_.size(1);

            int is_reduced = gvec_partition_->gvec().reduced();

            #ifdef __GPU
            if (pu_ == GPU) {
                /* stream #0 will be doing cuFFT */
                switch (direction) {
                    case 1: {
                        /* srteam #0 unpacks z-columns into proper position of FFT buffer */
                        unpack_z_cols_gpu((cuDoubleComplex*)fft_buffer_aux__.at<GPU>(),
                                          (cuDoubleComplex*)fft_buffer_.at<GPU>(),
                                          grid_.size(0),
                                          grid_.size(1),
                                          local_size_z_,
                                          gvec_partition_->gvec().num_zcol(),
                                          z_col_pos_.at<GPU>(),
                                          is_reduced,
                                          cufft_stream_id);
                        /* stream #0 executes FFT */
                        cufft::backward_transform(cufft_plan_xy_, (cuDoubleComplex*)fft_buffer_.at<GPU>());
                        break;
                    }
                    case -1: {
                        /* stream #0 executes FFT */
                        cufft::forward_transform(cufft_plan_xy_, (cuDoubleComplex*)fft_buffer_.at<GPU>());
                        /* stream #0 packs z-columns */
                        pack_z_cols_gpu((cuDoubleComplex*)fft_buffer_aux__.at<GPU>(),
                                        (cuDoubleComplex*)fft_buffer_.at<GPU>(),
                                        grid_.size(0),
                                        grid_.size(1),
                                        local_size_z_,
                                        gvec_partition_->gvec().num_zcol(),
                                        z_col_pos_.at<GPU>(),
                                        cufft_stream_id);
                        break;
                    }
                }
                acc::sync_stream(0);
            }
            #endif

            if (pu_ == CPU) {
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    #pragma omp for schedule(static)
                    for (int iz = 0; iz < local_size_z_; iz++) {
                        switch (direction) {
                            case 1: {
                                /* clear xy-buffer */
                                std::fill(fftw_buffer_xy_[tid], fftw_buffer_xy_[tid] + size_xy, 0);
                                /* load z-columns into proper location */
                                for (int i = 0; i < gvec_partition_->gvec().num_zcol(); i++) {
                                    fftw_buffer_xy_[tid][z_col_pos_(i, 0)] = fft_buffer_aux__[iz + i * local_size_z_];

                                    if (is_reduced && i) {
                                        fftw_buffer_xy_[tid][z_col_pos_(i, 1)] = std::conj(fftw_buffer_xy_[tid][z_col_pos_(i, 0)]);
                                    }
                                }

                                /* execute local FFT transform */
                                fftw_execute(plan_backward_xy_[tid]);

                                /* copy xy plane to the main FFT buffer */
                                std::copy(fftw_buffer_xy_[tid], fftw_buffer_xy_[tid] + size_xy, &fft_buffer_[iz * size_xy]);

                                break;
                            }
                            case -1: {
                                /* copy xy plane from the main FFT buffer */
                                std::copy(&fft_buffer_[iz * size_xy], &fft_buffer_[iz * size_xy] + size_xy, fftw_buffer_xy_[tid]);

                                /* execute local FFT transform */
                                fftw_execute(plan_forward_xy_[tid]);

                                /* get z-columns */
                                for (int i = 0; i < gvec_partition_->gvec().num_zcol(); i++) {
                                    fft_buffer_aux__[iz  + i * local_size_z_] = fftw_buffer_xy_[tid][z_col_pos_(i, 0)];
                                }

                                break;
                            }
                            default: {
                                TERMINATE("wrong direction");
                            }
                        }
                    }
                }
            }
        }

        /// Apply 2D FFT transformation to z-columns of two real functions.
        template <int direction>
        void transform_xy(mdarray<double_complex, 1>& fft_buffer_aux1__,
                          mdarray<double_complex, 1>& fft_buffer_aux2__)
        {
            PROFILE("sddk::FFT3D::transform_xy");

            if (!gvec_partition_->gvec().reduced()) {
                TERMINATE("reduced set of G-vectors is required");
            }

            int size_xy = grid_.size(0) * grid_.size(1);

            #ifdef __GPU
            if (pu_ == GPU) {
                /* stream #0 will be doing cuFFT */
                switch (direction) {
                    case 1: {
                        /* srteam #0 unpacks z-columns into proper position of FFT buffer */
                        unpack_z_cols_2_gpu((cuDoubleComplex*)fft_buffer_aux1__.at<GPU>(),
                                            (cuDoubleComplex*)fft_buffer_aux2__.at<GPU>(),
                                            (cuDoubleComplex*)fft_buffer_.at<GPU>(),
                                            grid_.size(0),
                                            grid_.size(1),
                                            local_size_z_,
                                            gvec_partition_->gvec().num_zcol(),
                                            z_col_pos_.at<GPU>(),
                                            cufft_stream_id);
                        /* stream #0 executes FFT */
                        cufft::backward_transform(cufft_plan_xy_, (cuDoubleComplex*)fft_buffer_.at<GPU>());
                        break;
                    }
                    case -1: {
                        /* stream #0 executes FFT */
                        cufft::forward_transform(cufft_plan_xy_, (cuDoubleComplex*)fft_buffer_.at<GPU>());
                        /* stream #0 packs z-columns */
                        pack_z_cols_2_gpu((cuDoubleComplex*)fft_buffer_aux1__.at<GPU>(),
                                          (cuDoubleComplex*)fft_buffer_aux2__.at<GPU>(),
                                          (cuDoubleComplex*)fft_buffer_.at<GPU>(),
                                          grid_.size(0),
                                          grid_.size(1),
                                          local_size_z_,
                                          gvec_partition_->gvec().num_zcol(),
                                          z_col_pos_.at<GPU>(),
                                          cufft_stream_id);
                        break;
                    }
                }
                acc::sync_stream(cufft_stream_id);
            }
            #endif

            if (pu_ == CPU) {
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    #pragma omp for schedule(static)
                    for (int iz = 0; iz < local_size_z_; iz++) {
                        switch (direction) {
                            case 1: {
                                /* clear xy-buffer */
                                std::fill(fftw_buffer_xy_[tid], fftw_buffer_xy_[tid] + size_xy, 0);

                                /* load first z-column into proper location */
                                fftw_buffer_xy_[tid][z_col_pos_(0, 0)] = fft_buffer_aux1__[iz] +
                                    double_complex(0, 1) * fft_buffer_aux2__[iz];

                                /* load remaining z-columns into proper location */
                                for (int i = 1; i < gvec_partition_->gvec().num_zcol(); i++) {
                                    /* {x, y} part */
                                    fftw_buffer_xy_[tid][z_col_pos_(i, 0)] = fft_buffer_aux1__[iz + i * local_size_z_] +
                                        double_complex(0, 1) * fft_buffer_aux2__[iz + i * local_size_z_];

                                    /* {-x, -y} part */
                                    fftw_buffer_xy_[tid][z_col_pos_(i, 1)] = std::conj(fft_buffer_aux1__[iz + i * local_size_z_]) +
                                        double_complex(0, 1) * std::conj(fft_buffer_aux2__[iz + i * local_size_z_]);
                                }

                                /* execute local FFT transform */
                                fftw_execute(plan_backward_xy_[tid]);

                                /* copy xy plane to the main FFT buffer */
                                std::copy(fftw_buffer_xy_[tid], fftw_buffer_xy_[tid] + size_xy, &fft_buffer_[iz * size_xy]);

                                break;
                            }
                            case -1: {
                                /* copy xy plane from the main FFT buffer */
                                std::copy(&fft_buffer_[iz * size_xy], &fft_buffer_[iz * size_xy] + size_xy, fftw_buffer_xy_[tid]);

                                /* execute local FFT transform */
                                fftw_execute(plan_forward_xy_[tid]);

                                /* get z-columns */
                                for (int i = 0; i < gvec_partition_->gvec().num_zcol(); i++) {
                                    fft_buffer_aux1__[iz  + i * local_size_z_] = 0.5 *
                                        (fftw_buffer_xy_[tid][z_col_pos_(i, 0)] + std::conj(fftw_buffer_xy_[tid][z_col_pos_(i, 1)]));

                                    fft_buffer_aux2__[iz  + i * local_size_z_] = double_complex(0, -0.5) *
                                        (fftw_buffer_xy_[tid][z_col_pos_(i, 0)] - std::conj(fftw_buffer_xy_[tid][z_col_pos_(i, 1)]));
                                }

                                break;
                            }
                            default: {
                                TERMINATE("wrong direction");
                            }
                        }
                    }
                }
            }
        }

    public:

        /// Constructor.
        FFT3D(FFT3D_grid grid__,
              Communicator const& comm__,
              device_t pu__)
            : comm_(comm__)
            , pu_(pu__)
            , grid_(grid__)
        {
            PROFILE("sddk::FFT3D::FFT3D");

            /* split z-direction */
            spl_z_ = splindex<block>(grid_.size(2), comm_.size(), comm_.rank());
            local_size_z_ = spl_z_.local_size();
            offset_z_     = spl_z_.global_offset();

            if (pu_ == CPU) {
                host_memory_type_ = memory_t::host;
            } else {
                host_memory_type_ = memory_t::host_pinned;
            }

            /* allocate main buffer */
            fft_buffer_ = mdarray<double_complex, 1>(local_size(), host_memory_type_, "FFT3D.fft_buffer_");

            /* allocate 1d and 2d buffers */
            for (int i = 0; i < omp_get_max_threads(); i++) {
                fftw_buffer_z_.push_back((double_complex*)fftw_malloc(grid_.size(2) * sizeof(double_complex)));
                fftw_buffer_xy_.push_back((double_complex*)fftw_malloc(grid_.size(0) * grid_.size(1) * sizeof(double_complex)));
            }

            plan_forward_z_   = std::vector<fftw_plan>(omp_get_max_threads());
            plan_forward_xy_  = std::vector<fftw_plan>(omp_get_max_threads());
            plan_backward_z_  = std::vector<fftw_plan>(omp_get_max_threads());
            plan_backward_xy_ = std::vector<fftw_plan>(omp_get_max_threads());

            for (int i = 0; i < omp_get_max_threads(); i++) {
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
            if (pu_ == GPU) {

                #ifdef __GPU_DIRECT
                #pragma message "=========== GPU direct is enabled =============="
                is_gpu_direct_ = true;
                #endif

                int auto_alloc{0};
                int dim_xy[] = {grid_.size(1), grid_.size(0)};
                /* create plan handler for xy transform */
                cufft::create_plan_handle(&cufft_plan_xy_);
                /* create plan for xy transform */
                cufft::create_batch_plan(cufft_plan_xy_, 2, dim_xy, dim_xy, 1, grid_.size(0) * grid_.size(1), local_size_z_, auto_alloc);
                /* stream #0 will execute FFTs */
                cufft::set_stream(cufft_plan_xy_, 0);

                /* allocate arrays with z- offsets and sizes on the host and device*/
                z_offsets_ = mdarray<int, 1>(comm_.size(), memory_t::host | memory_t::device);
                z_sizes_   = mdarray<int, 1>(comm_.size(), memory_t::host | memory_t::device);

                /* copy z- offsets and sizes in mdarray since we can store it also on device*/
                for (int r = 0; r < comm_.size(); r++) {
                    z_offsets_(r) = spl_z_.global_offset(r);
                    z_sizes_(r) = spl_z_.local_size(r);

                    if (max_zloc_size_ < z_sizes_(r)) {
                        max_zloc_size_ = z_sizes_(r);
                    }
                }

                /* copy them to device */
                z_offsets_.copy<memory_t::host, memory_t::device>();
                z_sizes_.copy<memory_t::host, memory_t::device>();
            }
            #endif
        }

        /// Destructor.
        ~FFT3D()
        {
            if (gvec_partition_) {
                dismiss();
            }
            for (int i = 0; i < omp_get_max_threads(); i++) {
                fftw_free(fftw_buffer_z_[i]);
                fftw_free(fftw_buffer_xy_[i]);

                fftw_destroy_plan(plan_forward_z_[i]);
                fftw_destroy_plan(plan_forward_xy_[i]);
                fftw_destroy_plan(plan_backward_z_[i]);
                fftw_destroy_plan(plan_backward_xy_[i]);
            }
            #ifdef __GPU
            if (pu_ == GPU) {
                cufft::destroy_plan_handle(cufft_plan_xy_);
                if (cufft_plan_z_created_) {
                    cufft::destroy_plan_handle(cufft_plan_z_);
                }
            }
            #endif
        }

        /// Load real-space values to the FFT buffer.
        /** \param [in] data CPU pointer to the real-space data. */
        template <typename T>
        inline void input(T* data__)
        {
            for (int i = 0; i < local_size(); i++) {
                fft_buffer_[i] = data__[i];
            }
            if (pu_ == GPU) {
                fft_buffer_.copy<memory_t::host, memory_t::device>();
            }
        }

        /// Get real-space values from the FFT buffer.
        /** \param [out] data CPU pointer to the real-space data. */
        inline void output(double* data__)
        {
            if (pu_ == GPU) {
                fft_buffer_.copy<memory_t::device, memory_t::host>();
            }
            for (int i = 0; i < local_size(); i++) {
                data__[i] = fft_buffer_[i].real();
            }
        }

        /// Get real-space values from the FFT buffer.
        /** \param [out] data CPU pointer to the real-space data. */
        inline void output(double_complex* data__)
        {
            switch (pu_) {
                case CPU: {
                    std::memcpy(data__, fft_buffer_.at<CPU>(), local_size() * sizeof(double_complex));
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    acc::copyout(data__, fft_buffer_.at<GPU>(), local_size());
                    #endif
                    break;
                }
            }
        }

        /// Informational about the FFT grid.
        FFT3D_grid const& grid() const
        {
            return grid_;
        }

        /// Total size of the FFT grid.
        inline int size() const
        {
            return grid_.size();
        }

        /// Size of the local part of FFT buffer.
        inline int local_size() const
        {
            return grid_.size(0) * grid_.size(1) * local_size_z_;
        }

        inline int local_size_z() const
        {
            return local_size_z_;
        }

        inline int offset_z() const
        {
            return offset_z_;
        }

        inline int size_x() const
        {
            return grid_.size(0);
        }

        inline int size_y() const
        {
            return grid_.size(1);
        }

        /// Direct access to the FFT buffer
        inline double_complex& buffer(int idx__)
        {
            return fft_buffer_[idx__];
        }

        /// FFT buffer.
        inline mdarray<double_complex, 1>& buffer()
        {
            return fft_buffer_;
        }

        /// Communicator of the FFT transform.
        Communicator const& comm() const
        {
            return comm_;
        }

        /// True if this FFT transformation is parallel.
        inline bool parallel() const
        {
            return (comm_.size() != 1);
        }

        /// Return the type of processing unit.
        inline device_t pu() const
        {
            return pu_;
        }

        /// Prepare FFT driver to transfrom functions with the specific G-vector distribution.
        void prepare(Gvec_partition const& gvec__)
        {
            PROFILE("sddk::FFT3D::prepare");

            if (gvec_partition_) {
                TERMINATE("FFT3D is already prepared for another G-vector partition");
            }

            gvec_partition_ = &gvec__;

            int nc = gvec__.gvec().reduced() ? 2 : 1;

            sddk::timer t1("sddk::FFT3D::prepare|cpu");
            /* get positions of z-columns in xy plane */
            z_col_pos_ = mdarray<int, 2>(gvec__.gvec().num_zcol(), nc, memory_t::host, "FFT3D.z_col_pos_");
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < gvec__.gvec().num_zcol(); i++) {
                int icol = gvec__.idx_zcol<index_domain_t::global>(i);
                int x = grid().coord_by_gvec(gvec__.gvec().zcol(icol).x, 0);
                int y = grid().coord_by_gvec(gvec__.gvec().zcol(icol).y, 1);
                assert(x >= 0 && x < grid().size(0));
                assert(y >= 0 && y < grid().size(1));
                z_col_pos_(i, 0) = x + y * grid_.size(0);
                if (gvec__.gvec().reduced()) {
                    x = grid().coord_by_gvec(-gvec__.gvec().zcol(icol).x, 0);
                    y = grid().coord_by_gvec(-gvec__.gvec().zcol(icol).y, 1);
                    assert(x >= 0 && x < grid().size(0));
                    assert(y >= 0 && y < grid().size(1));
                    z_col_pos_(i, 1) = x + y * grid_.size(0);
                }
            }
            t1.stop();

            #ifdef __GPU
            if (pu_ == GPU) {
                sddk::timer t2("sddk::FFT3D::prepare|gpu");
                size_t work_size;
                map_gvec_to_fft_buffer_ = mdarray<int, 1>(gvec__.gvec_count_fft(), memory_t::host | memory_t::device,
                                                          "FFT3D.map_zcol_to_fft_buffer_");
                /* loop over local set of columns */
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < gvec__.zcol_count_fft(); i++) {
                    /* global index of z-column */
                    int icol = gvec_partition_->idx_zcol<index_domain_t::local>(i);
                    /* loop over z-colmn */
                    for (size_t j = 0; j < gvec__.gvec().zcol(icol).z.size(); j++) {
                        /* local index of the G-vector */
                        size_t ig = gvec__.zcol_offs(icol) + j;
                        /* coordinate inside FFT 1D bufer */
                        int z = grid().coord_by_gvec(gvec__.gvec().zcol(icol).z[j], 2);
                        assert(z >= 0 && z < grid().size(2));
                        map_gvec_to_fft_buffer_[ig] = i * grid_.size(2) + z;
                    }
                }
                map_gvec_to_fft_buffer_.copy<memory_t::host, memory_t::device>();

                /* for the rank that stores {x=0,y=0} column we need to create a small second mapping */
                if (gvec__.gvec().reduced() && comm_.rank() == 0) {
                    map_gvec_to_fft_buffer_x0y0_ = mdarray<int, 1>(gvec__.gvec().zcol(0).z.size(), memory_t::host | memory_t::device,
                                                                   "FFT3D.map_zcol_to_fft_buffer_x0y0_");
                    for (size_t j = 0; j < gvec__.gvec().zcol(0).z.size(); j++) {
                        int z = grid().coord_by_gvec(-gvec__.gvec().zcol(0).z[j], 2);
                        assert(z >= 0 && z < grid().size(2));
                        map_gvec_to_fft_buffer_x0y0_[j] = z;
                    }
                    map_gvec_to_fft_buffer_x0y0_.copy<memory_t::host, memory_t::device>();
                }

                int dim_z[] = {grid_.size(2)};
                int dims_xy[] = {grid_.size(1), grid_.size(0)};

                /* check if we need to create a batch cuFFT plan for larger number of z-columns */
                if (gvec__.zcol_count_fft() > zcol_count_max_) {
                    if (cufft_plan_z_created_) {
                        cufft::destroy_plan_handle(cufft_plan_z_);
                        cufft_plan_z_created_ = false;
                    }
                    /* now this is the maximum number of columns */
                    zcol_count_max_ = gvec__.zcol_count_fft();
                    cufft::create_plan_handle(&cufft_plan_z_);
                    cufft::set_stream(cufft_plan_z_, cufft_stream_id);

                    cufft::create_batch_plan(cufft_plan_z_, 1, dim_z, dim_z, 1, grid_.size(2), zcol_count_max_, 0);
                    cufft_plan_z_created_ = true;
                }

                /* maximum worksize of z and xy transforms */
                work_size = std::max(cufft::get_work_size(2, dims_xy, local_size_z_),
                                     cufft::get_work_size(1, dim_z, gvec__.zcol_count_fft()));

                /* use as temp array also after z-transform*/
                work_size = std::max(work_size, sizeof(double_complex) * grid_.size(2) * local_size_z_);

                /* allocate cufft work buffer */
                cufft_work_buf_ = mdarray<char, 1>(work_size, memory_t::device, "FFT3D.cufft_work_buf_");

                /* set work area for cufft */
                cufft::set_work_area(cufft_plan_xy_, cufft_work_buf_.at<GPU>());
                cufft::set_work_area(cufft_plan_z_, cufft_work_buf_.at<GPU>());

                fft_buffer_aux1_.allocate(memory_t::device);
                fft_buffer_aux2_.allocate(memory_t::device);

                fft_buffer_.allocate(memory_t::device);

                z_col_pos_.allocate(memory_t::device);
                z_col_pos_.copy<memory_t::host, memory_t::device>();
            }
            #endif
        }

        void dismiss()
        {
            if (pu_ == GPU) {
                fft_buffer_aux1_.deallocate(memory_t::device);
                fft_buffer_aux2_.deallocate(memory_t::device);
                z_col_pos_.deallocate(memory_t::device);
                fft_buffer_.deallocate(memory_t::device);
#ifdef __GPU
                cufft_work_buf_.deallocate(memory_t::device);
                map_gvec_to_fft_buffer_.deallocate(memory_t::device);
                map_gvec_to_fft_buffer_x0y0_.deallocate(memory_t::device);
#endif
            }
            gvec_partition_ = nullptr;
        }

        /// Transform a single functions.
        template <int direction, device_t data_ptr_type = CPU>
        void transform(double_complex* data__)
        {
            PROFILE("sddk::FFT3D::transform");

            if (!gvec_partition_) {
                TERMINATE("FFT3D is not ready");
            }

            /* reallocate auxiliary buffer if needed */
            size_t sz_max;
            if (comm_.size() > 1) {
                int rank = comm_.rank();
                int num_zcol_local = gvec_partition_->zcol_count_fft(rank);
                /* we need this buffer size for mpi_alltoall */
                sz_max = std::max(grid_.size(2) * num_zcol_local, local_size());
            } else {
                sz_max = grid_.size(2) * gvec_partition_->gvec().num_zcol();
            }
            if (sz_max > fft_buffer_aux1_.size()) {
                fft_buffer_aux1_ = mdarray<double_complex, 1>(sz_max, host_memory_type_, "fft_buffer_aux1_");
                if (pu_ == GPU) {
                    fft_buffer_aux1_.allocate(memory_t::device);
                }
            }

            switch (direction) {
                case 1: {
                    transform_z<direction, data_ptr_type>(data__, fft_buffer_aux1_);
                    transform_xy<direction>(fft_buffer_aux1_);
                    break;
                }
                case -1: {
                    transform_xy<direction>(fft_buffer_aux1_);
                    transform_z<direction, data_ptr_type>(data__, fft_buffer_aux1_);
                    break;
                }
                default: {
                    TERMINATE("wrong direction");
                }
            }
        }

        /// Transform two real functions.
        template <int direction, device_t data_ptr_type = CPU>
        void transform(double_complex* data1__, double_complex* data2__)
        {
            PROFILE("sddk::FFT3D::transform");

            if (!gvec_partition_) {
                TERMINATE("FFT3D is not ready");
            }

            if (!gvec_partition_->gvec().reduced()) {
                TERMINATE("reduced set of G-vectors is required");
            }

            /* reallocate auxiliary buffers if needed */
            size_t sz_max;
            if (comm_.size() > 1) {
                int rank = comm_.rank();
                int num_zcol_local = gvec_partition_->zcol_count_fft(rank);
                /* we need this buffer for mpi_alltoall */
                sz_max = std::max(grid_.size(2) * num_zcol_local, local_size());
            } else {
                sz_max = grid_.size(2) * gvec_partition_->gvec().num_zcol();
            }

            if (sz_max > fft_buffer_aux1_.size()) {
                fft_buffer_aux1_ = mdarray<double_complex, 1>(sz_max, host_memory_type_, "fft_buffer_aux1_");
                if (pu_ == GPU) {
                    fft_buffer_aux1_.allocate(memory_t::device);
                }
            }
            if (sz_max > fft_buffer_aux2_.size()) {
                fft_buffer_aux2_ = mdarray<double_complex, 1>(sz_max, host_memory_type_, "fft_buffer_aux2_");
                if (pu_ == GPU) {
                    fft_buffer_aux2_.allocate(memory_t::device);
                }
            }
            switch (direction) {
                case 1: {
                    transform_z<direction, data_ptr_type>(data1__, fft_buffer_aux1_);
                    transform_z<direction, data_ptr_type>(data2__, fft_buffer_aux2_);
                    transform_xy<direction>(fft_buffer_aux1_, fft_buffer_aux2_);
                    break;
                }
                case -1: {
                    transform_xy<direction>(fft_buffer_aux1_, fft_buffer_aux2_);
                    transform_z<direction, data_ptr_type>(data1__, fft_buffer_aux1_);
                    transform_z<direction, data_ptr_type>(data2__, fft_buffer_aux2_);
                    break;
                }
                default: {
                    TERMINATE("wrong direction");
                }
            }
        }
};

} // namespace sddk

#endif // __FFT3D_H__

/** \page ft_pw Fourier transform and plane wave normalization
 *
 *  We use plane waves in two different cases: a) plane waves (or augmented plane waves in the case of APW+lo method)
 *  as a basis for expanding Kohn-Sham wave functions and b) plane waves are used to expand charge density and
 *  potential. When we are dealing with plane wave basis functions it is convenient to adopt the following
 *  normalization:
 *  \f[
 *      \langle {\bf r} |{\bf G+k} \rangle = \frac{1}{\sqrt \Omega} e^{i{\bf (G+k)r}}
 *  \f]
 *  such that
 *  \f[
 *      \langle {\bf G+k} |{\bf G'+k} \rangle_{\Omega} = \delta_{{\bf GG'}}
 *  \f]
 *  in the unit cell. However, for the expansion of periodic functions such as density or potential, the following
 *  convention is more appropriate:
 *  \f[
 *      \rho({\bf r}) = \sum_{\bf G} e^{i{\bf Gr}} \rho({\bf G})
 *  \f]
 *  where
 *  \f[
 *      \rho({\bf G}) = \frac{1}{\Omega} \int_{\Omega} e^{-i{\bf Gr}} \rho({\bf r}) d{\bf r} =
 *          \frac{1}{\Omega} \sum_{{\bf r}_i} e^{-i{\bf Gr}_i} \rho({\bf r}_i) \frac{\Omega}{N} =
 *          \frac{1}{N} \sum_{{\bf r}_i} e^{-i{\bf Gr}_i} \rho({\bf r}_i)
 *  \f]
 *  i.e. with such convention the plane-wave expansion coefficients are obtained with a normalized FFT.
 */

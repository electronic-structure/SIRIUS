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

/** \file fft3d.h
 *   
 *  \brief Contains declaration and partial implementation of FFT3D class.
 */

#ifndef __FFT3D_H__
#define __FFT3D_H__

#include <fftw3.h>
#include "typedefs.h"
#include "mdarray.h"
#include "splindex.h"
#include "vector3d.h"
#include "descriptors.h"
#include "fft3d_grid.h"
#include "gvec.h"

namespace sirius {

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
 */
class FFT3D
{
    protected:
        
        /// Communicator for the parallel FFT.
        Communicator const& comm_;

        /// Main processing unit of this FFT.
        processing_unit_t pu_;
        
        /// Split z-direction.
        splindex<block> spl_z_;

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

        #ifdef __GPU
        cufftHandle cufft_plan_;
        mdarray<char, 1> cufft_work_buf_;
        int cufft_nbatch_;
        #endif

        bool prepared_;
        
        template <int direction, bool use_reduction>
        void transform_z_serial(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__, mdarray<double_complex, 1>& fft_buffer_aux__);

        template <int direction, bool use_reduction>
        void transform_z_parallel(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__, mdarray<double_complex, 1>& fft_buffer_aux__);

        template <int direction, bool use_reduction>
        void transform_xy(Gvec_FFT_distribution const& gvec_fft_distr__, mdarray<double_complex, 1>& fft_buffer_aux__);

        template <int direction>
        void transform_xy(Gvec_FFT_distribution const& gvec_fft_distr__, mdarray<double_complex, 1>& fft_buffer_aux1__, mdarray<double_complex, 1>& fft_buffer_aux2__);

    public:

        FFT3D(FFT3D_grid grid__,
              Communicator const& comm__,
              processing_unit_t pu__,
              double gpu_workload = 0.8);

        ~FFT3D();

        template <int direction>
        void transform(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data__);

        template <int direction>
        void transform(Gvec_FFT_distribution const& gvec_fft_distr__, double_complex* data1__, double_complex* data2__);

        //template<typename T>
        //inline void input(int n__, int const* map__, T const* data__)
        //{
        //    memset(fftw_buffer_, 0, local_size() * sizeof(double_complex));
        //    for (int i = 0; i < n__; i++) fftw_buffer_[map__[i]] = data__[i];
        //}

        template <typename T>
        inline void input(T* data__)
        {
            for (int i = 0; i < local_size(); i++) fft_buffer_[i] = data__[i];
            #ifdef __GPU
            if (pu_ == GPU) fft_buffer_.copy_to_device();
            #endif
        }
        
        inline void output(double* data__)
        {
            #ifdef __GPU
            if (pu_ == GPU) fft_buffer_.copy_to_host();
            #endif
            for (int i = 0; i < local_size(); i++) data__[i] = fft_buffer_[i].real();
        }
        
        inline void output(double_complex* data__)
        {
            switch (pu_)
            {
                case CPU:
                {
                    std::memcpy(data__, fft_buffer_.at<CPU>(), local_size() * sizeof(double_complex));
                    break;
                }
                case GPU:
                {
                    #ifdef __GPU
                    acc::copyout(data__, fft_buffer_.at<GPU>(), local_size());
                    #endif
                    break;
                }
            }
        }
        
        //inline void output(int n__, int const* map__, double_complex* data__)
        //{
        //    for (int i = 0; i < n__; i++) data__[i] = fftw_buffer_[map__[i]];
        //}

        //inline void output(int n__, int const* map__, double_complex* data__, double beta__)
        //{
        //    for (int i = 0; i < n__; i++) data__[i] += beta__ * fftw_buffer_[map__[i]];
        //}

        FFT3D_grid const& grid() const
        {
            return grid_;
        }
        
        /// Total size of the FFT grid.
        inline int size() const
        {
            return grid_.size();
        }

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

        /// Direct access to the fft buffer
        inline double_complex& buffer(int idx__)
        {
            return fft_buffer_[idx__];
        }
        
        template <processing_unit_t pu>
        inline double_complex* buffer()
        {
            return fft_buffer_.at<pu>();
        }
        
        Communicator const& comm() const
        {
            return comm_;
        }

        inline bool parallel() const
        {
            return (comm_.size() != 1);
        }

        inline bool hybrid() const
        {
            return (pu_ == GPU);
        }

        void prepare()
        {
            #ifdef __GPU
            if (pu_ == GPU)
            {
                fft_buffer_aux1_.allocate_on_device();
                fft_buffer_aux2_.allocate_on_device();
                allocate_on_device();
            }
            #endif
            prepared_ = true;
        }

        void dismiss()
        {
            #ifdef __GPU
            if (pu_ == GPU)
            {
                fft_buffer_aux1_.deallocate_on_device();
                fft_buffer_aux2_.deallocate_on_device();
                deallocate_on_device();
            }
            #endif
            prepared_ = false;
        }

        #ifdef __GPU
        void allocate_on_device()
        {
            PROFILE();
            fft_buffer_.pin_memory();
            fft_buffer_.allocate_on_device();
            
            size_t work_size = cufft_get_size_2d(grid_.size(0), grid_.size(1), cufft_nbatch_);

            cufft_work_buf_ = mdarray<char, 1>(nullptr, work_size, "cufft_work_buf_");
            cufft_work_buf_.allocate_on_device();
            cufft_set_work_area(cufft_plan_, cufft_work_buf_.at<GPU>());
        }

        void deallocate_on_device()
        {
            fft_buffer_.unpin_memory();
            fft_buffer_.deallocate_on_device();
            cufft_work_buf_.deallocate_on_device();
        }

        void copy_to_device()
        {
            fft_buffer_.copy_to_device();
        }
        #endif
};

};

#endif // __FFT3D_H__

/** \page ft_pw Fourier transform and plane wave normalization
 *
 *  We use plane waves in two different cases: a) plane waves (or augmented plane waves in the case of APW+lo method)
 *  form a basis for expanding Kohn-Sham wave functions and b) plane waves are used to expand charge density and
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


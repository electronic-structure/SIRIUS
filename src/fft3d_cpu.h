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

/** \file fft3d_cpu.h
 *   
 *  \brief Contains CPU specialization.
 */

 #include "matrix3d.h"

namespace sirius {

/// CPU interfacce to FFT3D.
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
 */
class FFT3D_CPU
{
    private:

        /// Number of working threads inside each FFT.
        int num_fft_workers_;
        
        /// Number of threads doing individual FFTs.
        int num_fft_threads_;
        
        /// Communicator for the parallel FFT.
        Communicator comm_;
        
        /// Auxiliary array to store full z-sticks of FFT buffer.
        mdarray<double_complex, 2> fft_buffer_aux_;

        splindex<block> spl_z_;

        /// Size of each dimension.
        int grid_size_[3];

        int local_size_;

        int local_size_z_;

        int offset_z_;

        /// Reciprocal space range
        std::pair<int, int> grid_limits_[3];
        
        /// Backward transformation plan for each thread
        std::vector<fftw_plan> plan_backward_;

        std::vector<fftw_plan> plan_backward_z_;

        std::vector<fftw_plan> plan_backward_xy_;
        
        /// Forward transformation plan for each thread
        std::vector<fftw_plan> plan_forward_;

        std::vector<fftw_plan> plan_forward_z_;

        std::vector<fftw_plan> plan_forward_xy_;
    
        /// In/out buffer for each thread
        std::vector<double_complex*> fftw_buffer_;

        std::vector<double_complex*> fftw_buffer_z_;
        std::vector<double_complex*> fftw_buffer_xy_;

        /// Execute backward transformation.
        inline void backward(int thread_id = 0)
        {    
            fftw_execute(plan_backward_[thread_id]);
        }
        
        /// Execute forward transformation.
        inline void forward(int thread_id = 0)
        {    
            fftw_execute(plan_forward_[thread_id]);
            double norm = 1.0 / size();
            #pragma omp parallel for schedule(static) num_threads(num_fft_workers_)
            for (int i = 0; i < local_size(); i++) fftw_buffer_[thread_id][i] *= norm;
        }

        /// Find smallest optimal grid size starting from n.
        int find_grid_size(int n)
        {
            while (true)
            {
                int m = n;
                for (int k = 2; k <= 5; k++)
                {
                    while (m % k == 0) m /= k;
                }
                if (m == 1) 
                {
                    return n;
                }
                else 
                {
                    n++;
                }
            }
        }

        void backward_custom(std::vector< std::pair<int, int> > const& z_sticks_coord__, int thread_id__);
        
        void forward_custom(std::vector< std::pair<int, int> > const& z_sticks_coord__, int thread_id__);
        
    public:

        FFT3D_CPU(vector3d<int> dims__,
                  int num_fft_threads__,
                  int num_fft_workers__,
                  Communicator const& comm__);

        ~FFT3D_CPU()
        {
            PROFILE();

            for (int i = 0; i < num_fft_threads_; i++)
            {
                fftw_free(fftw_buffer_[i]);
                fftw_destroy_plan(plan_backward_[i]);
                fftw_destroy_plan(plan_forward_[i]);
            }

            for (int i = 0; i < num_fft_workers_ * num_fft_threads_; i++)
            {
                fftw_free(fftw_buffer_z_[i]);
                fftw_free(fftw_buffer_xy_[i]);
                fftw_destroy_plan(plan_forward_z_[i]);
                fftw_destroy_plan(plan_forward_xy_[i]);
                fftw_destroy_plan(plan_backward_z_[i]);
                fftw_destroy_plan(plan_backward_xy_[i]);
            }
        }

        ///// Execute the transformation for a given thread.
        //inline void transform(int direction__, int thread_id__ = 0)
        //{
        //    assert(thread_id__ < num_fft_threads_);

        //    switch (direction__)
        //    {
        //        case 1:
        //        {
        //            backward(thread_id__);
        //            break;
        //        }
        //        case -1:
        //        {
        //            forward(thread_id__);
        //            break;
        //        }
        //        default:
        //        {
        //            error_local(__FILE__, __LINE__, "wrong FFT direction");
        //        }
        //    }
        //}

        void transform(int direction__, std::vector< std::pair<int, int> > const& z_sticks_coord__, int thread_id__ = 0)
        {
            assert(thread_id__ < num_fft_threads_);

            switch (direction__)
            {
                case 1:
                {
                    backward_custom(z_sticks_coord__, thread_id__);
                    break;
                }
                case -1:
                {
                    forward_custom(z_sticks_coord__, thread_id__);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }

        template<typename T>
        inline void input(int n__, int const* map__, T* data__, int thread_id__ = 0)
        {
            assert(thread_id__ < num_fft_threads_);
            
            memset(fftw_buffer_[thread_id__], 0, local_size() * sizeof(double_complex));
            for (int i = 0; i < n__; i++) fftw_buffer_[thread_id__][map__[i]] = data__[i];
        }

        template <typename T>
        inline void input(T* data__, int thread_id__ = 0)
        {
            assert(thread_id__ < num_fft_threads_);
            
            for (int i = 0; i < local_size(); i++) fftw_buffer_[thread_id__][i] = data__[i];
        }
        
        inline void output(double* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);

            for (int i = 0; i < local_size(); i++) data[i] = std::real(fftw_buffer_[thread_id][i]);
        }
        
        inline void output(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads_);

            memcpy(data, fftw_buffer_[thread_id], local_size() * sizeof(double_complex));
        }
        
        inline void output(int n__, int const* map__, double_complex* data__, int thread_id__ = 0)
        {
            assert(thread_id__ < num_fft_threads_);

            for (int i = 0; i < n__; i++) data__[i] = fftw_buffer_[thread_id__][map__[i]];
        }

        inline void output(int n, int const* map, double_complex* data, int thread_id, double alpha)
        {
            assert(thread_id < num_fft_threads_);

            for (int i = 0; i < n; i++) data[i] += alpha * fftw_buffer_[thread_id][map[i]];
        }
        
        inline const std::pair<int, int>& grid_limits(int idim)
        {
            return grid_limits_[idim];
        }

        /// Size of a given dimension.
        inline int size(int d) const
        {
            assert(d >= 0 && d < 3);
            return grid_size_[d]; 
        }

        /// Total size of the FFT grid.
        inline int size() const
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        inline int local_size() const
        {
            return grid_size_[0] * grid_size_[1] * local_size_z_;
        }

        inline int local_size_z() const
        {
            return local_size_z_;
        }

        inline int offset_z() const
        {
            return offset_z_;
        }

        /// Return linear index of a plane-wave harmonic with fractional coordinates (i0, i1, i2) inside fft buffer.
        inline int index(int i0, int i1, int i2) const
        {
            if (i0 < 0) i0 += grid_size_[0];
            if (i1 < 0) i1 += grid_size_[1];
            if (i2 < 0) i2 += grid_size_[2];

            return (i0 + i1 * grid_size_[0] + i2 * grid_size_[0] * grid_size_[1]);
        }

        /// Direct access to the fft buffer
        inline double_complex& buffer(int i, int thread_id = 0)
        {
            return fftw_buffer_[thread_id][i];
        }
        
        vector3d<int> grid_size() const
        {
            return vector3d<int>(grid_size_[0], grid_size_[1], grid_size_[2]);
        }

        inline vector3d<int> gvec_by_grid_pos(int i0__, int i1__, int i2__) const
        {
            if (i0__ > grid_limits_[0].second) i0__ -= grid_size_[0];
            if (i1__ > grid_limits_[1].second) i1__ -= grid_size_[1];
            if (i2__ > grid_limits_[2].second) i2__ -= grid_size_[2];

            return vector3d<int>(i0__, i1__, i2__);
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        inline bool parallel() const
        {
            return (comm_.size() != 1);
        }

        inline int num_fft_threads() const
        {
            return num_fft_threads_;
        }

        inline int num_fft_workers() const
        {
            return num_fft_workers_;
        }
};

};

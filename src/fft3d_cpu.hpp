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

/** \file fft3d_cpu.hpp
 *   
 *  \brief Contains CPU specialization.
 */

/// CPU specialization of FFT3D class.
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
 *  FFTW performs an "out of place" transformation, which means that we need to allocate both input and output buffers.
 *  To get the most performance out of multithreading we are going to put whole FFTs into different threads instead
 *  of using threaded implementation for each transform. 
*/
template<> 
class FFT3D<cpu>
{
    private:

        /// Size of each dimension.
        int grid_size_[3];

        /// reciprocal space range
        std::pair<int, int> grid_limits_[3];
        
        /// backward transformation plan for each thread
        std::vector<fftw_plan> plan_backward_;
        
        /// forward transformation plan for each thread
        std::vector<fftw_plan> plan_forward_;
    
        /// inout buffer for each thread
        mdarray<double_complex, 2> fftw_buffer_;
        
        /// Split index of FFT buffer.
        splindex<block> spl_fft_size_;

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
            for (int i = 0; i < size(); i++) fftw_buffer_(i, thread_id) *= norm;
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
        
    public:

        FFT3D(vector3d<int> dims, Communicator const& comm__)
        {
            Timer t("sirius::FFT3D<cpu>::FFT3D");
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(dims[i]);
                
                grid_limits_[i].second = grid_size_[i] / 2;
                grid_limits_[i].first = grid_limits_[i].second - grid_size_[i] + 1;
            }

            #ifdef _FFTW_THREADED_
            fftw_plan_with_nthreads(Platform::num_fft_threads());

            fftw_buffer_ = mdarray<double_complex, 2>(size(), 1);

            plan_backward_.resize(1);
            plan_forward_.resize(1);

            plan_backward_[0] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                 (fftw_complex*)&fftw_buffer_(0, 0), 
                                                 (fftw_complex*)&fftw_buffer_(0, 0), 1, FFTW_MEASURE);
            plan_forward_[0] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                (fftw_complex*)&fftw_buffer_(0, 0), 
                                                (fftw_complex*)&fftw_buffer_(0, 0), -1, FFTW_MEASURE);
            fftw_plan_with_nthreads(1);
            #else
            fftw_buffer_ = mdarray<double_complex, 2>(size(), Platform::num_fft_threads());

            plan_backward_.resize(Platform::num_fft_threads());
            plan_forward_.resize(Platform::num_fft_threads());

            for (int i = 0; i < Platform::num_fft_threads(); i++)
            {
                plan_backward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                     (fftw_complex*)&fftw_buffer_(0, i), 
                                                     (fftw_complex*)&fftw_buffer_(0, i), 1, FFTW_MEASURE);
                plan_forward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                    (fftw_complex*)&fftw_buffer_(0, i), 
                                                    (fftw_complex*)&fftw_buffer_(0, i), -1, FFTW_MEASURE);
            }
            #endif
            spl_fft_size_ = splindex<block>(size(), comm__.size(), comm__.rank());
        }

        ~FFT3D()
        {
            #ifdef _FFTW_THREADED_
            fftw_destroy_plan(plan_backward_[0]);
            fftw_destroy_plan(plan_forward_[0]);
            #else
            for (int i = 0; i < Platform::num_fft_threads(); i++)
            {
                fftw_destroy_plan(plan_backward_[i]);
                fftw_destroy_plan(plan_forward_[i]);
            }
            #endif
        }

        /// Zero the input buffer for a given thread.
        inline void zero(int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            memset(&fftw_buffer_(0, thread_id), 0, size() * sizeof(double_complex));
        }

        template<typename T>
        inline void input(int n, int* map, T* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());
            
            zero(thread_id);
            for (int i = 0; i < n; i++) fftw_buffer_(map[i], thread_id) = data[i];
        }

        inline void input(double* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());
            
            for (int i = 0; i < size(); i++) fftw_buffer_(i, thread_id) = data[i];
        }
        
        inline void input(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());
            
            memcpy(&fftw_buffer_(0, thread_id), data, size() * sizeof(double_complex));
        }
        
        /// Execute the transformation for a given thread.
        inline void transform(int direction, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            switch(direction)
            {
                case 1:
                {
                    backward(thread_id);
                    break;
                }
                case -1:
                {
                    forward(thread_id);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }

        inline void output(double* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            for (int i = 0; i < size(); i++) data[i] = real(fftw_buffer_(i, thread_id));
        }
        
        inline void output(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            memcpy(data, &fftw_buffer_(0, thread_id), size() * sizeof(double_complex));
        }
        
        inline void output(int n, int* map, double_complex* data, int thread_id = 0)
        {
            assert(thread_id < Platform::num_fft_threads());

            for (int i = 0; i < n; i++) data[i] = fftw_buffer_(map[i], thread_id);
        }

        inline void output(int n, int* map, double_complex* data, int thread_id, double alpha)
        {
            assert(thread_id < Platform::num_fft_threads());

            for (int i = 0; i < n; i++) data[i] += alpha * fftw_buffer_(map[i], thread_id);
        }
        
        inline const std::pair<int, int>& grid_limits(int idim)
        {
            return grid_limits_[idim];
        }

        /// Total size of the FFT grid.
        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        /// Size of a given dimension.
        inline int size(int d)
        {
            assert(d >= 0 && d < 3);
            return grid_size_[d]; 
        }

        /// Return linear index of a plane-wave harmonic with fractional coordinates (10, i1, i2) inside fft buffer.
        inline int index(int i0, int i1, int i2)
        {
            if (i0 < 0) i0 += grid_size_[0];
            if (i1 < 0) i1 += grid_size_[1];
            if (i2 < 0) i2 += grid_size_[2];

            return (i0 + i1 * grid_size_[0] + i2 * grid_size_[0] * grid_size_[1]);
        }

        inline int local_size()
        {
            return static_cast<int>(spl_fft_size_.local_size());
        }

        inline int global_index(int irloc)
        {
            return static_cast<int>(spl_fft_size_[irloc]);
        }

        inline int global_offset()
        {
            return static_cast<int>(spl_fft_size_.global_offset());
        }
        
        /// Direct access to the fft buffer
        inline double_complex& buffer(int i, int thread_id = 0)
        {
            return fftw_buffer_(i, thread_id);
        }
        
        vector3d<int> grid_size()
        {
            return vector3d<int>(grid_size_);
        }
};

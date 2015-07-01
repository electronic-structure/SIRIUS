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
*/
template<> 
class FFT3D<CPU>
{
    private:

        /// Number of working threads inside each FFT.
        int num_fft_workers_;
        
        /// Number of threads doing individual FFTs.
        int num_fft_threads_;

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
        
        int num_gvec_;

        mdarray<int, 2> gvec_;
        std::vector< vector3d<double> > gvec_cart_; // TODO: check if we really need to store it, or create "on the fly"

        mdarray<int, 3> gvec_index_;

        std::vector<int> index_map_;
        std::vector<int> gvec_shell_;
        std::vector<double> gvec_shell_len_;

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

        FFT3D(vector3d<int> dims__,
              int num_fft_threads__,
              int num_fft_workers__)
            : num_fft_workers_(num_fft_workers__),
              num_fft_threads_(num_fft_threads__)
        {
            Timer t("sirius::FFT3D<CPU>::FFT3D");
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(dims__[i]);
                
                grid_limits_[i].second = grid_size_[i] / 2;
                grid_limits_[i].first = grid_limits_[i].second - grid_size_[i] + 1;
            }
            
            fftw_plan_with_nthreads(num_fft_workers_);

            fftw_buffer_ = mdarray<double_complex, 2>(size(), num_fft_threads_);

            plan_backward_.resize(num_fft_threads_);
            plan_forward_.resize(num_fft_threads_);

            for (int i = 0; i < num_fft_threads_; i++)
            {
                plan_backward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                     (fftw_complex*)&fftw_buffer_(0, i), 
                                                     (fftw_complex*)&fftw_buffer_(0, i), 1, FFTW_ESTIMATE);
                plan_forward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                    (fftw_complex*)&fftw_buffer_(0, i), 
                                                    (fftw_complex*)&fftw_buffer_(0, i), -1, FFTW_ESTIMATE);
            }
            fftw_plan_with_nthreads(1);
        }

        ~FFT3D()
        {
            for (int i = 0; i < num_fft_threads_; i++)
            {
                fftw_destroy_plan(plan_backward_[i]);
                fftw_destroy_plan(plan_forward_[i]);
            }
        }

        template<typename T>
        inline void input(int n, int const* map, T* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads());
            
            memset(&fftw_buffer_(0, thread_id), 0, size() * sizeof(double_complex));
            for (int i = 0; i < n; i++) fftw_buffer_(map[i], thread_id) = data[i];
        }

        inline void input(double* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads());
            
            for (int i = 0; i < size(); i++) fftw_buffer_(i, thread_id) = data[i];
        }
        
        inline void input(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads());
            
            memcpy(&fftw_buffer_(0, thread_id), data, size() * sizeof(double_complex));
        }
        
        /// Execute the transformation for a given thread.
        inline void transform(int direction, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads());

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
            assert(thread_id < num_fft_threads());

            for (int i = 0; i < size(); i++) data[i] = real(fftw_buffer_(i, thread_id));
        }
        
        inline void output(double_complex* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads());

            memcpy(data, &fftw_buffer_(0, thread_id), size() * sizeof(double_complex));
        }
        
        inline void output(int n, int const* map, double_complex* data, int thread_id = 0)
        {
            assert(thread_id < num_fft_threads());

            for (int i = 0; i < n; i++) data[i] = fftw_buffer_(map[i], thread_id);
        }

        inline void output(int n, int const* map, double_complex* data, int thread_id, double alpha)
        {
            assert(thread_id < num_fft_threads());

            for (int i = 0; i < n; i++) data[i] += alpha * fftw_buffer_(map[i], thread_id);
        }
        
        inline const std::pair<int, int>& grid_limits(int idim)
        {
            return grid_limits_[idim];
        }

        /// Total size of the FFT grid.
        inline int size() const
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        /// Size of a given dimension.
        inline int size(int d) const
        {
            assert(d >= 0 && d < 3);
            return grid_size_[d]; 
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
            return fftw_buffer_(i, thread_id);
        }
        
        vector3d<int> grid_size() const
        {
            return vector3d<int>(grid_size_);
        }

        void init_gvec(double Gmax__, matrix3d<double> const& M__)
        {
            mdarray<int, 2> gvec_tmp(3, size());
            std::vector< std::pair<double, int> > gvec_tmp_length;
            
            for (int i0 = grid_limits(0).first; i0 <= grid_limits(0).second; i0++)
            {
                for (int i1 = grid_limits(1).first; i1 <= grid_limits(1).second; i1++)
                {
                    for (int i2 = grid_limits(2).first; i2 <= grid_limits(2).second; i2++)
                    {
                        int ig = (int)gvec_tmp_length.size();

                        gvec_tmp(0, ig) = i0;
                        gvec_tmp(1, ig) = i1;
                        gvec_tmp(2, ig) = i2;
                        
                        auto vc = M__ * vector3d<double>(i0, i1, i2);

                        gvec_tmp_length.push_back(std::pair<double, int>(vc.length(), ig));
                    }
                }
            }

            /* sort G-vectors by length */
            std::sort(gvec_tmp_length.begin(), gvec_tmp_length.end());

            /* create sorted list of G-vectors */
            gvec_ = mdarray<int, 2>(3, size());
            gvec_cart_ = std::vector< vector3d<double> >(size());

            /* find number of G-vectors within the cutoff */
            num_gvec_ = 0;
            for (int i = 0; i < size(); i++)
            {
                for (int x = 0; x < 3; x++) gvec_(x, i) = gvec_tmp(x, gvec_tmp_length[i].second);

                gvec_cart_[i] = M__ * vector3d<double>(gvec_(0, i), gvec_(1, i), gvec_(2, i));
                
                if (gvec_tmp_length[i].first <= Gmax__) num_gvec_++;
            }
            
            gvec_index_ = mdarray<int, 3>(mdarray_index_descriptor(grid_limits(0).first, grid_limits(0).second),
                                          mdarray_index_descriptor(grid_limits(1).first, grid_limits(1).second),
                                          mdarray_index_descriptor(grid_limits(2).first, grid_limits(2).second));
            index_map_.resize(size());
            
            gvec_shell_.resize(size());
            gvec_shell_len_.clear();
            
            for (int ig = 0; ig < size(); ig++)
            {
                int i0 = gvec_(0, ig);
                int i1 = gvec_(1, ig);
                int i2 = gvec_(2, ig);

                /* mapping from G-vector to it's index */
                gvec_index_(i0, i1, i2) = ig;

                /* mapping of FFT buffer linear index */
                index_map_[ig] = index(i0, i1, i2);

                /* find G-shells */
                double t = gvec_tmp_length[ig].first;
                if (gvec_shell_len_.empty() || fabs(t - gvec_shell_len_.back()) > 1e-10) gvec_shell_len_.push_back(t);
                gvec_shell_[ig] = (int)gvec_shell_len_.size() - 1;
            }
        }
        
        /// Return number of G-vectors within the cutoff.
        inline int num_gvec() const
        {
            return num_gvec_;
        }

        /// Return G-vector in fractional coordinates (this are the three Miller indices).
        inline vector3d<int> gvec(int ig__) const
        {
            return vector3d<int>(gvec_(0, ig__), gvec_(1, ig__), gvec_(2, ig__));
        }
        
        /// Return G-vector in Cartesian coordinates.
        inline vector3d<double> gvec_cart(int ig__) const
        {
            assert(ig__ >= 0 && ig__ < (int)gvec_cart_.size());
            return gvec_cart_[ig__];
        }

        /// Return length of a G-vector.
        inline double gvec_len(int ig__) const
        {
            return gvec_shell_len(gvec_shell(ig__));
        }

        /// Return number of G-vector shells within the cutoff.
        /** G-vectors with the same length belong to the same shell. */
        inline int num_gvec_shells_inner()
        {
            return gvec_shell(num_gvec_);
        }

        /// Return total number of G-vector shells, including incomplete shells.
        /** Incomplete G-shells are formed by G-vectors outisde the cutoff radius (for example,
         *  by G-vectors at corners of FFT grid).
         */
        inline int num_gvec_shells_total()
        {
            return (int)gvec_shell_len_.size();
        }

        /// Return index of a G-vector shell for a given G-vector.
        inline int gvec_shell(int ig__) const
        {
            assert(ig__ >= 0 && ig__ < (int)gvec_shell_.size());
            return gvec_shell_[ig__];
        }

        /// Return length of a G-vector shell.
        inline double gvec_shell_len(int igsh__) const
        {
            assert(igsh__ >= 0 && igsh__ < (int)gvec_shell_len_.size());
            return gvec_shell_len_[igsh__];
        }

        inline int const* index_map() const
        {
            return &index_map_[0];
        }

        inline int index_map(int ig__) const
        {
            assert(ig__ >= 0 && ig__ < (int)index_map_.size());
            return index_map_[ig__];
        }

        inline int num_fft_threads() const
        {
            return num_fft_threads_;
        }

        inline int gvec_index(vector3d<int> gvec__) const
        {
            return gvec_index_(gvec__[0], gvec__[1], gvec__[2]);
        }
};

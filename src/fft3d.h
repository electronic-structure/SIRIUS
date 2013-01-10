#ifndef __FFT3D_H__
#define __FFT3D_H__

/** \file fft3d.h
    
    \brief Interface to FFTW3 library.
    
    FFT convention:
    \f[
        f({\bf r}) = \sum_{{\bf G}} e^{i{\bf G}{\bf r}} f({\bf G})
    \f]
    is a \em backward transformation from a set of pw coefficients to a function.  

    \f[
        f({\bf G}) = \frac{1}{\Omega} \int e^{-i{\bf G}{\bf r}} f({\bf r}) d {\bf r} = 
            \frac{1}{N} \sum_{{\bf r}_j} e^{-i{\bf G}{\bf r}_j} f({\bf r}_j)
    \f]
    is a \em forward transformation from a function to a set of coefficients. 
*/

namespace sirius
{

class FFT3D
{
    private:

        /// size of each dimension
        int grid_size_[3];

        /// reciprocal space range
        int grid_limits_[3][2];
        
        /// backward transformation plan for each thread
        std::vector<fftw_plan> plan_backward_;
        
        /// forward transformation plan for each thread
        std::vector<fftw_plan> plan_forward_;
    
        /// inout buffer for each thread
        mdarray<complex16, 2> fftw_input_buffer_;
        
        /// output buffer for each thread
        mdarray<complex16, 2> fftw_output_buffer_;

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
            for (int i = 0; i < size(); i++)
                fftw_output_buffer_(i, thread_id) *= norm;
        }

        /// Find smallest optimal grid size starting from n.
        int find_grid_size(int n)
        {
            while (true)
            {
                int m = n;
                for (int k = 2; k <= 5; k++)
                    while (m % k == 0)
                        m /= k;
                if (m == 1) return n;
                else n++;
            }
        } 
        
        /// Determine the optimal FFT grid size and set grid limits.
        void set_grid_size(int* dims)
        {
            for (int i = 0; i < 3; i++)
            {
                grid_size_[i] = find_grid_size(dims[i]);
                
                grid_limits_[i][1] = grid_size_[i] / 2;
                grid_limits_[i][0] = grid_limits_[i][1] - grid_size_[i] + 1;
            }
        }

    public:

        void init(int* dims)
        {
            set_grid_size(dims);
            
            fftw_input_buffer_.set_dimensions(size(), Platform::num_threads());
            fftw_input_buffer_.allocate();

            fftw_output_buffer_.set_dimensions(size(), Platform::num_threads());
            fftw_output_buffer_.allocate();
 
            plan_backward_.resize(Platform::num_threads());
            plan_forward_.resize(Platform::num_threads());

            for (int i = 0; i < Platform::num_threads(); i++)
            {
                plan_backward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                     (fftw_complex*)&fftw_input_buffer_(0, i), 
                                                     (fftw_complex*)&fftw_output_buffer_(0, i), 1, FFTW_MEASURE);
                plan_forward_[i] = fftw_plan_dft_3d(size(2), size(1), size(0), 
                                                    (fftw_complex*)&fftw_input_buffer_(0, i), 
                                                    (fftw_complex*)&fftw_output_buffer_(0, i), -1, FFTW_MEASURE);
            }
        }
        
        void clear()
        {
            for (int i = 0; i < Platform::num_threads(); i++)
            {
                fftw_destroy_plan(plan_backward_[i]);
                fftw_destroy_plan(plan_forward_[i]);
            }
            plan_backward_.clear();
            plan_forward_.clear();
            fftw_input_buffer_.deallocate();
            fftw_output_buffer_.deallocate();
            fftw_cleanup();
        }
        
        inline void zero(int thread_id = 0)
        {
            memset(&fftw_input_buffer_(0, thread_id), 0, size() * sizeof(complex16));
        }

        template<typename T>
        inline void input(int n, int* map, T* data, int thread_id = 0)
        {
            zero(thread_id);

            for (int i = 0; i < n; i++)
                fftw_input_buffer_(map[i], thread_id) = data[i];
        }

        inline void input(double* data, int thread_id = 0)
        {
            for (int i = 0; i < size(); i++)
                fftw_input_buffer_(i, thread_id) = data[i];
        }
        
        inline void input(complex16* data, int thread_id = 0)
        {
            memcpy(&fftw_input_buffer_(0, thread_id), data, size() * sizeof(complex16));
        }

        inline void transform(int direction, int thread_id = 0)
        {
            switch(direction)
            {
                case 1:
                    backward(thread_id);
                    break;

                case -1:
                    forward(thread_id);
                    break;

                default:
                    error(__FILE__, __LINE__, "wrong FFT direction");
            }
        }

        inline void output(double* data, int thread_id = 0)
        {
            for (int i = 0; i < size(); i++)
                data[i] = real(fftw_output_buffer_(i, thread_id));
        }
        
        inline void output(complex16* data, int thread_id = 0)
        {
            memcpy(data, &fftw_output_buffer_(0, thread_id), size() * sizeof(complex16));
        }
        
        inline void output(int n, int* map, complex16* data, int thread_id = 0)
        {
            for (int i = 0; i < n; i++)
                data[i] = fftw_output_buffer_(map[i], thread_id);
        }
        
        inline int grid_limits(int d, int i)
        {
            return grid_limits_[d][i];
        }

        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }
        
        inline int size(int i)
        {
            return grid_size_[i]; 
        }

        inline int index(int i0, int i1, int i2)
        {
            if (i0 < 0) i0 += grid_size_[0];
            if (i1 < 0) i1 += grid_size_[1];
            if (i2 < 0) i2 += grid_size_[2];

            return (i0 + i1 * grid_size_[0] + i2 * grid_size_[0] * grid_size_[1]);
        }
};

};

#endif // __FFT3D_H__

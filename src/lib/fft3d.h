#ifndef __FFT3D_H__
#define __FFT3D_H__

#include "fft3d_base.h"

namespace sirius
{

//#ifdef _FFTW_

class FFT3D : public FFT3D_base
{
    private:
    
        fftw_plan plan_backward;
        
        fftw_plan plan_forward;
        
        std::vector<complex16> fftw_input_buffer;
        
        std::vector<complex16> fftw_output_buffer;
    
    public:

        void init(int* n)
        {
            FFT3D_base::init(n);
            
            clear();
            
            fftw_input_buffer.resize(size());
            fftw_output_buffer.resize(size());
            plan_backward = fftw_plan_dft_3d(size(2), size(1), size(0), (fftw_complex*)&fftw_input_buffer[0], 
                (fftw_complex*)&fftw_output_buffer[0], 1, FFTW_MEASURE);
            plan_forward = fftw_plan_dft_3d(size(2), size(1), size(0), (fftw_complex*)&fftw_input_buffer[0], 
                (fftw_complex*)&fftw_output_buffer[0], -1, FFTW_MEASURE);
        }
        
        void clear()
        {
            fftw_destroy_plan(plan_backward);
            fftw_destroy_plan(plan_forward);
        }
        
        inline void zero()
        {
            memset(&fftw_input_buffer[0], 0, size() * sizeof(complex16));
        }

        template<typename T>
        inline void input(int n, int* map, T* data)
        {
            zero();

            for (int i = 0; i < n; i++)
                fftw_input_buffer[map[i]] = data[i];
        }

        inline void input(double* data)
        {
            for (int i = 0; i < size(); i++)
                fftw_input_buffer[i] = data[i];
        }
        
        inline void input(complex16* data)
        {
            memcpy(&fftw_input_buffer[0], data, size() * sizeof(complex16));
        }

        inline void backward()
        {    
            fftw_execute(plan_backward);
        }
        
        inline void forward()
        {    
            fftw_execute(plan_forward);
            double norm = 1.0 / size();
            for (int i = 0; i < size(); i++)
                fftw_output_buffer[i] *= norm;
        }

        inline void transform(int direction)
        {
            switch(direction)
            {
                case 1:
                    backward();
                    break;

                case -1:
                    forward();
                    break;

                default:
                    error(__FILE__, __LINE__, "wrong FFT direction");
            }
        }

        inline void output(double* data)
        {
            for (int i = 0; i < size(); i++)
                data[i] = real(fftw_output_buffer[i]);
        }
        
        inline void output(complex16* data)
        {
            memcpy(data, &fftw_output_buffer[0], size() * sizeof(complex16));
        }
        
        inline void output(int n, int* map, complex16* data)
        {
            for (int i = 0; i < n; i++)
                data[i] = fftw_output_buffer[map[i]];
        }

};

//#else

/*class FFT3D : public FFT3D_base
{    
    private:
    
        std::vector<gsl_fft_complex_wavetable*> wavetable_; 
        std::vector<gsl_fft_complex_workspace*> workspace_;
        
    public:

        void init(int* n)
        {
            FFT3D_base::init(n);
            
            clear();
            
            for (int i = 0; i < 3; i++) 
            {
                wavetable_.push_back(gsl_fft_complex_wavetable_alloc((int)grid_size_[i]));
                workspace_.push_back(gsl_fft_complex_workspace_alloc((int)grid_size_[i]));
            }
        }
        
        void clear()
        {
            for (int i = 0; i < (int)wavetable_.size(); i++)
            {
                gsl_fft_complex_wavetable_free(wavetable_[i]);
                gsl_fft_complex_workspace_free(workspace_[i]);
            }
            
            wavetable_.clear();
            workspace_.clear();
        }
        
        void transform(complex16* zdata_, int direction)
        {
            double* data_ = (double*)zdata_;

            mdarray<double,4> data(data_, 2, grid_size_[0], grid_size_[1], grid_size_[2]);
            
            gsl_fft_direction sign;
            
            switch(direction)
            {
                case 1:
                    sign = backward;
                    break;
                    
                case -1:
                    sign = forward;
                    break;
                    
                default:
                    error(__FILE__, __LINE__, "wrong fft direction");
            }
            
            for (int k = 0; k < grid_size_[2]; k++)
                for (int j = 0; j < grid_size_[1]; j++)
                    gsl_fft_complex_transform (&data(0, 0, j, k), 1, grid_size_[0], wavetable_[0], workspace_[0], sign);
            
            for (int k = 0; k < grid_size_[2]; k++)
                for (int i = 0; i < grid_size_[0]; i++)
                    gsl_fft_complex_transform (&data(0, i, 0, k), grid_size_[0], grid_size_[1], wavetable_[1], workspace_[1], sign);
                    
            for (int j = 0; j < grid_size_[1]; j++)
                for (int i = 0; i < grid_size_[0]; i++)
                    gsl_fft_complex_transform (&data(0, i, j, 0), grid_size_[0] * grid_size_[1], grid_size_[2], wavetable_[2], workspace_[2], sign);

            if (direction == -1)
            {
                double norm = 1.0 / size();
                for (int i = 0; i < 2 * size(); i++)
                    data_[i] *= norm;
            }
        }
};
*/
//#endif

};

#endif // __FFT3D_H__

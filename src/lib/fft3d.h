#include "fft3d_base.h"

namespace sirius
{

#ifdef _FFTW_

class FFT3D : public FFT3D_base
{
    private:
    
        fftw_plan plan_backward;
        
        fftw_plan plan_forward;
        
        std::vector<complex16> fftw_input_buffer;
        
        std::vector<complex16> fftw_output_buffer;
    
    public:

        void init()
        {
            fftw_input_buffer.resize(size());
            fftw_output_buffer.resize(size());
            plan_backward = fftw_plan_dft_3d(size(0), size(1), size(2), (fftw_complex*)&fftw_input_buffer[0], 
                (fftw_complex*)&fftw_output_buffer[0], 1, FFTW_ESTIMATE);
            plan_forward = fftw_plan_dft_3d(size(0), size(1), size(2), (fftw_complex*)&fftw_input_buffer[0], 
                (fftw_complex*)&fftw_output_buffer[0], -1, FFTW_ESTIMATE);
        }
        
        void transform(double* data_, int direction)
        {
            if (direction == 1)
            {
                memcpy(&fftw_input_buffer[0], data_, size() * sizeof(double) * 2);
                fftw_execute(plan_backward);
                memcpy(data_, &fftw_output_buffer[0], size() * sizeof(double) * 2);
            }
            if (direction == -1)
            {
                memcpy(&fftw_input_buffer[0], data_, size() * sizeof(double) * 2);
                fftw_execute(plan_forward);
                memcpy(data_, &fftw_output_buffer[0], size() * sizeof(double) * 2);
                
                double norm = 1.0 / size();
                for (int i = 0; i < 2 * size(); i++)
                    data_[i] *= norm;
            }
        }
};

#else

class FFT3D : public FFT3D_base
{    
    private:
    
        std::vector<gsl_fft_complex_wavetable*> wavetable_; 
        std::vector<gsl_fft_complex_workspace*> workspace_;
        
    public:

        void init()
        {
            for (int i = 0; i < 3; i++) 
                wavetable_.push_back(gsl_fft_complex_wavetable_alloc((int)grid_size_[i]));
                
            for (int i = 0; i < 3; i++)
                workspace_.push_back(gsl_fft_complex_workspace_alloc((int)grid_size_[i]));
        }
        
        void transform(double* data_, int direction)
        {
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

#endif

};



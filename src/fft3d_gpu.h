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

/** \file fft3d_gpu.h
 *   
 *  \brief Contains GPU specialization.
 */

namespace sirius {

/// GPU wrapper for FFT3D.
class FFT3D_GPU
{
    private:
        
        cufftHandle plan_;
        
        size_t work_size_;

        vector3d<int> grid_size_;

        int num_fft_;

    public:

        FFT3D(vector3d<int> grid_size__, int num_fft__) 
            : grid_size_(grid_size__),
              num_fft_(num_fft__)
        {
            Timer t("sirius::FFT3D_GPU::FFT3D");
            cufft_create_plan_handle(&plan_);
            cufft_create_batch_plan(plan_, grid_size_[0], grid_size_[1], grid_size_[2], num_fft_);
            work_size_ = cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], num_fft_);
        }

        ~FFT3D()
        {
            cufft_destroy_plan_handle(plan_);
        }
        
        //== /** Maximum number of simultaneous FFTs that can fit into free memory of a GPU */
        //== int num_fft_max(size_t free_mem)
        //== {
        //==     int nfft = 0;
        //==     while (cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], nfft + 1) < free_mem) nfft++;
        //==     return nfft;
        //== }

        inline size_t work_area_size()
        {
            //return cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], nfft__);
            return work_size_;
        }

        //inline void initialize(int num_fft__, void* work_area__)
        //{
        //    num_fft_ = num_fft__;
        //    cufft_create_batch_plan(plan_, grid_size_[0], grid_size_[1], grid_size_[2], num_fft_);
        //    cufft_set_work_area(plan_, work_area__);
        //}

        inline void set_work_area_ptr(void* ptr__)
        {
            cufft_set_work_area(plan_, ptr__);
        }

        inline void batch_load(int num_pw_components__, int* map__, double_complex* data__, double_complex* fft_buffer__)
        {
            cufft_batch_load_gpu(size(), num_pw_components__, num_fft_, map__, data__, fft_buffer__);
        }

        inline void batch_unload(int num_pw_components__, int* map__, double_complex* fft_buffer__, double_complex* data__, double beta__)
        {
            cufft_batch_unload_gpu(size(), num_pw_components__, num_fft_, map__, fft_buffer__, data__, beta__);
        }

        inline void transform(int direction__, double_complex* fft_buffer__)
        {
            switch (direction__)
            {
                case 1:
                {
                    cufft_backward_transform(plan_, fft_buffer__);
                    break;
                }
                case -1:
                {
                    cufft_forward_transform(plan_, fft_buffer__);
                    break;
                }
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong FFT direction");
                }
            }
        }
        
        /// Total size of the FFT grid.
        inline int size()
        {
            return grid_size_[0] * grid_size_[1] * grid_size_[2]; 
        }

        /// Size of a given dimension.
        inline int size(int d)
        {
            return grid_size_[d]; 
        }

        inline int num_fft()
        {
            return num_fft_;
        }
};

};

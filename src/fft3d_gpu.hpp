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

/** \file fft3d_gpu.hpp
 *   
 *  \brief Contains GPU specialization.
 */

/// GPU specialization of FFT3D class.
template<> 
class FFT3D<gpu>
{
    private:

        vector3d<int> grid_size_;

    public:

        FFT3D(vector3d<int> grid_size__) : grid_size_(grid_size__)
        {
            cufft_create_plan_handle();
        }
        
        /** Maximum number of simultaneous FFTs that can fit into free memory of a GPU */
        int num_fft_max(size_t free_mem)
        {
            int nfft = 0;
            while (cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], nfft + 1) < free_mem) nfft++;
            return nfft;
        }

        inline size_t work_area_size(int nfft)
        {
            return cufft_get_size(grid_size_[0], grid_size_[1], grid_size_[2], nfft);
        }

        inline void initialize(int num_fft, void* work_area)
        {
            cufft_create_batch_plan(grid_size_[0], grid_size_[1], grid_size_[2], num_fft);
            cufft_set_work_area(work_area);
        }

        inline void finalize()
        {
            cufft_destroy_batch_plan();
        }

        inline void batch_load(int num_elements, int* map, void* data, void* fft_buffer)
        {
            cufft_batch_load_gpu(num_elements, map, data, fft_buffer);
        }

        inline void batch_unload(int num_elements, int* map, void* fft_buffer, void* data)
        {
            cufft_batch_unload_gpu(num_elements, map, fft_buffer, data);
        }

        inline void transform(int direction, void* fft_buffer)
        {
            switch (direction)
            {
                case 1:
                {
                    cufft_backward_transform(fft_buffer);
                    break;
                }
                case -1:
                {
                    cufft_forward_transform(fft_buffer);
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
};

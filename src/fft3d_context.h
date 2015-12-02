// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file fft3d_context.h
 *   
 *  \brief Contains declaration and implementation of sirius::FFT3D_context class.
 */

#ifndef __FFT3D_CONTEXT_H__
#define __FFT3D_CONTEXT_H__

namespace sirius {

/// Manages multiple FFTs.
class FFT3D_context
{
    private:

        processing_unit_t pu_;

        FFT3D_grid fft_grid_;
        
        /// Number of independent FFTs.
        int num_fft_streams_;
        
        /// Number of threads for each independent FFT.
        int num_threads_fft_;

        /// FFT wrapper for dense grid.
        std::vector<FFT3D*> fft_;

    public:

        FFT3D_context(MPI_grid const& mpi_grid__,
                      FFT3D_grid& fft_grid__,
                      int num_fft_streams__,
                      int num_threads_fft__,
                      processing_unit_t pu__,
                      double gpu_workload__ = 0.8)
            : pu_(pu__),
              fft_grid_(fft_grid__),
              num_fft_streams_(num_fft_streams__),
              num_threads_fft_(num_threads_fft__)
        {
            bool parallel_fft = (mpi_grid__.dimension_size(1) > 1);

            if (parallel_fft)
            {
                num_threads_fft_ *= num_fft_streams_;
                num_fft_streams_ = 1;
            }

            for (int tid = 0; tid < num_fft_streams_; tid++)
            {
                processing_unit_t pu = (tid == 0) ? pu_ : CPU;
                fft_.push_back(new FFT3D(fft_grid__, num_threads_fft_, mpi_grid__.communicator(1 << 1), pu, gpu_workload__));
            }
        }

        ~FFT3D_context()
        {
            for (auto obj: fft_) delete obj;
        }

        inline int num_fft_streams() const
        {
            return num_fft_streams_;
        }

        inline FFT3D* fft() const
        {
            return fft_[0];
        }

        inline FFT3D* fft(int idx__) const
        {
            return fft_[idx__];
        }

        inline FFT3D_grid const& fft_grid() const
        {
            return fft_grid_;
        }

        void allocate_workspace(Gvec const& gvec__)
        {
            for (auto obj: fft_) obj->allocate_workspace(gvec__);
        }

        void deallocate_workspace()
        {
            for (auto obj: fft_) obj->deallocate_workspace();
        }

        inline processing_unit_t pu() const
        {
            return pu_;
        }
};

};

#endif

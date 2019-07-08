// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file residuals_aux.cu
 *
 *  \brief CUDA kernel to compute wave-function residuals on GPUs.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

__global__ void compute_residuals_gpu_kernel
(
    int const num_rows_loc__,
    double const* eval__,
    acc_complex_double_t const* hpsi__,
    acc_complex_double_t const* opsi__,
    acc_complex_double_t* res__
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        /* res = hpsi_j - e_j * opsi_j */
        res__[k] = accCsub(hpsi__[k], make_accDoubleComplex(opsi__[k].x * eval__[ibnd], opsi__[k].y * eval__[ibnd]));
    }
}

//== __global__ void compute_residuals_norm_gpu_kernel
//== (
//==     int num_gkvec_row,
//==     int* res_idx,
//==     acc_complex_double_t const* res,
//==     double* res_norm,
//==     int reduced,
//==     int mpi_rank
//== )
//== {
//==     int N = num_blocks(num_gkvec_row, blockDim.x);
//== 
//==     ACC_DYNAMIC_SHARED( char, sdata_ptr)
//==     double* sdata = (double*)&sdata_ptr[0];
//== 
//==     sdata[threadIdx.x] = 0.0;
//== 
//==     for (int n = 0; n < N; n++)
//==     {
//==         int igk = n * blockDim.x + threadIdx.x;
//==         if (igk < num_gkvec_row)
//==         {
//==             int k = array2D_offset(igk, blockIdx.x, num_gkvec_row);
//==             sdata[threadIdx.x] += res[k].x * res[k].x + res[k].y * res[k].y;
//==         }
//==     }
//==     __syncthreads();
//== 
//==     for (int s = 1; s < blockDim.x; s *= 2)
//==     {
//==         if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
//==         __syncthreads();
//==     }
//== 
//==     if (!reduced)
//==     {
//==         res_norm[res_idx[blockIdx.x]] = sdata[0];
//==     }
//==     else
//==     {
//==         if (mpi_rank == 0)
//==         {
//==             double x = res[array2D_offset(0, blockIdx.x, num_gkvec_row)].x;
//==             res_norm[res_idx[blockIdx.x]] = 2 * sdata[0] - x * x;
//==         }
//==         else
//==         {
//==             res_norm[res_idx[blockIdx.x]] = 2 * sdata[0];
//==         }
//==     }
//== }
//== 
//== extern "C" void residuals_aux_gpu(int num_gvec_loc__,
//==                                   int num_res_local__,
//==                                   int* res_idx__,
//==                                   double* eval__,
//==                                   acc_complex_double_t const* hpsi__,
//==                                   acc_complex_double_t const* opsi__,
//==                                   double const* h_diag__,
//==                                   double const* o_diag__,
//==                                   acc_complex_double_t* res__,
//==                                   double* res_norm__,
//==                                   double* p_norm__,
//==                                   int gkvec_reduced__,
//==                                   int mpi_rank__)
//== {
//==     dim3 grid_t(64);
//==     dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_res_local__);
//== 
//==     compute_residuals_gpu_kernel <<<grid_b, grid_t>>>
//==     (
//==         num_gvec_loc__,
//==         eval__,
//==         hpsi__,
//==         opsi__,
//==         res__
//==     );
//== 
//==     grid_b = dim3(num_res_local__);
//== 
//==     compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
//==     (
//==         num_gvec_loc__,
//==         res_idx__,
//==         res__,
//==         res_norm__,
//==         gkvec_reduced__,
//==         mpi_rank__
//==     );
//== 
//==     grid_b = dim3(num_blocks(num_gvec_loc__, grid_t.x), num_res_local__);
//== 
//==     apply_preconditioner_gpu_kernel <<<grid_b, grid_t>>>
//==     (
//==         num_gvec_loc__,
//==         res_idx__,
//==         eval__,
//==         h_diag__,
//==         o_diag__,
//==         res__
//==     );
//== 
//==     grid_b = dim3(num_res_local__);
//== 
//==     compute_residuals_norm_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
//==     (
//==         num_gvec_loc__,
//==         res_idx__,
//==         res__,
//==         p_norm__,
//==         gkvec_reduced__,
//==         mpi_rank__
//==     );
//== }

extern "C" void compute_residuals_gpu(acc_complex_double_t* hpsi__,
                                      acc_complex_double_t* opsi__,
                                      acc_complex_double_t* res__,
                                      int num_rows_loc__,
                                      int num_bands__,
                                      double* eval__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((compute_residuals_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_rows_loc__,
        eval__,
        hpsi__,
        opsi__,
        res__
    );
}

__global__ void add_square_sum_gpu_kernel
(
    int num_rows_loc__,
    acc_complex_double_t const* wf__,
    int reduced__,
    int mpi_rank__,
    double* result__
)
{
    int N = num_blocks(num_rows_loc__, blockDim.x);

    ACC_DYNAMIC_SHARED( char, sdata_ptr)
    double* sdata = (double*)&sdata_ptr[0];

    sdata[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++) {
        int j = n * blockDim.x + threadIdx.x;
        if (j < num_rows_loc__) {
            int k = array2D_offset(j, blockIdx.x, num_rows_loc__);
            sdata[threadIdx.x] += (wf__[k].x * wf__[k].x + wf__[k].y * wf__[k].y);
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (!reduced__) {
            result__[blockIdx.x] += sdata[0];
        } else {
            if (mpi_rank__ == 0) {
                double x = wf__[array2D_offset(0, blockIdx.x, num_rows_loc__)].x;
                result__[blockIdx.x] += (2 * sdata[0] - x * x);
            }
            else {
                result__[blockIdx.x] += 2 * sdata[0];
            }
        }
    }
}

extern "C" void add_square_sum_gpu(acc_complex_double_t* wf__,
                                   int num_rows_loc__,
                                   int nwf__,
                                   int reduced__,
                                   int mpi_rank__,
                                   double* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    accLaunchKernel((add_square_sum_gpu_kernel), dim3(grid_b), dim3(grid_t), grid_t.x * sizeof(double), 0, 
        num_rows_loc__,
        wf__,
        reduced__,
        mpi_rank__,
        result__
    );
}

__global__ void apply_preconditioner_gpu_kernel(int const num_rows_loc__,
                                                double const* eval__,
                                                double const* h_diag__,
                                                double const* o_diag__,
                                                acc_complex_double_t* res__)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        double p = (h_diag__[j] - eval__[ibnd] * o_diag__[j]);
        p = 0.5 * (1 + p + sqrt(1.0 + (p - 1) * (p - 1)));
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        res__[k] = make_accDoubleComplex(res__[k].x / p, res__[k].y / p);
    }
}

extern "C" void apply_preconditioner_gpu(acc_complex_double_t* res__,
                                         int num_rows_loc__,
                                         int num_bands__,
                                         double* eval__,
                                         const double* h_diag__,
                                         const double* o_diag__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((apply_preconditioner_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, num_rows_loc__, eval__, h_diag__, o_diag__, res__);
}

__global__ void make_real_g0_gpu_kernel(acc_complex_double_t* res__,
                                        int              ld__)
{
    acc_complex_double_t z = res__[array2D_offset(0, blockIdx.x, ld__)];
    if (threadIdx.x == 0) {
        res__[array2D_offset(0, blockIdx.x, ld__)] = make_accDoubleComplex(z.x, 0);
    }
}

extern "C" void make_real_g0_gpu(acc_complex_double_t* res__,
                                 int              ld__,
                                 int              n__)
{
    dim3 grid_t(32);
    dim3 grid_b(n__);

    accLaunchKernel((make_real_g0_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, res__, ld__);
}























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

/** \file cuda_uspp_kernels.cu
 *
 *  \brief CUDA kernel for the PW-PW method.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "hip/hip_runtime.h"
#include "hip/hip_complex.h"

extern hipStream_t* streams;

__global__ void compute_chebyshev_order1_gpu_kernel
(
    int num_gkvec__,
    double c__,
    double r__,
    hipDoubleComplex* phi0__,
    hipDoubleComplex* phi1__
)
{
    int igk = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;

    if (igk < num_gkvec__)
    {
        int i = array2D_offset(igk, j, num_gkvec__);
        // phi0 * c
        hipDoubleComplex z1 = hipCmul(phi0__[i], make_hipDoubleComplex(c__, 0));
        // phi1 - phi0 * c
        hipDoubleComplex z2 = hipCsub(phi1__[i], z1);
        // (phi1 - phi0 * c) / r
        phi1__[i] = hipCdiv(z2, make_hipDoubleComplex(r__, 0));
    }
}

__global__ void compute_chebyshev_orderk_gpu_kernel
(
    int num_gkvec__,
    double c__,
    double r__,
    hipDoubleComplex* phi0__,
    hipDoubleComplex* phi1__,
    hipDoubleComplex* phi2__
)
{
    int igk = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;

    if (igk < num_gkvec__)
    {
        int i = array2D_offset(igk, j, num_gkvec__);
        // phi1 * c
        hipDoubleComplex z1 = hipCmul(phi1__[i], make_hipDoubleComplex(c__, 0));
        // phi2 - phi1 * c
        hipDoubleComplex z2 = hipCsub(phi2__[i], z1);
        // (phi2 - phi1 * c) * 2 / r
        hipDoubleComplex z3 = hipCmul(z2, make_hipDoubleComplex(2.0 / r__, 0));
        // (phi2 - phi1 * c) * 2 / r - phi0
        phi2__[i] = hipCsub(z3, phi0__[i]);
    }
}

extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 hipDoubleComplex* phi0,
                                                 hipDoubleComplex* phi1,
                                                 hipDoubleComplex* phi2)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), n);

    if (phi2 == NULL)
    {
        hipLaunchKernelGGL((compute_chebyshev_order1_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
            num_gkvec,
            c,
            r,
            phi0,
            phi1
        );
    }
    else
    {
        hipLaunchKernelGGL((compute_chebyshev_orderk_gpu_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
            num_gkvec,
            c,
            r,
            phi0,
            phi1,
            phi2
        );
    }
}


//== #define BLOCK_SIZE 32
//== 
//== __global__ void generate_beta_phi_gpu_kernel(int num_gkvec, 
//==                                              int num_beta,
//==                                              int num_phi,
//==                                              int* beta_t_idx, 
//==                                              double* atom_pos, 
//==                                              double* gkvec, 
//==                                              hipDoubleComplex* beta_pw_type,
//==                                              hipDoubleComplex* phi,
//==                                              hipDoubleComplex* beta_phi)
//== {
//==     int idx_beta = blockDim.x * blockIdx.x + threadIdx.x;
//==     int idx_phi = blockDim.y * blockIdx.y + threadIdx.y;
//==     int ia, offset_t;
//==     double x0, y0, z0;
//== 
//==     if (idx_beta < num_beta)
//==     {
//==         ia = beta_t_idx[array2D_offset(0, idx_beta, 2)];
//==         offset_t = beta_t_idx[array2D_offset(1, idx_beta, 2)];
//==         x0 = atom_pos[array2D_offset(0, ia, 3)];
//==         y0 = atom_pos[array2D_offset(1, ia, 3)];
//==         z0 = atom_pos[array2D_offset(2, ia, 3)];
//==     }
//== 
//==     int N = num_blocks(num_gkvec, BLOCK_SIZE);
//== 
//==     hipDoubleComplex val = make_hipDoubleComplex(0.0, 0.0);
//== 
//==     for (int m = 0; m < N; m++)
//==     {
//==         __shared__ hipDoubleComplex beta_pw_tile[BLOCK_SIZE][BLOCK_SIZE];
//==         __shared__ hipDoubleComplex phi_tile[BLOCK_SIZE][BLOCK_SIZE];
//== 
//==         int bs = (m + 1) * BLOCK_SIZE > num_gkvec ? num_gkvec - m * BLOCK_SIZE : BLOCK_SIZE;
//== 
//==         int igk = m * BLOCK_SIZE + threadIdx.y;
//== 
//==         if (igk < num_gkvec && idx_beta < num_beta)
//==         {
//==             double x1 = gkvec[array2D_offset(igk, 0, num_gkvec)];
//==             double y1 = gkvec[array2D_offset(igk, 1, num_gkvec)];
//==             double z1 = gkvec[array2D_offset(igk, 2, num_gkvec)];
//== 
//==             double p = twopi * (x0 * x1 + y0 * y1 + z0 * z1);
//==             double sinp = sin(p);
//==             double cosp = cos(p);
//== 
//==             beta_pw_tile[threadIdx.x][threadIdx.y] = hipCmul(hipConj(beta_pw_type[array2D_offset(igk, offset_t, num_gkvec)]), 
//==                                                             make_hipDoubleComplex(cosp, sinp));
//== 
//==         }
//==         
//==         igk = m * BLOCK_SIZE + threadIdx.x;
//== 
//==         if (igk < num_gkvec && idx_phi < num_phi)
//==             phi_tile[threadIdx.y][threadIdx.x] = phi[array2D_offset(igk, idx_phi, num_gkvec)];
//== 
//==         __syncthreads();
//== 
//==         for (int i = 0; i < bs; i++) val = hipCadd(val, hipCmul(beta_pw_tile[threadIdx.x][i], phi_tile[threadIdx.y][i]));
//== 
//==         __syncthreads();
//==     }
//== 
//==     if (idx_beta < num_beta && idx_phi < num_phi) beta_phi[array2D_offset(idx_beta, idx_phi, num_beta)] = val;
//== }
//== 
//== 
//== extern "C" void generate_beta_phi_gpu(int num_gkvec, 
//==                                       int num_beta, 
//==                                       int num_phi, 
//==                                       int* beta_t_idx, 
//==                                       double* atom_pos,
//==                                       double* gkvec,
//==                                       void* beta_pw_type,
//==                                       void* phi,
//==                                       void* beta_phi)
//== {
//== 
//==     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
//==     dim3 numBlocks(num_blocks(num_beta, BLOCK_SIZE), num_blocks(num_phi, BLOCK_SIZE));
//== 
//==     hipLaunchKernelGGL((generate_beta_phi_gpu_kernel), dim3(//==         numBlocks), dim3(//==         threadsPerBlock), 0, 0, num_gkvec, 
//==                            num_beta,
//==                            num_phi,
//==                            beta_t_idx, 
//==                            atom_pos,
//==                            gkvec, 
//==                            (hipDoubleComplex*)beta_pw_type,
//==                            (hipDoubleComplex*)phi,
//==                            (hipDoubleComplex*)beta_phi);
//== }




//__global__ void copy_beta_psi_gpu_kernel
//(
//    hipDoubleComplex const* beta_psi,
//    int beta_psi_ld, 
//    double const* wo,
//    hipDoubleComplex* beta_psi_wo,
//    int beta_psi_wo_ld
//)
//{
//    int xi = threadIdx.x;
//    int j = blockIdx.x;
//
//    beta_psi_wo[array2D_offset(xi, j, beta_psi_wo_ld)] = hipCmul(hipConj(beta_psi[array2D_offset(xi, j, beta_psi_ld)]),
//                                                                make_hipDoubleComplex(wo[j], 0.0));
//}

//extern "C" void copy_beta_psi_gpu(int nbf,
//                                  int nloc,
//                                  hipDoubleComplex const* beta_psi,
//                                  int beta_psi_ld,
//                                  double const* wo,
//                                  hipDoubleComplex* beta_psi_wo,
//                                  int beta_psi_wo_ld,
//                                  int stream_id)
//{
//    dim3 grid_t(nbf);
//    dim3 grid_b(nloc);
//    
//    hipStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
//    
//    copy_beta_psi_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
//    (
//        beta_psi,
//        beta_psi_ld,
//        wo,
//        beta_psi_wo,
//        beta_psi_wo_ld
//    );
//}

__global__ void compute_inner_product_gpu_kernel
(
    int num_gkvec_row,
    hipDoubleComplex const* f1,
    hipDoubleComplex const* f2,
    double* prod
)
{
    int N = num_blocks(num_gkvec_row, blockDim.x);

    HIP_DYNAMIC_SHARED( char, sdata_ptr)
    double* sdata = (double*)&sdata_ptr[0];

    sdata[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++)
    {
        int igk = n * blockDim.x + threadIdx.x;
        if (igk < num_gkvec_row)
        {
            int k = array2D_offset(igk, blockIdx.x, num_gkvec_row);
            sdata[threadIdx.x] += f1[k].x * f2[k].x + f1[k].y *f2[k].y;
        }
    }

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = sdata[threadIdx.x] + sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    prod[blockIdx.x] = sdata[0];
}

extern "C" void compute_inner_product_gpu(int num_gkvec_row,
                                          int n,
                                          hipDoubleComplex const* f1,
                                          hipDoubleComplex const* f2,
                                          double* prod)
{
    dim3 grid_t(64);
    dim3 grid_b(n);

    hipLaunchKernelGGL((compute_inner_product_gpu_kernel), dim3(grid_b), dim3(grid_t), grid_t.x * sizeof(double), 0, 
        num_gkvec_row,
        f1,
        f2,
        prod
    );
}


__global__ void add_checksum_gpu_kernel
(
    hipDoubleComplex const* wf__,
    int num_rows_loc__,
    hipDoubleComplex* result__
)
{
    int N = num_blocks(num_rows_loc__, blockDim.x);

    HIP_DYNAMIC_SHARED( char, sdata_ptr)
    double* sdata_x = (double*)&sdata_ptr[0];
    double* sdata_y = (double*)&sdata_ptr[blockDim.x * sizeof(double)];

    sdata_x[threadIdx.x] = 0.0;
    sdata_y[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++) {
        int j = n * blockDim.x + threadIdx.x;
        if (j < num_rows_loc__) {
            int k = array2D_offset(j, blockIdx.x, num_rows_loc__);
            sdata_x[threadIdx.x] += wf__[k].x;
            sdata_y[threadIdx.x] += wf__[k].y;
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata_x[threadIdx.x] = sdata_x[threadIdx.x] + sdata_x[threadIdx.x + s];
            sdata_y[threadIdx.x] = sdata_y[threadIdx.x] + sdata_y[threadIdx.x + s];
        }
        __syncthreads();
    }

    result__[blockIdx.x] = hipCadd(result__[blockIdx.x], make_hipDoubleComplex(sdata_x[0], sdata_y[0]));
}

extern "C" void add_checksum_gpu(hipDoubleComplex* wf__,
                                 int num_rows_loc__,
                                 int nwf__,
                                 hipDoubleComplex* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    hipLaunchKernelGGL((add_checksum_gpu_kernel), dim3(grid_b), dim3(grid_t), 2 * grid_t.x * sizeof(double), 0, 
        wf__,
        num_rows_loc__,
        result__
    );
}

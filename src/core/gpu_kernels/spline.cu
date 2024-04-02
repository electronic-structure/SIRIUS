/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file spline.cu
 *
 *  \brief CUDA kernels to perform operations on splines.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

using namespace sirius;
using namespace sirius::acc;

__global__ void spline_inner_product_gpu_kernel_v3(int num_points__,
                                                   int const* idx_ri__,
                                                   double const* x__,
                                                   double const* dx__,
                                                   double const* f__,
                                                   double const* g__,
                                                   double* result__)
{
    int nb = num_blocks(num_points__, blockDim.x);
    int idx_f = idx_ri__[array2D_offset(0, blockIdx.x, 2)];
    int idx_g = idx_ri__[array2D_offset(1, blockIdx.x, 2)];

    ACC_DYNAMIC_SHARED( char, sdata_ptr)
    double* sdata = (double*)&sdata_ptr[0];

    int a_offs_f = array3D_offset(0, 0, idx_f, num_points__, 4);
    int b_offs_f = array3D_offset(0, 1, idx_f, num_points__, 4);
    int c_offs_f = array3D_offset(0, 2, idx_f, num_points__, 4);
    int d_offs_f = array3D_offset(0, 3, idx_f, num_points__, 4);

    int a_offs_g = array3D_offset(0, 0, idx_g, num_points__, 4);
    int b_offs_g = array3D_offset(0, 1, idx_g, num_points__, 4);
    int c_offs_g = array3D_offset(0, 2, idx_g, num_points__, 4);
    int d_offs_g = array3D_offset(0, 3, idx_g, num_points__, 4);


    sdata[threadIdx.x] = 0;

    for (int ib = 0; ib < nb; ib++)
    {
        int i = ib * blockDim.x + threadIdx.x;
        if (i < num_points__ - 1)
        {
            double xi = x__[i];
            double dxi = dx__[i];

            double a1 = f__[a_offs_f + i];
            double b1 = f__[b_offs_f + i];
            double c1 = f__[c_offs_f + i];
            double d1 = f__[d_offs_f + i];
            
            double a2 = g__[a_offs_g + i];
            double b2 = g__[b_offs_g + i];
            double c2 = g__[c_offs_g + i];
            double d2 = g__[d_offs_g + i];
                
            double k0 = a1 * a2;
            double k1 = d1 * b2 + c1 * c2 + b1 * d2;
            double k2 = d1 * a2 + c1 * b2 + b1 * c2 + a1 * d2;
            double k3 = c1 * a2 + b1 * b2 + a1 * c2;
            double k4 = d1 * c2 + c1 * d2;
            double k5 = b1 * a2 + a1 * b2;
            double k6 = d1 * d2; // 25 flop in total

            //double v1 = dxi * k6 * (1.0 / 9.0);
            //double r = (k4 + 2.0 * k6 * xi) * 0.125;
            //double v2 = dxi * (r + v1);
            //double v3 = dxi * ((k1 + xi * (2.0 * k4 + k6 * xi)) * (1.0 / 7.0) + v2);
            //double v4 = dxi * ((k2 + xi * (2.0 * k1 + k4 * xi)) * (1.0 / 6.0) + v3);
            //double v5 = dxi * ((k3 + xi * (2.0 * k2 + k1 * xi)) * 0.2 + v4);
            //double v6 = dxi * ((k5 + xi * (2.0 * k3 + k2 * xi)) * 0.25 + v5);
            //double v7 = dxi * ((k0 + xi * (2.0 * k5 + k3 * xi)) / 3.0 + v6);
            //double v8 = dxi * ((xi * (2.0 * k0 + xi * k5)) * 0.5 + v7);

            double v = dxi * k6 * 0.11111111111111111111;
            
            double r1 = k4 * 0.125 + k6 * xi * 0.25;
            v = dxi * (r1 + v);

            double r2 = (k1 + xi * (2.0 * k4 + k6 * xi)) * 0.14285714285714285714;
            v = dxi * (r2 + v);

            double r3 = (k2 + xi * (2.0 * k1 + k4 * xi)) * 0.16666666666666666667;
            v = dxi * (r3 + v);

            double r4 = (k3 + xi * (2.0 * k2 + k1 * xi)) * 0.2;
            v = dxi * (r4 + v);

            double r5 = (k5 + xi * (2.0 * k3 + k2 * xi)) * 0.25;
            v = dxi * (r5 + v); 

            double r6 = (k0 + xi * (2.0 * k5 + k3 * xi)) * 0.33333333333333333333;
            v = dxi * (r6 + v);

            double r7 = (xi * (2.0 * k0 + xi * k5)) * 0.5;
            v = dxi * (r7 + v);

            sdata[threadIdx.x] += dxi * (k0 * xi * xi + v);
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    result__[blockIdx.x] = sdata[0];
}

extern "C" void spline_inner_product_gpu_v3(int const* idx_ri__,
                                            int num_ri__,
                                            int num_points__,
                                            double const* x__,
                                            double const* dx__,
                                            double const* f__, 
                                            double const* g__,
                                            double* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_ri__);

    accLaunchKernel((spline_inner_product_gpu_kernel_v3), dim3(grid_b), dim3(grid_t), grid_t.x * sizeof(double), 0, 
        num_points__,
        idx_ri__,
        x__,
        dx__,
        f__,
        g__,
        result__
    );
}



/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file residuals_aux.cu
 *
 *  \brief CUDA kernel to compute wave-function residuals on GPUs.
 */

#include "core/acc/acc_common.hpp"
#include "core/acc/acc_runtime.hpp"

using namespace sirius;
using namespace sirius::acc;

template <typename T>
__global__ void compute_residuals_gpu_kernel
(
    int const num_rows_loc__,
    T const* eval__,
    gpu_complex_type<T> const* hpsi__,
    gpu_complex_type<T> const* opsi__,
    gpu_complex_type<T>* res__
);

template <>
__global__ void compute_residuals_gpu_kernel<double>
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

template <>
__global__ void compute_residuals_gpu_kernel<float>
    (
        int const num_rows_loc__,
        float const* eval__,
        acc_complex_float_t const* hpsi__,
        acc_complex_float_t const* opsi__,
        acc_complex_float_t* res__
    )
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        /* res = hpsi_j - e_j * opsi_j */
        res__[k] = accCsubf(hpsi__[k], make_accFloatComplex(opsi__[k].x * eval__[ibnd], opsi__[k].y * eval__[ibnd]));
    }
}

extern "C" { 
void compute_residuals_gpu_double(acc_complex_double_t* hpsi__,
                                  acc_complex_double_t* opsi__,
                                  acc_complex_double_t* res__,
                                  int num_rows_loc__,
                                  int num_bands__,
                                  double* eval__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((compute_residuals_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0,
        num_rows_loc__,
        eval__,
        hpsi__,
        opsi__,
        res__
    );
}

void compute_residuals_gpu_float(acc_complex_float_t* hpsi__,
                                 acc_complex_float_t* opsi__,
                                 acc_complex_float_t* res__,
                                 int num_rows_loc__,
                                 int num_bands__,
                                 float* eval__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((compute_residuals_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0,
                    num_rows_loc__,
                    eval__,
                    hpsi__,
                    opsi__,
                    res__
    );
}
}

template <typename T>
__global__ void add_square_sum_gpu_kernel
(
    int num_rows_loc__,
    gpu_complex_type<T> const* wf__,
    int reduced__,
    int mpi_rank__,
    T* result__
)
{
    int N = num_blocks(num_rows_loc__, blockDim.x);

    ACC_DYNAMIC_SHARED( char, sdata_ptr)
    T* sdata = (T*)&sdata_ptr[0];

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
                T x = wf__[array2D_offset(0, blockIdx.x, num_rows_loc__)].x;
                result__[blockIdx.x] += (2 * sdata[0] - x * x);
            }
            else {
                result__[blockIdx.x] += 2 * sdata[0];
            }
        }
    }
}

extern "C" {
void add_square_sum_gpu_double(acc_complex_double_t* wf__,
                               int num_rows_loc__,
                               int nwf__,
                               int reduced__,
                               int mpi_rank__,
                               double* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    accLaunchKernel((add_square_sum_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), grid_t.x * sizeof(double), 0,
                    num_rows_loc__, wf__, reduced__, mpi_rank__, result__);
}

void add_square_sum_gpu_float(acc_complex_float_t* wf__,
                              int num_rows_loc__,
                              int nwf__,
                              int reduced__,
                              int mpi_rank__,
                              float* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    accLaunchKernel((add_square_sum_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), grid_t.x * sizeof(float), 0,
                    num_rows_loc__, wf__, reduced__, mpi_rank__, result__);
}
}

template <typename T, typename F>
static inline __device__ std::enable_if_t<std::is_scalar<F>::value, F>
inner_diag_local_aux(gpu_complex_type<T> z1__, gpu_complex_type<T> z2__)
{
    return z1__.x * z2__.x + z1__.y * z2__.y;
}

/// For complex-type F (complex<double> or complex<float>).
template <typename T, typename F>
static inline __device__ std::enable_if_t<!std::is_scalar<F>::value, F>
inner_diag_local_aux(gpu_complex_type<T> z1__, gpu_complex_type<T> z2__)
{
    return mul_accNumbers(make_accComplex(z1__.x, -z1__.y), z2__);
}


template <typename T, typename F>
__global__ void inner_diag_local_gpu_kernel(gpu_complex_type<T> const* wf1__, int ld1__,
        gpu_complex_type<T> const* wf2__, int ld2__, int ngv_loc__, int reduced__, F* result__)
{
    int N = num_blocks(ngv_loc__, blockDim.x);

    ACC_DYNAMIC_SHARED(char, sdata_ptr)
    F* sdata = (F*)&sdata_ptr[0];

    sdata[threadIdx.x] = accZero<F>();

    for (int i = 0; i < N; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if (j < ngv_loc__) {
            int k1 = array2D_offset(j, blockIdx.x, ld1__);
            int k2 = array2D_offset(j, blockIdx.x, ld2__);
            sdata[threadIdx.x] = add_accNumbers(sdata[threadIdx.x], inner_diag_local_aux<T, F>(wf1__[k1], wf2__[k2]));
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata[threadIdx.x] = add_accNumbers(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (!reduced__) {
            result__[blockIdx.x] = add_accNumbers(result__[blockIdx.x], sdata[0]);
        } else {
            /* compute 2*sdata[0] */
            sdata[0] = add_accNumbers(sdata[0], sdata[0]);
            if (reduced__ == 1) {
                /* for gamma-point case F can only be double or float */
                real_type<F> x = wf1__[array2D_offset(0, blockIdx.x, ld1__)].x * wf2__[array2D_offset(0, blockIdx.x, ld2__)].x;
                /* trick the compiler here */
                F* a = (F*)((void*)&x);
                result__[blockIdx.x] = sub_accNumbers(sdata[0], *a);
            } else { /* reduced > 1 -> all other G-vectors */
                result__[blockIdx.x] = sdata[0];
            }
        }
    }
}


extern "C" {

void inner_diag_local_gpu_double_complex_double(gpu_complex_type<double>* wf1__, int ld1__,
        gpu_complex_type<double>* wf2__, int ld2__, int ngv_loc__, int nwf__,
        gpu_complex_type<double>* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    accLaunchKernel((inner_diag_local_gpu_kernel<double, gpu_complex_type<double>>), dim3(grid_b), dim3(grid_t),
            grid_t.x * sizeof(gpu_complex_type<double>), 0,
            wf1__, ld1__, wf2__, ld2__, ngv_loc__, 0, result__);

}

void inner_diag_local_gpu_double_double(gpu_complex_type<double>* wf1__, int ld1__,
        gpu_complex_type<double>* wf2__, int ld2__, int ngv_loc__, int nwf__, int reduced__,
        double* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    accLaunchKernel((inner_diag_local_gpu_kernel<double, double>), dim3(grid_b), dim3(grid_t),
            grid_t.x * sizeof(double), 0,
            wf1__, ld1__, wf2__, ld2__, ngv_loc__, reduced__, result__);

}

}


/*
inner_diag_local_gpu_float_float
inner_diag_local_gpu_float_double
inner_diag_local_gpu_float_complex_float
inner_diag_local_gpu_float_complex_double
inner_diag_local_gpu_double_double
inner_diag_local_gpu_double_complex_double
*/

template <typename T>
__global__ void apply_preconditioner_gpu_kernel(int const num_rows_loc__,
                                                T const* eval__,
                                                T const* h_diag__,
                                                T const* o_diag__,
                                                gpu_complex_type<T>* res__);

template <>
__global__ void apply_preconditioner_gpu_kernel<double>(int const num_rows_loc__,
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

template <>
__global__ void apply_preconditioner_gpu_kernel<float>(int const num_rows_loc__,
                                                        float const* eval__,
                                                        float const* h_diag__,
                                                        float const* o_diag__,
                                                        acc_complex_float_t* res__)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        float p = (h_diag__[j] - eval__[ibnd] * o_diag__[j]);
        p = 0.5 * (1 + p + sqrt(1.0 + (p - 1) * (p - 1)));
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        res__[k] = make_accFloatComplex(res__[k].x / p, res__[k].y / p);
    }
}

extern "C" {
void apply_preconditioner_gpu_double(acc_complex_double_t* res__,
                                     int num_rows_loc__,
                                     int num_bands__,
                                     double* eval__,
                                     const double* h_diag__,
                                     const double* o_diag__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((apply_preconditioner_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0, num_rows_loc__, eval__, h_diag__, o_diag__, res__);
}

void apply_preconditioner_gpu_float(acc_complex_float_t* res__,
                                    int num_rows_loc__,
                                    int num_bands__,
                                    float* eval__,
                                    const float* h_diag__,
                                    const float* o_diag__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((apply_preconditioner_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0, num_rows_loc__, eval__, h_diag__, o_diag__, res__);
}
}

template <typename T>
__global__ void make_real_g0_gpu_kernel(gpu_complex_type<T>* res__, int ld__);

template <>
__global__ void make_real_g0_gpu_kernel<double>(acc_complex_double_t* res__,
                                                int              ld__)
{
    acc_complex_double_t z = res__[array2D_offset(0, blockIdx.x, ld__)];
    if (threadIdx.x == 0) {
        res__[array2D_offset(0, blockIdx.x, ld__)] = make_accDoubleComplex(z.x, 0);
    }
}

template <>
__global__ void make_real_g0_gpu_kernel<float>(acc_complex_float_t* res__,
                                                int              ld__)
{
    acc_complex_float_t z = res__[array2D_offset(0, blockIdx.x, ld__)];
    if (threadIdx.x == 0) {
        res__[array2D_offset(0, blockIdx.x, ld__)] = make_accFloatComplex(z.x, 0);
    }
}

extern "C" {
void make_real_g0_gpu_double(acc_complex_double_t* res__, int ld__, int n__)
{
    dim3 grid_t(32);
    dim3 grid_b(n__);

    accLaunchKernel((make_real_g0_gpu_kernel<double>), dim3(grid_b), dim3(grid_t), 0, 0, res__, ld__);
}

void make_real_g0_gpu_float(acc_complex_float_t* res__, int ld__, int n__)
{
    dim3 grid_t(32);
    dim3 grid_b(n__);

    accLaunchKernel((make_real_g0_gpu_kernel<float>), dim3(grid_b), dim3(grid_t), 0, 0, res__, ld__);
}
}

template <typename T, typename F>
__global__ void axpby_gpu_kernel(F const* beta__, gpu_complex_type<T>* y__, int ld2__, int ngv_loc__)
{
    /* index of the wave-function coefficient */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    /* idex of the band */
    int ibnd = blockIdx.y;

    if (j < ngv_loc__) {
        int k2 = array2D_offset(j, ibnd, ld2__);
        y__[k2] = mul_accNumbers(beta__[ibnd], y__[k2]);
    }
}

template <typename T, typename F>
__global__ void axpby_gpu_kernel(F const* alpha__, gpu_complex_type<T> const* x__, int ld1__,
        F const* beta__, gpu_complex_type<T>* y__, int ld2__, int ngv_loc__)
{
    /* index of the wave-function coefficient */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    /* idex of the band */
    int ibnd = blockIdx.y;

    auto alpha = alpha__[ibnd];
    auto beta = beta__[ibnd];

    if  (j < ngv_loc__) {
        int k1 = array2D_offset(j, ibnd, ld1__);
        int k2 = array2D_offset(j, ibnd, ld2__);
        if (is_zero(beta)) {
            y__[k2] = mul_accNumbers(alpha, x__[k1]);
        } else if (is_zero(alpha)) {
            y__[k2] = mul_accNumbers(beta, y__[k2]);
        } else {
            y__[k2] = add_accNumbers(mul_accNumbers(alpha, x__[k1]), mul_accNumbers(beta, y__[k2]));
        }
    }
}

template <typename T, typename F>
__global__ void axpy_scatter_gpu_kernel(F const* alpha__, gpu_complex_type<T> const* x__, int ld1__,
int const* idx__, gpu_complex_type<T>* y__, int ld2__, int ngv_loc__)
{
    /* index of wave-function coefficient */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    /* index of the band / unconverged vector */
    int j_unconv = blockIdx.y;

    int jj = idx__[j_unconv];
    auto alpha = alpha__[j_unconv];
    if  (j < ngv_loc__) { // for all valid wf coefficients

        int k1 = array2D_offset(j, j_unconv, ld1__);
        int k2 = array2D_offset(j, jj, ld2__);

        y__[k2] = add_accNumbers(mul_accNumbers(alpha, x__[k1]), y__[k2]);
    }
}

extern "C" {

void axpby_gpu_double_complex_double(int nwf__, gpu_complex_type<double> const* alpha__, gpu_complex_type<double> const* x__, int ld1__,
    gpu_complex_type<double> const* beta__, gpu_complex_type<double>* y__, int ld2__, int ngv_loc__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(ngv_loc__, grid_t.x), nwf__);

    if (x__) {
        accLaunchKernel((axpby_gpu_kernel<double, gpu_complex_type<double>>), dim3(grid_b), dim3(grid_t), 0, 0,
                alpha__, x__, ld1__, beta__, y__, ld2__, ngv_loc__);
    } else {
        accLaunchKernel((axpby_gpu_kernel<double, gpu_complex_type<double>>), dim3(grid_b), dim3(grid_t), 0, 0,
                beta__, y__, ld2__, ngv_loc__);
    }
}


void axpby_gpu_double_double(int nwf__, double const* alpha__, gpu_complex_type<double> const* x__, int ld1__,
    double const* beta__, gpu_complex_type<double>* y__, int ld2__, int ngv_loc__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(ngv_loc__, grid_t.x), nwf__);

    if (x__) {
        accLaunchKernel((axpby_gpu_kernel<double, double>), dim3(grid_b), dim3(grid_t), 0, 0,
                alpha__, x__, ld1__, beta__, y__, ld2__, ngv_loc__);
    } else {
        accLaunchKernel((axpby_gpu_kernel<double, double>), dim3(grid_b), dim3(grid_t), 0, 0,
                beta__, y__, ld2__, ngv_loc__);
    }
}

void axpy_scatter_gpu_double_complex_double(int N_unconverged,
    gpu_complex_type<double> const* alpha__, gpu_complex_type<double> const* x__, int ld1__, int const* idx__,
    gpu_complex_type<double>* y__, int ld2__, int ngv_loc__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(ngv_loc__, grid_t.x), N_unconverged);
    accLaunchKernel((axpy_scatter_gpu_kernel<double, gpu_complex_type<double>>), dim3(grid_b), dim3(grid_t),
                     0, 0, alpha__, x__, ld1__, idx__, y__, ld2__, ngv_loc__);
}

void axpy_scatter_gpu_double_double(int N_unconverged,
    double const* alpha__, gpu_complex_type<double> const* x__, int ld1__, int const* idx__,
    gpu_complex_type<double>* y__, int ld2__, int ngv_loc__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(ngv_loc__, grid_t.x), N_unconverged);
    accLaunchKernel((axpy_scatter_gpu_kernel<double, double>), dim3(grid_b), dim3(grid_t),
                     0, 0, alpha__, x__, ld1__, idx__, y__, ld2__, ngv_loc__);
}

} // end extern "C"

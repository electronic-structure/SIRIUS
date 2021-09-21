/** \file memory_copy.cu
*
*  \brief CUDA kernels to copy array element of different precision
*/

#include "gpu/cuda_common.hpp"
#include "gpu/acc_runtime.hpp"

template <typename T, typename F>
__global__ void copy_kernel(const T* from_ptr__, F* to_ptr__, int n__)
{
    for (int itp = threadIdx.x + blockIdx.x * blockDim.x; itp < n__; itp += gridDim.x * blockDim.x) {
        to_ptr__[itp] = from_ptr__[itp];
    }
}

template <typename T, typename F>
__global__ void copy_complex_kernel(const T* from_ptr__, F* to_ptr__, int n__)
{
    for (int itp = threadIdx.x + blockIdx.x * blockDim.x; itp < n__; itp += gridDim.x * blockDim.x) {
        T temp_from = from_ptr__[itp];
        F temp_to;
        temp_to.x = temp_from.x;
        temp_to.y = temp_from.y;
        to_ptr__[itp] = temp_to;
    }
}

extern "C" void copy_double_to_float_gpu(float* to_device_ptr__, const double* from_device_ptr__, int n__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(n__, grid_t.x));
    accLaunchKernel((copy_kernel<double, float>), dim3(grid_b), dim3(grid_t), 0, 0, from_device_ptr__, to_device_ptr__, n__);
}

extern "C" void copy_float_to_double_gpu(double* to_device_ptr__, const float* from_device_ptr__, int n__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(n__, grid_t.x));
    accLaunchKernel((copy_kernel<float, double>), dim3(grid_b), dim3(grid_t), 0, 0, from_device_ptr__, to_device_ptr__, n__);
}

extern "C" void copy_double_to_float_complex_gpu(acc_complex_float_t* to_device_ptr__,
                                                 const acc_complex_double_t* from_device_ptr__, int n__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(n__, grid_t.x));
    accLaunchKernel((copy_complex_kernel<acc_complex_double_t, acc_complex_float_t>), dim3(grid_b), dim3(grid_t), 0, 0,
                    from_device_ptr__, to_device_ptr__, n__);
}

extern "C" void copy_float_to_double_complex_gpu(acc_complex_double_t* to_device_ptr__,
                                                 const acc_complex_float_t* from_device_ptr__, int n__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(n__, grid_t.x));
    accLaunchKernel((copy_complex_kernel<acc_complex_float_t, acc_complex_double_t>), dim3(grid_b), dim3(grid_t), 0, 0,
                    from_device_ptr__, to_device_ptr__, n__);
}
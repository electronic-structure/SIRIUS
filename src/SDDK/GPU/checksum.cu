#include "cuda_common.h"
#include "cuda_interface.h"

__global__ void double_complex_checksum_gpu_kernel
(
    cuDoubleComplex const* ptr__,
    size_t size__,
    cuDoubleComplex *result__
)
{
    int N = num_blocks(size__, blockDim.x);

    extern __shared__ char sdata_ptr[];
    double* sdata_x = (double*)&sdata_ptr[0];
    double* sdata_y = (double*)&sdata_ptr[blockDim.x * sizeof(double)];

    sdata_x[threadIdx.x] = 0.0;
    sdata_y[threadIdx.x] = 0.0;

    for (int n = 0; n < N; n++) {
        int j = n * blockDim.x + threadIdx.x;
        if (j < size__) {
            sdata_x[threadIdx.x] += ptr__[j].x;
            sdata_y[threadIdx.x] += ptr__[j].y;
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

    *result__ = make_cuDoubleComplex(sdata_x[0], sdata_y[0]);
}

extern "C" void double_complex_checksum_gpu(cuDoubleComplex const* ptr__,
                                            size_t size__,
                                            cuDoubleComplex* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(1);

    cuDoubleComplex* res = (cuDoubleComplex*)cuda_malloc(sizeof(cuDoubleComplex));

    double_complex_checksum_gpu_kernel <<<grid_b, grid_t, 2 * grid_t.x * sizeof(double)>>>
    (
        ptr__,
        size__,
        res
    );

    cuda_copy_to_host(result__, res, sizeof(cuDoubleComplex));

    cuda_free(res);
}

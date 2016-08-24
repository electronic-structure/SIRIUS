#include "kernels_common.h"

extern "C" void* cuda_malloc(size_t size);
extern "C" void cuda_free(void* ptr);
extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             cuDoubleComplex* alpha, cuDoubleComplex const* a, int32_t lda, cuDoubleComplex const* b, 
                             int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id);

extern "C" void cublas_dgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             double* alpha, double const* a, int32_t lda, double const* b, 
                             int32_t ldb, double* beta, double* c, int32_t ldc, int stream_id);

__global__ void generate_phase_factors_conj_gpu_kernel
(
    int num_gvec_loc__, 
    int num_atoms__, 
    double const* atom_pos__, 
    int const* gvec__, 
    double* phase_factors__
)
{
    int ia = blockIdx.y;
    int igloc = blockIdx.x * blockDim.x + threadIdx.x;

    if (igloc < num_gvec_loc__)
    {
        int gvx = gvec__[array2D_offset(0, igloc, 3)];
        int gvy = gvec__[array2D_offset(1, igloc, 3)];
        int gvz = gvec__[array2D_offset(2, igloc, 3)];
    
        double ax = atom_pos__[array2D_offset(0, ia, 3)];
        double ay = atom_pos__[array2D_offset(1, ia, 3)];
        double az = atom_pos__[array2D_offset(2, ia, 3)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        phase_factors__[array2D_offset(ia, 2 * igloc,     num_atoms__)] = cos(p);
        phase_factors__[array2D_offset(ia, 2 * igloc + 1, num_atoms__)] = -sin(p);
    }
}

extern "C" void generate_dm_pw_gpu(int num_atoms__,
                                   int num_gvec_loc__,
                                   int nbf__,
                                   double const* atom_pos__,
                                   int const* gvec__,
                                   double* phase_factors__, 
                                   double const* dm__,
                                   double* dm_pw__,
                                   int stream_id__)
{
    CUDA_timer t("generate_dm_pw_gpu");

    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    dim3 grid_t(32);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    generate_phase_factors_conj_gpu_kernel<<<grid_b, grid_t, 0, stream>>>
    (
        num_gvec_loc__, 
        num_atoms__, 
        atom_pos__, 
        gvec__, 
        phase_factors__
    );
    
    double alpha = 1;
    double beta = 0;

    cublas_dgemm(0, 0, nbf__ * (nbf__ + 1) / 2, num_gvec_loc__ * 2, num_atoms__, &alpha, 
                 dm__, nbf__ * (nbf__ + 1) / 2, phase_factors__, num_atoms__, &beta,
                 dm_pw__, nbf__ * (nbf__ + 1) / 2, stream_id__);
}


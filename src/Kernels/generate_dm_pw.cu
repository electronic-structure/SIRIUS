#include "kernels_common.h"

extern "C" void* cuda_malloc(size_t size);
extern "C" void cuda_free(void* ptr);
extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             cuDoubleComplex* alpha, cuDoubleComplex const* a, int32_t lda, cuDoubleComplex const* b, 
                             int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id);

__global__ void generate_phase_factors_conj_gpu_kernel
(
    int num_gvec_loc__, 
    int num_atoms__, 
    double const* atom_pos__, 
    int const* gvec__, 
    cuDoubleComplex* phase_factors__
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

        double sinp = sin(p);
        double cosp = cos(p);

        phase_factors__[array2D_offset(ia, igloc, num_atoms__)] = make_cuDoubleComplex(cosp, -sinp);
    }
}

extern "C" void generate_dm_pw_gpu(int num_atoms__,
                                   int num_gvec_loc__,
                                   int nbf__,
                                   double const* atom_pos__,
                                   int const* gvec__,
                                   cuDoubleComplex const* dm__,
                                   cuDoubleComplex* dm_pw__)
{
    CUDA_timer t("generate_dm_pw_gpu");

    cuDoubleComplex* phase_factors;
    phase_factors = (cuDoubleComplex*)cuda_malloc(num_atoms__ * num_gvec_loc__ * sizeof (cuDoubleComplex));

    dim3 grid_t(32);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    generate_phase_factors_conj_gpu_kernel<<<grid_b, grid_t>>>
    (
        num_gvec_loc__, 
        num_atoms__, 
        atom_pos__, 
        gvec__, 
        phase_factors
    );
    
    cuDoubleComplex zone = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex zzero = make_cuDoubleComplex(0.0, 0.0);

    cublas_zgemm(0, 0, nbf__ * nbf__, num_gvec_loc__, num_atoms__, &zone, 
                 dm__, nbf__ * nbf__, phase_factors, num_atoms__, &zzero,
                 dm_pw__, nbf__ * nbf__, -1);

    cuda_free(phase_factors);
}


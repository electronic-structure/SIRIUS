#include "../SDDK/GPU/cuda_common.h"

__global__ void mul_veff_with_phase_factors_gpu_kernel(int num_gvec_loc__,
                                                       cuDoubleComplex const* veff__, 
                                                       int const* gvec__, 
                                                       int num_atoms__,
                                                       double const* atom_pos__, 
                                                       cuDoubleComplex* veff_a__)
{
    int ia = blockIdx.y;
    double ax = atom_pos__[array2D_offset(ia, 0, num_atoms__)];
    double ay = atom_pos__[array2D_offset(ia, 1, num_atoms__)];
    double az = atom_pos__[array2D_offset(ia, 2, num_atoms__)];

    int igloc = blockDim.x * blockIdx.x + threadIdx.x;
    if (igloc < num_gvec_loc__)
    {
        int gvx = gvec__[array2D_offset(igloc, 0, num_gvec_loc__)];
        int gvy = gvec__[array2D_offset(igloc, 1, num_gvec_loc__)];
        int gvz = gvec__[array2D_offset(igloc, 2, num_gvec_loc__)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        veff_a__[array2D_offset(igloc, ia, num_gvec_loc__)] = cuConj(cuCmul(veff__[igloc], make_cuDoubleComplex(cos(p), sin(p))));
    }
}
 
extern "C" void mul_veff_with_phase_factors_gpu(int num_atoms__,
                                                int num_gvec_loc__, 
                                                cuDoubleComplex const* veff__, 
                                                int const* gvec__, 
                                                double const* atom_pos__,
                                                double* veff_a__,
                                                int stream_id__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    mul_veff_with_phase_factors_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
    (
        num_gvec_loc__,
        veff__,
        gvec__,
        num_atoms__,
        atom_pos__,
        (cuDoubleComplex*)veff_a__
    );
}

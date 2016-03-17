#include "kernels_common.h"

__global__ void generate_phase_factors_gpu_kernel
(
    int num_gvec_loc, 
    double const* atom_pos, 
    int const* gvec, 
    cuDoubleComplex* phase_factors
)
{
    int ia = blockIdx.y;
    int igloc = blockIdx.x * blockDim.x + threadIdx.x;

    if (igloc < num_gvec_loc)
    {
        int gvx = gvec[array2D_offset(0, igloc, 3)];
        int gvy = gvec[array2D_offset(1, igloc, 3)];
        int gvz = gvec[array2D_offset(2, igloc, 3)];
    
        double ax = atom_pos[array2D_offset(0, ia, 3)];
        double ay = atom_pos[array2D_offset(1, ia, 3)];
        double az = atom_pos[array2D_offset(2, ia, 3)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        double sinp = sin(p);
        double cosp = cos(p);

        phase_factors[array2D_offset(igloc, ia, num_gvec_loc)] = make_cuDoubleComplex(cosp, sinp);
    }
}


extern "C" void generate_phase_factors_gpu(int num_gvec_loc__,
                                           int num_atoms__,
                                           int const* gvec__,
                                           double const* atom_pos__,
                                           cuDoubleComplex* phase_factors__)

{
    dim3 grid_t(32);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    generate_phase_factors_gpu_kernel<<<grid_b, grid_t>>>
    (
        num_gvec_loc__, 
        atom_pos__, 
        gvec__, 
        phase_factors__
    );
}

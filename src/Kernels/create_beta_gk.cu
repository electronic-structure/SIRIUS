#include "../SDDK/GPU/cuda_common.hpp"

__global__ void create_beta_gk_gpu_kernel
(
    int num_gkvec__, 
    int const* beta_desc__,
    cuDoubleComplex const* beta_gk_t, 
    double const* gkvec, 
    double const* atom_pos,
    cuDoubleComplex* beta_gk
)
{
    int ia = blockIdx.y;
    int igk = blockDim.x * blockIdx.x + threadIdx.x;

    int nbf              = beta_desc__[array2D_offset(0, ia, 4)];
    int offset_beta_gk   = beta_desc__[array2D_offset(1, ia, 4)];
    int offset_beta_gk_t = beta_desc__[array2D_offset(2, ia, 4)];

    if (igk < num_gkvec__)
    {
        double p = 0;
        for (int x = 0; x < 3; x++) p += atom_pos[array2D_offset(x, ia, 3)] * gkvec[array2D_offset(x, igk, 3)];
        p *= twopi;

        double sinp = sin(p);
        double cosp = cos(p);

        for (int xi = 0; xi < nbf; xi++)
        {
            beta_gk[array2D_offset(igk, offset_beta_gk + xi, num_gkvec__)] =
                cuCmul(beta_gk_t[array2D_offset(igk, offset_beta_gk_t + xi, num_gkvec__)],
                       make_cuDoubleComplex(cosp, -sinp));
        }
    }
}

extern "C" void create_beta_gk_gpu(int num_atoms,
                                   int num_gkvec,
                                   int const* beta_desc,
                                   cuDoubleComplex const* beta_gk_t,
                                   double const* gkvec,
                                   double const* atom_pos,
                                   cuDoubleComplex* beta_gk)
{
    CUDA_timer t("create_beta_gk_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), num_atoms);

    create_beta_gk_gpu_kernel <<<grid_b, grid_t>>> 
    (
        num_gkvec,
        beta_desc,
        beta_gk_t,
        gkvec,
        atom_pos,
        beta_gk
    );
}


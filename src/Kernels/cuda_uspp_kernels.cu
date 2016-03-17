#include "kernels_common.h"

extern cudaStream_t* streams;
extern "C" void* cuda_malloc(size_t size);
extern "C" void cuda_free(void* ptr);
extern "C" void cublas_zgemm(int transa, int transb, int32_t m, int32_t n, int32_t k, 
                             cuDoubleComplex* alpha, cuDoubleComplex* a, int32_t lda, cuDoubleComplex* b, 
                             int32_t ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int32_t ldc, int stream_id);

__global__ void compute_chebyshev_order1_gpu_kernel
(
    int num_gkvec__,
    double c__,
    double r__,
    cuDoubleComplex* phi0__,
    cuDoubleComplex* phi1__
)
{
    int igk = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;

    if (igk < num_gkvec__)
    {
        int i = array2D_offset(igk, j, num_gkvec__);
        // phi0 * c
        cuDoubleComplex z1 = cuCmul(phi0__[i], make_cuDoubleComplex(c__, 0));
        // phi1 - phi0 * c
        cuDoubleComplex z2 = cuCsub(phi1__[i], z1);
        // (phi1 - phi0 * c) / r
        phi1__[i] = cuCdiv(z2, make_cuDoubleComplex(r__, 0));
    }
}

__global__ void compute_chebyshev_orderk_gpu_kernel
(
    int num_gkvec__,
    double c__,
    double r__,
    cuDoubleComplex* phi0__,
    cuDoubleComplex* phi1__,
    cuDoubleComplex* phi2__
)
{
    int igk = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;

    if (igk < num_gkvec__)
    {
        int i = array2D_offset(igk, j, num_gkvec__);
        // phi1 * c
        cuDoubleComplex z1 = cuCmul(phi1__[i], make_cuDoubleComplex(c__, 0));
        // phi2 - phi1 * c
        cuDoubleComplex z2 = cuCsub(phi2__[i], z1);
        // (phi2 - phi1 * c) * 2 / r
        cuDoubleComplex z3 = cuCmul(z2, make_cuDoubleComplex(2.0 / r__, 0));
        // (phi2 - phi1 * c) * 2 / r - phi0
        phi2__[i] = cuCsub(z3, phi0__[i]);
    }
}

extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 cuDoubleComplex* phi0,
                                                 cuDoubleComplex* phi1,
                                                 cuDoubleComplex* phi2)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gkvec, grid_t.x), n);

    if (phi2 == NULL)
    {
        compute_chebyshev_order1_gpu_kernel <<<grid_b, grid_t>>>
        (
            num_gkvec,
            c,
            r,
            phi0,
            phi1
        );
    }
    else
    {
        compute_chebyshev_orderk_gpu_kernel <<<grid_b, grid_t>>>
        (
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
//==                                              cuDoubleComplex* beta_pw_type,
//==                                              cuDoubleComplex* phi,
//==                                              cuDoubleComplex* beta_phi)
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
//==     cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);
//== 
//==     for (int m = 0; m < N; m++)
//==     {
//==         __shared__ cuDoubleComplex beta_pw_tile[BLOCK_SIZE][BLOCK_SIZE];
//==         __shared__ cuDoubleComplex phi_tile[BLOCK_SIZE][BLOCK_SIZE];
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
//==             beta_pw_tile[threadIdx.x][threadIdx.y] = cuCmul(cuConj(beta_pw_type[array2D_offset(igk, offset_t, num_gkvec)]), 
//==                                                             make_cuDoubleComplex(cosp, sinp));
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
//==         for (int i = 0; i < bs; i++) val = cuCadd(val, cuCmul(beta_pw_tile[threadIdx.x][i], phi_tile[threadIdx.y][i]));
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
//==     generate_beta_phi_gpu_kernel<<<
//==         numBlocks, 
//==         threadsPerBlock>>>(num_gkvec, 
//==                            num_beta,
//==                            num_phi,
//==                            beta_t_idx, 
//==                            atom_pos,
//==                            gkvec, 
//==                            (cuDoubleComplex*)beta_pw_type,
//==                            (cuDoubleComplex*)phi,
//==                            (cuDoubleComplex*)beta_phi);
//== }




//__global__ void copy_beta_psi_gpu_kernel
//(
//    cuDoubleComplex const* beta_psi,
//    int beta_psi_ld, 
//    double const* wo,
//    cuDoubleComplex* beta_psi_wo,
//    int beta_psi_wo_ld
//)
//{
//    int xi = threadIdx.x;
//    int j = blockIdx.x;
//
//    beta_psi_wo[array2D_offset(xi, j, beta_psi_wo_ld)] = cuCmul(cuConj(beta_psi[array2D_offset(xi, j, beta_psi_ld)]),
//                                                                make_cuDoubleComplex(wo[j], 0.0));
//}

//extern "C" void copy_beta_psi_gpu(int nbf,
//                                  int nloc,
//                                  cuDoubleComplex const* beta_psi,
//                                  int beta_psi_ld,
//                                  double const* wo,
//                                  cuDoubleComplex* beta_psi_wo,
//                                  int beta_psi_wo_ld,
//                                  int stream_id)
//{
//    dim3 grid_t(nbf);
//    dim3 grid_b(nloc);
//    
//    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
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
    cuDoubleComplex const* f1,
    cuDoubleComplex const* f2,
    double* prod
)
{
    int N = num_blocks(num_gkvec_row, blockDim.x);

    extern __shared__ char sdata_ptr[];
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
                                          cuDoubleComplex const* f1,
                                          cuDoubleComplex const* f2,
                                          double* prod)
{
    dim3 grid_t(64);
    dim3 grid_b(n);

    compute_inner_product_gpu_kernel <<<grid_b, grid_t, grid_t.x * sizeof(double)>>>
    (
        num_gkvec_row,
        f1,
        f2,
        prod
    );
}



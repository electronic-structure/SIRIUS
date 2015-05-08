#include "cuda_interface.h"

__global__ void spline_inner_product_gpu_kernel_v3(int num_points__,
                                                   int const* idx_ri__,
                                                   double const* x__,
                                                   double const* dx__,
                                                   double const* f__,
                                                   double const* g__,
                                                   double* result__)
{
    int nb = num_blocks(num_points__, blockDim.x);
    int idx = blockIdx.x;
    int idx_f = array2D_offset(0, idx, 2);
    int idx_g = array2D_offset(1, idx, 2);

    extern __shared__ char sdata_ptr[];
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
                
            double a1a2 = a1 * a2;
            double d1d2 = d1 * d2;
                
            double k1 = d1 * b2 + c1 * c2 + b1 * d2;

            double k2 = d1 * a2 + c1 * b2 + b1 * c2 + a1 * d2;

            double k3 = c1 * a2 + b1 * b2 + a1 * c2;

            double k4 = d1 * c2 + c1 * d2;
            
            double k5 = b1 * a2 + a1 * b2;

            sdata[threadIdx.x] += dxi * ((a1a2 * xi * xi) + 
                                  dxi * ((xi * (2.0 * a1a2 + xi * k5)) / 2.0 +
                                  dxi * ((a1a2 + xi * (2.0 * k5 + k3 * xi)) / 3.0 + 
                                  dxi * ((k5 + xi * (2.0 * k3 + k2 * xi)) / 4.0 +
                                  dxi * ((k3 + xi * (2.0 * k2 + k1 * xi)) / 5.0 + 
                                  dxi * ((k2 + xi * (2.0 * k1 + k4 * xi)) / 6.0 + 
                                  dxi * ((k1 + xi * (2.0 * k4 + d1d2 * xi)) / 7.0 + 
                                  dxi * ((k4 + 2.0 * d1d2 * xi) / 8.0 + 
                                  dxi * d1d2 / 9.0)))))))); 
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    result__[idx] = sdata[0];
}

extern "C" double spline_inner_product_gpu_v2(int size__, double const* x__, double const* dx__, double const* f__, 
                                              double const* g__, double* d_buf__, double* h_buf__, int stream_id__)
{
    cudaStream_t stream = (stream_id__ == -1) ? NULL : streams[stream_id__];

    dim3 grid_t(256);
    dim3 grid_b(num_blocks(size__, grid_t.x));

    //double* d_result;
    //CALL_CUDA(cudaMalloc, (&d_result, grid_b.x * sizeof(double)));

    spline_inner_product_gpu_kernel_v2 <<<grid_b, grid_t, grid_t.x * sizeof(double), stream>>>
    (
        size__,
        x__,
        dx__,
        f__,
        g__,
        d_buf__
    );

    //double* h_result = (double*)malloc(grid_b.x * sizeof(double));
    CALL_CUDA(cudaMemcpyAsync, (h_buf__, d_buf__, grid_b.x * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CALL_CUDA(cudaStreamSynchronize, (stream));
    
    //cudaMemcpy(h_result, d_result, grid_b.x * sizeof(double), cudaMemcpyDeviceToHost);
    //CALL_CUDA(cudaFree, (d_result));

    double result = 0;
    for (int ib = 0; ib < grid_b.x; ib++) result += h_buf__[ib];
    //free(h_result);
    
    return result;
}



//==================================
// High-level functions and kernels
//==================================

template <typename T, typename U>
__device__ U spline_inner_product_gpu_function(int ld, int size, double* r_dr, T* s1_coefs, U* s2_coefs)
{
    int N = size / blockDim.x;
    if (size % blockDim.x != 0) N++;

    extern __shared__ char sdata_ptr[];
    U* sdata = (U*)&sdata_ptr[0];

    int a_offs = 0 * ld;
    int b_offs = 1 * ld;
    int c_offs = 2 * ld;
    int d_offs = 3 * ld;

    sdata[threadIdx.x] = 0;

    for (int n = 0; n < N; n++)
    {
        int i = n * blockDim.x + threadIdx.x;
        if (i < size - 1)
        {
            double x0 = r_dr[i];
            double dx = r_dr[ld + i];

            T a1 = s1_coefs[a_offs + i];
            T b1 = s1_coefs[b_offs + i];
            T c1 = s1_coefs[c_offs + i];
            T d1 = s1_coefs[d_offs + i];
            
            U a2 = s2_coefs[a_offs + i];
            U b2 = s2_coefs[b_offs + i];
            U c2 = s2_coefs[c_offs + i];
            U d2 = s2_coefs[d_offs + i];
                
            U a1a2 = a1 * a2;
            U d1d2 = d1 * d2;
                
            U k1 = d1 * b2 + c1 * c2 + b1 * d2;

            U k2 = d1 * a2 + c1 * b2 + b1 * c2 + a1 * d2;

            U k3 = c1 * a2 + b1 * b2 + a1 * c2;

            U k4 = d1 * c2 + c1 * d2;
            
            U k5 = b1 * a2 + a1 * b2;

            sdata[threadIdx.x] += dx * ((a1a2 * x0 * x0) + 
                                  dx * ((x0 * (2.0 * a1a2 + x0 * k5)) / 2.0 +
                                  dx * ((a1a2 + x0 * (2.0 * k5 + k3 * x0)) / 3.0 + 
                                  dx * ((k5 + x0 * (2.0 * k3 + k2 * x0)) / 4.0 +
                                  dx * ((k3 + x0 * (2.0 * k2 + k1 * x0)) / 5.0 + 
                                  dx * ((k2 + x0 * (2.0 * k1 + k4 * x0)) / 6.0 + 
                                  dx * ((k1 + x0 * (2.0 * k4 + d1d2 * x0)) / 7.0 + 
                                  dx * ((k4 + 2.0 * d1d2 * x0) / 8.0 + 
                                  dx * d1d2 / 9.0)))))))); 
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    //if (threadIdx.x == 0) for (int i = 1; i < blockDim.x; i++) sdata[0] += sdata[i];

    return sdata[0];
}

template <> __device__ 
cuDoubleComplex spline_inner_product_gpu_function<double, cuDoubleComplex>(int ld, int size, double* r_dr, 
                                                                           double* s1_coefs, 
                                                                           cuDoubleComplex* s2_coefs)
{
    int N = size / blockDim.x;
    if (size % blockDim.x != 0) N++;

    extern __shared__ char sdata_ptr[];
    cuDoubleComplex* sdata = (cuDoubleComplex*)&sdata_ptr[0];

    int a_offs = 0 * ld;
    int b_offs = 1 * ld;
    int c_offs = 2 * ld;
    int d_offs = 3 * ld;

    sdata[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);

    for (int n = 0; n < N; n++)
    {
        int i = n * blockDim.x + threadIdx.x;
        if (i < size - 1)
        {
            double x0 = r_dr[i];
            double dx = r_dr[ld + i];

            double a1 = s1_coefs[a_offs + i];
            double b1 = s1_coefs[b_offs + i];
            double c1 = s1_coefs[c_offs + i];
            double d1 = s1_coefs[d_offs + i];
            
            cuDoubleComplex a2 = s2_coefs[a_offs + i];
            cuDoubleComplex b2 = s2_coefs[b_offs + i];
            cuDoubleComplex c2 = s2_coefs[c_offs + i];
            cuDoubleComplex d2 = s2_coefs[d_offs + i];
                
            cuDoubleComplex a1a2 = make_cuDoubleComplex(a1 * a2.x, a1 * a2.y);
            cuDoubleComplex d1d2 = make_cuDoubleComplex(d1 * d2.x, d1 * d2.y);
                
            cuDoubleComplex k1 = make_cuDoubleComplex(d1 * b2.x + c1 * c2.x + b1 * d2.x, 
                                                      d1 * b2.y + c1 * c2.y + b1 * d2.y);

            cuDoubleComplex k2 = make_cuDoubleComplex(d1 * a2.x + c1 * b2.x + b1 * c2.x + a1 * d2.x, 
                                                      d1 * a2.y + c1 * b2.y + b1 * c2.y + a1 * d2.y);

            cuDoubleComplex k3 = make_cuDoubleComplex(c1 * a2.x + b1 * b2.x + a1 * c2.x, 
                                                      c1 * a2.y + b1 * b2.y + a1 * c2.y);

            cuDoubleComplex k4 = make_cuDoubleComplex(d1 * c2.x + c1 * d2.x, d1 * c2.y + c1 * d2.y);
            
            cuDoubleComplex k5 = make_cuDoubleComplex(b1 * a2.x + a1 * b2.x, b1 * a2.y + a1 * b2.y);

            cuDoubleComplex z = make_cuDoubleComplex(
                                  dx * ((a1a2.x * x0 * x0) + 
                                  dx * ((x0 * (2.0 * a1a2.x + x0 * k5.x)) / 2.0 +
                                  dx * ((a1a2.x + x0 * (2.0 * k5.x + k3.x * x0)) / 3.0 + 
                                  dx * ((k5.x + x0 * (2.0 * k3.x + k2.x * x0)) / 4.0 +
                                  dx * ((k3.x + x0 * (2.0 * k2.x + k1.x * x0)) / 5.0 + 
                                  dx * ((k2.x + x0 * (2.0 * k1.x + k4.x * x0)) / 6.0 + 
                                  dx * ((k1.x + x0 * (2.0 * k4.x + d1d2.x * x0)) / 7.0 + 
                                  dx * ((k4.x + 2.0 * d1d2.x * x0) / 8.0 + 
                                  dx * d1d2.x / 9.0)))))))),
                                  dx * ((a1a2.y * x0 * x0) + 
                                  dx * ((x0 * (2.0 * a1a2.y + x0 * k5.y)) / 2.0 +
                                  dx * ((a1a2.y + x0 * (2.0 * k5.y + k3.y * x0)) / 3.0 + 
                                  dx * ((k5.y + x0 * (2.0 * k3.y + k2.y * x0)) / 4.0 +
                                  dx * ((k3.y + x0 * (2.0 * k2.y + k1.y * x0)) / 5.0 + 
                                  dx * ((k2.y + x0 * (2.0 * k1.y + k4.y * x0)) / 6.0 + 
                                  dx * ((k1.y + x0 * (2.0 * k4.y + d1d2.y * x0)) / 7.0 + 
                                  dx * ((k4.y + 2.0 * d1d2.y * x0) / 8.0 + 
                                  dx * d1d2.y / 9.0)))))))));

            sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], z);
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) 
    {
        if (threadIdx.x % (2 * s) == 0) sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    
    //if (threadIdx.x == 0) for (int i = 1; i < blockDim.x; i++) sdata[0] = cuCadd(sdata[0], sdata[i]);

    return sdata[0];
}

template <typename T, typename U>
__global__ void spline_inner_product_gpu_kernel(int ld, int size, double* r_dr, T* s1_coefs, U* s2_coefs, U* result)
{
    result[0] = spline_inner_product_gpu_function(ld, size, r_dr, s1_coefs, s2_coefs);
}

template <typename T>
void spline_inner_product_gpu(int size, double* r_dr, T* s1_coefs, T* s2_coefs)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(1);

    T* d_result;
    cudaMalloc(&d_result, 1 * sizeof(T));
    spline_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>(size, size, r_dr, s1_coefs, s2_coefs, d_result);

    T* h_result = (T*)malloc(1 * sizeof(T));
    cudaMemcpy(h_result, d_result, 1 * sizeof(T), cudaMemcpyDeviceToHost);

    printf("GPU result : %18.12f \n", h_result[0]);

    cudaFree(d_result);
    free(h_result);
    
    //cuDoubleComplex* d_zresult;
    //cudaMalloc(&d_zresult, 1 * sizeof(cuDoubleComplex));
    //
    //cuDoubleComplex* zs2;
    //cudaMalloc(&zs2, size * 4 * sizeof(cuDoubleComplex));
    //
    //for (int i = 0; i < size * 4; i++) zs2[i] = make_cuDoubleComplex(s2_coefs[i], s2_coefs[i]);

    //spline_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>(size, size, r_dr, s1_coefs, zs2, d_zresult);

    //cuDoubleComplex* h_zresult = (cuDoubleComplex*)malloc(1 * sizeof(cuDoubleComplex));
    //cudaMemcpy(h_zresult, d_zresult, 1 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    //printf("GPU result : %18.12f %18.12f\n", h_zresult[0].x, h_zresult[0].y);

    //cudaFree(d_zresult);
    //free(h_zresult);
    //free(zs2);
}

template void spline_inner_product_gpu<double>(int size, double* r_dr, double* s1_coefs, double* s2_coefs);







// Input array dimensions:
//   sbessel_coefs(max_num_mt_points * 4, lmax_pw + 1, num_atom_types, num_gkvec_row);
//   lo_coefs(max_num_mt_points * 4, num_lo);
//   jlo(num_gkvec, num_lo);
__global__ void sbessel_lo_inner_product_gpu_kernel(int* kargs, int num_gkvec, int* l_by_ilo, int* iat_by_ilo, 
                                                    int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, 
                                                    double* lo_coefs, double* jlo)
{
    int num_atom_types = kargs[0];
    int max_nmtp = kargs[1];
    int lmax_pw = kargs[2];

    int igk = blockIdx.x;
    int ilo = blockIdx.y;

    int l = l_by_ilo[ilo];
    int iat = iat_by_ilo[ilo];
    int nmtp = nmtp_by_iat[iat];

    double* jl_ptr = &sbessel_coefs[array4D_offset(0, l, iat, igk, max_nmtp * 4, lmax_pw + 1, num_atom_types)];
    double* lo_ptr = &lo_coefs[array2D_offset(0, ilo, max_nmtp * 4)];
    double* r_dr_ptr = &r_dr[array2D_offset(0, iat, 2 * max_nmtp)];
    
    jlo[array2D_offset(igk, ilo, num_gkvec)] = 
        spline_inner_product_gpu_function(max_nmtp, nmtp, r_dr_ptr, jl_ptr, lo_ptr);
}


void sbessel_lo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int* l_by_ilo, int* iat_by_ilo, 
                                  int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* lo_coefs, double* jlo)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_gkvec, num_lo);

    sbessel_lo_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>
        (kargs, num_gkvec, l_by_ilo, iat_by_ilo, nmtp_by_iat, r_dr, sbessel_coefs, lo_coefs, jlo);
}

// Compute <jl|V|lo>
// Input array dimensions:
//   vlo(max_num_mt_points * 4, lmmax_pw, num_lo_col)
//   jvlo(lmmax_pw, num_gkvec, num_lo)
__global__ void sbessel_vlo_inner_product_gpu_kernel(int* kargs, int num_gkvec, int* l_by_lm, int* iat_by_ilo, 
                                                     int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, 
                                                     cuDoubleComplex* vlo_coefs, cuDoubleComplex* jvlo)
{
    int num_atom_types = kargs[0];
    int max_nmtp = kargs[1];
    int lmax_pw = kargs[2];
    int lmmax_pw = kargs[3];

    int igk = blockIdx.x;
    int ilo = blockIdx.y;
    int lm = blockIdx.z;

    int l = l_by_lm[lm];
    int iat = iat_by_ilo[ilo];
    int nmtp = nmtp_by_iat[iat];
    
    double* jl_ptr = &sbessel_coefs[array4D_offset(0, l, iat, igk, max_nmtp * 4, lmax_pw + 1, num_atom_types)];
    cuDoubleComplex* vlo_ptr = &vlo_coefs[array3D_offset(0, lm, ilo, 4 * max_nmtp, lmmax_pw)];
    double* r_dr_ptr = &r_dr[array2D_offset(0, iat, 2 * max_nmtp)];
    
    jvlo[array3D_offset(lm, igk, ilo, lmmax_pw, num_gkvec)] = 
        spline_inner_product_gpu_function(max_nmtp, nmtp, r_dr_ptr, jl_ptr, vlo_ptr);
}

// Compute <jl|V|lo>
void sbessel_vlo_inner_product_gpu(int* kargs, int num_gkvec, int num_lo, int lmmax_pw, int* l_by_lm, int* iat_by_ilo, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, void* vlo_coefs, void* jvlo)
{
    dim3 threadsPerBlock(64);
    dim3 numBlocks(num_gkvec, num_lo, lmmax_pw);

    sbessel_vlo_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16>>>
        (kargs, num_gkvec, l_by_lm, iat_by_ilo, nmtp_by_iat, r_dr, sbessel_coefs, (cuDoubleComplex*)vlo_coefs, 
         (cuDoubleComplex*)jvlo);
}

__global__ void sbessel_vlm_inner_product_gpu_kernel(int* kargs, int* iat_by_ia, int* l_by_lm, int* nmtp_by_iat,
                                                     double* r_dr, double* sbessel_coefs, double* vlm_coefs, 
                                                     double* jvlm)
{
    int max_nmtp = kargs[1];
    int lmax_pot = kargs[2];
    int lmmax_pot = kargs[3];
    
    int lm = blockIdx.x;
    int ia = blockIdx.y;

    int iat = iat_by_ia[ia];
    int nmtp = nmtp_by_iat[ia];
    int l = l_by_lm[lm];

    double* jl_ptr = &sbessel_coefs[array3D_offset(0, l, iat, max_nmtp * 4, lmax_pot + 1)];
    double* vlm_ptr = &vlm_coefs[array3D_offset(0, lm, ia, max_nmtp * 4, lmmax_pot)];
    double* r_dr_ptr = &r_dr[array2D_offset(0, iat, 2 * max_nmtp)];

    jvlm[array2D_offset(lm, ia, lmmax_pot)] = 
        spline_inner_product_gpu_function(max_nmtp, nmtp, r_dr_ptr, jl_ptr, vlm_ptr);
}


void sbessel_vlm_inner_product_gpu(int* kargs, int lmmax_pot, int num_atoms, int* iat_by_ia, int* l_by_lm, 
                                   int* nmtp_by_iat, double* r_dr, double* sbessel_coefs, double* vlm_coefs, 
                                   double* jvlm, int stream_id)
{
    cudaStream_t stream = (stream_id == -1) ? NULL : streams[stream_id];
    dim3 threadsPerBlock(64);
    dim3 numBlocks(lmmax_pot, num_atoms);
    
    sbessel_vlm_inner_product_gpu_kernel<<<numBlocks, threadsPerBlock, 64 * 16, stream>>>
        (kargs, iat_by_ia, l_by_lm, nmtp_by_iat, r_dr, sbessel_coefs, vlm_coefs, jvlm);
}


//__global__ void add_band_density_gpu_kernel(int nmtp, int lmmax_rho, int max_nmtp, int max_num_gaunt, int* gaunt12_size, 
//                                            int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, 
//                                            cuDoubleComplex* gaunt12_cg, cuDoubleComplex* fylm, double weight, 
//                                            int ia, double* dens)
//{
//    int ir = blockDim.x * blockIdx.x + threadIdx.x;
//    int lm = blockIdx.y;
//
//    int offs3 = array3D_offset(ir, lm, ia, max_nmtp, lmmax_rho);
//
//    if (ir < nmtp)
//    {
//        for (int k = 0; k < gaunt12_size[lm]; k++)
//        {
//            int offs = array2D_offset(k, lm, max_num_gaunt);
//            int lm1 = gaunt12_lm1_by_lm3[offs];
//            int lm2 = gaunt12_lm2_by_lm3[offs];
//            cuDoubleComplex cg = gaunt12_cg[offs];
//            
//            int offs1 = array2D_offset(ir, lm1, max_nmtp);
//            int offs2 = array2D_offset(ir, lm2, max_nmtp);
//
//            cuDoubleComplex z = cuCmul(cuConj(fylm[offs1]), fylm[offs2]);
//
//            dens[offs3] += weight * cuCreal(cuCmul(z, cg));
//        }
//    }
//}

__global__ void add_band_density_gpu_kernel(int lmmax_rho, int lmmax_wf, int max_nmtp, int* ia_by_ialoc, 
                                            int* iat_by_ia, int* nmtp_by_iat, int max_num_gaunt, 
                                            int* gaunt12_size, int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, 
                                            cuDoubleComplex* gaunt12_cg, cuDoubleComplex* fylm, double weight, 
                                            double* dens)
{
    int lm = blockIdx.x;
    int ialoc = blockIdx.y;
    int ia = ia_by_ialoc[ialoc];
    int iat = iat_by_ia[ia];
    int nmtp = nmtp_by_iat[iat];

    int offs3 = array3D_offset(0, lm, ialoc, max_nmtp, lmmax_rho);

    int N = nmtp / blockDim.x;
    if (nmtp % blockDim.x != 0) N++;

    for (int k = 0; k < gaunt12_size[lm]; k++)
    {
        int offs = array2D_offset(k, lm, max_num_gaunt);

        int lm1 = gaunt12_lm1_by_lm3[offs];
        int lm2 = gaunt12_lm2_by_lm3[offs];
        cuDoubleComplex cg = gaunt12_cg[offs];
        
        int offs1 = array3D_offset(0, lm1, ia, max_nmtp, lmmax_wf);
        int offs2 = array3D_offset(0, lm2, ia, max_nmtp, lmmax_wf);
        
        for (int n = 0; n < N; n++)
        {
            int ir = n * blockDim.x + threadIdx.x;
            if (ir < nmtp)
            {
                cuDoubleComplex z = cuCmul(cuConj(fylm[offs1 + ir]), fylm[offs2 + ir]);

                dens[offs3 + ir] += weight * cuCreal(cuCmul(z, cg));
            }
        }
    }
}

void add_band_density_gpu(int lmmax_rho, int lmmax_wf, int max_nmtp, int num_atoms_loc, int* ia_by_ialoc, 
                          int* iat_by_ia, int* nmtp_by_iat, int max_num_gaunt, int* gaunt12_size, 
                          int* gaunt12_lm1_by_lm3, int* gaunt12_lm2_by_lm3, void* gaunt12_cg, void* fylm, 
                          double weight, double* dens)
{
    dim3 threadsPerBlock(128);
    dim3 numBlocks(lmmax_rho, num_atoms_loc);
    add_band_density_gpu_kernel<<<numBlocks, threadsPerBlock>>>
        (lmmax_rho, lmmax_wf, max_nmtp, ia_by_ialoc, iat_by_ia, nmtp_by_iat, max_num_gaunt, gaunt12_size, 
         gaunt12_lm1_by_lm3, gaunt12_lm2_by_lm3, (cuDoubleComplex*)gaunt12_cg, (cuDoubleComplex*)fylm, weight, dens);
}
    


__global__ void scale_matrix_columns_gpu_kernel
(
    int nrow,
    cuDoubleComplex* mtrx,
    double* a
)
{
    int icol = blockIdx.y;
    int irow = blockIdx.x * blockDim.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] =
            cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(a[icol], 0));
    }
}

// scale each column of the matrix by a column-dependent constant
extern "C" void scale_matrix_columns_gpu(int nrow,
                                        int ncol,
                                        cuDoubleComplex* mtrx,
                                        double* a)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    scale_matrix_columns_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow,
        mtrx,
        a
    );
}

__global__ void scale_matrix_rows_gpu_kernel
(
    int nrow,
    cuDoubleComplex* mtrx,
    double* v
)
{
    int icol = blockIdx.y;
    int irow = blockDim.x * blockIdx.x + threadIdx.x;
    if (irow < nrow) 
    {
        mtrx[array2D_offset(irow, icol, nrow)] = 
            cuCmul(mtrx[array2D_offset(irow, icol, nrow)], make_cuDoubleComplex(v[irow], 0));
    }
}

// scale each row of the matrix by a row-dependent constant
extern "C" void scale_matrix_rows_gpu(int nrow,
                                      int ncol,
                                      cuDoubleComplex* mtrx,
                                      double* v)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(nrow, grid_t.x), ncol);

    scale_matrix_rows_gpu_kernel <<<grid_b, grid_t>>>
    (
        nrow,
        mtrx,
        v
    );
}

//== __global__ void update_it_density_matrix_0_gpu_kernel(int fft_size, 
//==                                                       int nfft_max, 
//==                                                       cuDoubleComplex* psi_it, 
//==                                                       double* wt,
//==                                                       double* it_density_matrix)
//== {
//==     int ir = blockIdx.x * blockDim.x + threadIdx.x;
//==     for (int i = 0; i < nfft_max; i++)
//==     {
//==         if (ir < fft_size)
//==         {
//==             cuDoubleComplex z = psi_it[array3D_offset(ir, i, 0, fft_size, nfft_max)];
//==             it_density_matrix[array2D_offset(ir, 0, fft_size)] += (z.x * z.x + z.y * z.y) * wt[i];
//==         }
//==     }
//== }

__global__ void update_it_density_matrix_1_gpu_kernel(int fft_size,
                                                      int nfft_max,
                                                      int ispn,
                                                      cuDoubleComplex const* psi_it,
                                                      double const* wt,
                                                      double* it_density_matrix)
{
    int ir = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < nfft_max; i++)
    {
        if (ir < fft_size)
        {
            cuDoubleComplex z = psi_it[array3D_offset(ir, i, ispn, fft_size, nfft_max)];
            it_density_matrix[array2D_offset(ir, ispn, fft_size)] += (z.x * z.x + z.y * z.y) * wt[i];
        }
    }
}


//extern "C" void update_it_density_matrix_gpu(int fft_size, 
//                                             int nfft_max, 
//                                             int num_spins, 
//                                             int num_mag_dims, 
//                                             cuDoubleComplex* psi_it, 
//                                             double* wt, 
//                                             double* it_density_matrix)
//{
//    CUDA_timer t("update_it_density_matrix_gpu");
//
//    dim3 grid_t(64);
//    dim3 grid_b(num_blocks(fft_size, grid_t.x));
//
//    switch (num_mag_dims)
//    {
//        //== case 3:
//        //== {
//        //==     for (int ir = 0; ir < fft_->size(); ir++)
//        //==     {
//        //==         double_complex z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
//        //==         it_density_matrix(ir, 2) += 2.0 * real(z);
//        //==         it_density_matrix(ir, 3) -= 2.0 * imag(z);
//        //==     }
//        //== }
//        case 1:
//        {
//            update_it_density_matrix_1_gpu_kernel <<<grid_b, grid_t>>>
//            (
//                fft_size,
//                nfft_max,
//                psi_it,
//                wt,
//                it_density_matrix
//            );
//        }
//        case 0:
//        {
//            update_it_density_matrix_0_gpu_kernel <<<grid_b, grid_t>>>
//            (
//                fft_size,
//                nfft_max,
//                psi_it,
//                wt,
//                it_density_matrix
//            );
//        }
//    }
//}

extern "C" void update_it_density_matrix_1_gpu(int fft_size, 
                                               int ispin,
                                               cuDoubleComplex const* psi_it, 
                                               double const* wt, 
                                               double* it_density_matrix)
{
    CUDA_timer t("update_it_density_matrix_gpu");

    dim3 grid_t(64);
    dim3 grid_b(num_blocks(fft_size, grid_t.x));

    update_it_density_matrix_1_gpu_kernel <<<grid_b, grid_t>>>
    (
        fft_size,
        1,
        ispin,
        psi_it,
        wt,
        it_density_matrix
    );

//==     switch (num_mag_dims)
//==     {
//==         //== case 3:
//==         //== {
//==         //==     for (int ir = 0; ir < fft_->size(); ir++)
//==         //==     {
//==         //==         double_complex z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
//==         //==         it_density_matrix(ir, 2) += 2.0 * real(z);
//==         //==         it_density_matrix(ir, 3) -= 2.0 * imag(z);
//==         //==     }
//==         //== }
//==         case 1:
//==         {
//==             update_it_density_matrix_1_gpu_kernel <<<grid_b, grid_t>>>
//==             (
//==                 fft_size,
//==                 nfft_max,
//==                 psi_it,
//==                 wt,
//==                 it_density_matrix
//==             );
//==         }
//==         case 0:
//==         {
//==             update_it_density_matrix_0_gpu_kernel <<<grid_b, grid_t>>>
//==             (
//==                 fft_size,
//==                 nfft_max,
//==                 psi_it,
//==                 wt,
//==                 it_density_matrix
//==             );
//==         }
//==     }
}

inline __device__ uint32_t random(size_t seed)
{
    uint32_t h = 5381;

    return (h << (seed % 15)) + h;
}

__global__ void randomize_on_gpu_kernel
(
    double* ptr__,
    size_t size__
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size__) ptr__[i] = double(random(i)) / (1 << 31);
}

extern "C" void randomize_on_gpu(double* ptr, size_t size)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(size, grid_t.x));

    randomize_on_gpu_kernel <<<grid_b, grid_t>>>
    (
        ptr,
        size
    );
}



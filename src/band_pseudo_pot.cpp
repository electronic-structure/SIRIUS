#include <thread>
#include <mutex>
#include "band.h"
#include "debug.hpp"

namespace sirius {

#ifdef __SCALAPACK
/** \param [in] phi Input wave-functions [storage: CPU || GPU].
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU].
 */
void Band::add_non_local_contribution_parallel(K_point* kp__,
                                               int N__,
                                               int n__,
                                               dmatrix<double_complex>& phi__,
                                               dmatrix<double_complex>& op_phi__, 
                                               matrix<double_complex>& beta_gk__,
                                               mdarray<int, 1> const& packed_mtrx_offset__,
                                               mdarray<double_complex, 1>& op_mtrx_packed__,
                                               double_complex alpha)
{
    PROFILE();

    Timer t("sirius::Band::add_non_local_contribution_parallel");

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    /* allocate space for <beta|phi> array */
    int nbf_max = unit_cell_.max_mt_basis_size() * unit_cell_.beta_chunk(0).num_atoms_;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);

    /* result of atom-block-diagonal operator multiplied by <beta|phi> */
    matrix<double_complex> tmp(nbf_max, nloc);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi_tmp.allocate_on_device();
        tmp.allocate_on_device();
    }
    #endif

    for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++)
    {
        /* number of beta-projectors in the current chunk */
        int nbeta =  unit_cell_.beta_chunk(ib).num_beta_;
        int natoms = unit_cell_.beta_chunk(ib).num_atoms_;

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_phi;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), nbeta, nloc);
                break;
            }
            case GPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), beta_phi_tmp.at<GPU>(), nbeta, nloc);
                break;
            }
        }

        //Timer t1("sirius::Band::add_non_local_contribution_parallel|beta_phi", kp__->comm_row());
        kp__->generate_beta_gk(natoms, unit_cell_.beta_chunk(ib).atom_pos_, unit_cell_.beta_chunk(ib).desc_, beta_gk__);
        kp__->generate_beta_phi(nbeta, phi__.panel(), nloc, (int)s0.local_size(), beta_gk__, beta_phi);
        //double tval = t1.stop();

        //if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        //{
        //    printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/rank\n",
        //           nbeta, nloc, kp__->num_gkvec(),
        //           tval, 8e-9 * nbeta * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        //}

        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk__, op_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, op_phi__.panel(), nloc, (int)s0.local_size(),
                                         alpha, tmp);
    }
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

void Band::set_fv_h_o_fast_parallel(int N__,
                                    int n__,
                                    K_point* kp__,
                                    dmatrix<double_complex>& phi_slab__,
                                    dmatrix<double_complex>& hphi_slab__,
                                    dmatrix<double_complex>& ophi_slab__,
                                    dmatrix<double_complex>& h__,
                                    dmatrix<double_complex>& o__,
                                    dmatrix<double_complex>& h_old__,
                                    dmatrix<double_complex>& o_old__,
                                    mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::set_fv_h_o_fast_parallel", kp__->comm());

    splindex<block_cyclic> s0_col(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_col(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s0_row(N__,       kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_row(N__ + n__, kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < (int)s0_col.local_size(); i++)
    {
        memcpy(&h__(0, i), &h_old__(0, i), s0_row.local_size() * sizeof(double_complex));
        memcpy(&o__(0, i), &o_old__(0, i), s0_row.local_size() * sizeof(double_complex));
    }

    matrix<double_complex> tmp;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            tmp = matrix<double_complex>(kappa__.at<CPU>(), N__ + n__, n__);
            break;
        }
        case GPU:
        {
            #ifdef __GPU
            tmp = matrix<double_complex>(kappa__.at<CPU>(), kappa__.at<GPU>(), N__ + n__, n__);
            #endif
            break;
        }
    }
    
    int col_offs = (int)s0_col.local_size();
    Timer t2("sirius::Band::set_fv_h_o_fast_parallel|zgemm_eff", kp__->comm());
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<CPU>(), phi_slab__.ld(), 
                              hphi_slab__.at<CPU>(0, N__), hphi_slab__.ld(), tmp.at<CPU>(), tmp.ld());
            break;
        }
        case GPU:
        {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<GPU>(), phi_slab__.ld(),
                              hphi_slab__.at<GPU>(0, N__), hphi_slab__.ld(), tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            #endif
            break;
        }
    }
    kp__->comm().allreduce(tmp.at<CPU>(), (int)tmp.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)(s1_col.local_size() - col_offs); i++)
    {
        for (int j = 0; j < (int)s1_row.local_size(); j++)
        {
            h__(j, col_offs + i) = tmp(s1_row[j], s1_col[col_offs + i] - N__);
        }
    }

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<CPU>(), phi_slab__.ld(), 
                              ophi_slab__.at<CPU>(0, N__), ophi_slab__.ld(), tmp.at<CPU>(), tmp.ld());
            break;
        }
        case GPU:
        {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<GPU>(), phi_slab__.ld(),
                              ophi_slab__.at<GPU>(0, N__), ophi_slab__.ld(), tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            #endif
            break;
        }
    }
    kp__->comm().allreduce(tmp.at<CPU>(), (int)tmp.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)(s1_col.local_size() - col_offs); i++)
    {
        for (int j = 0; j < (int)s1_row.local_size(); j++)
        {
            o__(j, col_offs + i) = tmp(s1_row[j], s1_col[col_offs + i] - N__);
        }
    }
    double tval = t2.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm with M, N, K: %6i %6i %6i for H and O: %12.4f sec, %12.4f GFlops/rank\n",
               N__ + n__, n__, kp__->num_gkvec(), tval,
               2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    /* restore the bottom block of the matrix */
    if (N__ != 0)
    {
        linalg<CPU>::tranc(n__, N__, h__, 0, N__, h__, N__, 0);
        linalg<CPU>::tranc(n__, N__, o__, 0, N__, o__, N__, 0);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }
}

void Band::residuals_fast_parallel(int N__,
                                   int num_bands__,
                                   K_point* kp__,
                                   std::vector<double>& eval__,
                                   matrix<double_complex>& evec__,
                                   dmatrix<double_complex>& hphi__,
                                   dmatrix<double_complex>& ophi__,
                                   dmatrix<double_complex>& hpsi__,
                                   dmatrix<double_complex>& opsi__,
                                   dmatrix<double_complex>& res__,
                                   std::vector<double>& h_diag__,
                                   std::vector<double>& o_diag__,
                                   std::vector<double>& res_norm__,
                                   mdarray<double_complex, 1>& kappa__)
{
    PROFILE();

    Timer t("sirius::Band::residuals_fast_parallel", kp__->comm());

    int num_gkvec_loc = kp__->num_gkvec_loc();
    
    Timer t2("sirius::Band::residuals_fast_parallel|zgemm");
    if (parameters_.processing_unit() == CPU)
    {
        /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, hphi__.panel(), evec__, hpsi__.panel());
        /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, ophi__.panel(), evec__, opsi__.panel());
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, hphi__.at<GPU>(), hphi__.ld(),
                          evec__.at<GPU>(), evec__.ld(), hpsi__.at<GPU>(), hpsi__.ld());

        cublas_get_matrix(num_gkvec_loc, num_bands__, sizeof(double_complex), hpsi__.at<GPU>(), hpsi__.ld(),
                          hpsi__.at<CPU>(), hpsi__.ld());

        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, ophi__.at<GPU>(), ophi__.ld(),
                          evec__.at<GPU>(), evec__.ld(), opsi__.at<GPU>(), opsi__.ld());

        cublas_get_matrix(num_gkvec_loc, num_bands__, sizeof(double_complex), opsi__.at<GPU>(), opsi__.ld(),
                          opsi__.at<CPU>(), opsi__.ld());
        #endif
    }
    double tval = t2.stop();


    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm with M, N, K: %6i %6i %6i for hpsi and opsi: %12.4f sec, %12.4f GFlops/rank\n",
               kp__->num_gkvec(), num_bands__, N__, tval,
               2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }
    
    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double norm2 = 0;
        for (int igk = 0; igk < num_gkvec_loc; igk++) 
        {
            res__(igk, i) = hpsi__(igk, i) - eval__[i] * opsi__(igk, i);
            norm2 += real(conj(res__(igk, i)) * res__(igk, i));
        }
        res_norm__[i] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        for (int igk = 0; igk < num_gkvec_loc; igk++)
        {
            double p = h_diag__[igk] - eval__[i] * o_diag__[igk];

            p *= 2; // QE formula is in Ry; here we convert to Ha
            p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
            res__(igk, i) /= p;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double d = 0;
        for (int igk = 0; igk < num_gkvec_loc; igk++) 
            d += real(conj(res__(igk, i)) * res__(igk, i));
        norm2[i] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double d = 1.0 / std::sqrt(norm2[i]);
        for (int igk = 0; igk < num_gkvec_loc; igk++) res__(igk, i) *= d;
    }
}
#endif // __SCALAPACK

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_serial(K_point* kp__, 
                          std::vector<double> const& effective_potential__, 
                          std::vector<double> const& pw_ekin__, 
                          int N__,
                          int n__,
                          matrix<double_complex>& phi__,
                          matrix<double_complex>& hphi__,
                          mdarray<double_complex, 1>& kappa__,
                          mdarray<int, 1>& packed_mtrx_offset__,
                          mdarray<double_complex, 1>& d_mtrx_packed__)
{
    Timer t("sirius::Band::apply_h_serial");

    matrix<double_complex> phi, hphi;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(kp__->num_gkvec() * unit_cell_.mt_basis_size());
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* apply local part of Hamiltonian */
    apply_h_local_serial(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        /* copy hphi do device */
        hphi.copy_to_device();
    }
    #endif

    add_non_local_contribution_serial(kp__, N__, n__, phi__, hphi__, kappa__, packed_mtrx_offset__, d_mtrx_packed__, complex_one);
}

/** \param [in] phi Input wave-functions [storage: CPU || GPU].
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU].
 */
void Band::add_non_local_contribution_serial(K_point* kp__,
                                             int N__,
                                             int n__,
                                             matrix<double_complex>& phi__,
                                             matrix<double_complex>& op_phi__, 
                                             mdarray<double_complex, 1>& kappa__,
                                             mdarray<int, 1> const& packed_mtrx_offset__,
                                             mdarray<double_complex, 1>& op_mtrx_packed__,
                                             double_complex alpha)
{
    PROFILE();

    Timer t("sirius::Band::add_non_local_contribution_serial");

    matrix<double_complex> phi, op_phi, beta_gk;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__),    phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), op_phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(kp__->num_gkvec() * unit_cell_.mt_basis_size());
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        beta_gk = matrix<double_complex>(nullptr, kappa__.at<GPU>(), kp__->num_gkvec(), unit_cell_.mt_basis_size());
    }
    
    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(unit_cell_.mt_lo_basis_size(), n__);

    /* operator multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(unit_cell_.mt_lo_basis_size(), n__);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        kp__->generate_beta_phi(unit_cell_.mt_lo_basis_size(), phi, n__, 0, kp__->beta_gk(), beta_phi);

        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_lo_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         kp__->beta_gk(), op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         op_phi, n__, 0, alpha, work);
    }
    else
    {
        #ifdef __GPU
        kp__->generate_beta_gk(unit_cell_.num_atoms(), unit_cell_.beta_chunk(0).atom_pos_, unit_cell_.beta_chunk(0).desc_, beta_gk);
        phi.copy_to_device();
        kp__->generate_beta_phi(unit_cell_.mt_basis_size(), phi, n__, 0, beta_gk, beta_phi);
        
        op_phi.copy_to_device();
        kp__->add_non_local_contribution(unit_cell_.num_atoms(), unit_cell_.mt_basis_size(), unit_cell_.beta_chunk(0).desc_,
                                         beta_gk, op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         op_phi, n__, 0, alpha, work);
        op_phi.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};


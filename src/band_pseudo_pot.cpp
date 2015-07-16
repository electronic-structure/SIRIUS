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
    log_function_enter(__func__);
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

    log_function_exit(__func__);
}

void Band::add_non_local_contribution_parallel(K_point* kp__,
                                               dmatrix<double_complex>& phi__,
                                               dmatrix<double_complex>& op_phi__,
                                               dmatrix<double_complex>& op__,
                                               double_complex alpha)
{
    //auto uc = parameters_.unit_cell();
    //int num_bands = parameters_.num_fv_states();

    STOP();

    ///* <\beta_{\xi}^{\alpha}|\phi_j> */
    //dmatrix<double_complex> beta_phi(unit_cell_.mt_basis_size(), num_bands, kp__->blacs_grid());
    ///* compute <beta|phi> */
    //linalg<CPU>::gemm(2, 0, unit_cell_.mt_basis_size(), num_bands, kp__->num_gkvec(), complex_one, 
    //                  kp__->beta_gk_panel(), phi__, complex_zero, beta_phi);

    //dmatrix<double_complex> tmp(unit_cell_.mt_basis_size(), num_bands, kp__->blacs_grid());
    //linalg<CPU>::gemm(0, 0, unit_cell_.mt_basis_size(), num_bands, unit_cell_.mt_basis_size(), complex_one,
    //                  op__, beta_phi, complex_zero, tmp);

    //linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, unit_cell_.mt_basis_size(), alpha,
    //                  kp__->beta_gk_panel(), tmp, complex_one, op_phi__);
}


/** \param [in] phi Input wave-function [storage: CPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU].
 */
void Band::apply_h_local_parallel(K_point* kp__,
                                  std::vector<double> const& effective_potential__,
                                  std::vector<double> const& pw_ekin__,
                                  int N__,
                                  int n__,
                                  dmatrix<double_complex>& phi__,
                                  dmatrix<double_complex>& hphi__)
{
    STOP();

//    log_function_enter(__func__);
//    Timer t("sirius::Band::apply_h_local_parallel", kp__->comm_row());
//
//    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
//    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
//
//    splindex<block> sub_spl_n(s1.local_size() - s0.local_size(), kp__->num_ranks_row(), kp__->rank_row());
//    
//    int nphi = (int)sub_spl_n.local_size();
//
//    memcpy(&hphi__(0, s0.local_size()), &phi__(0, s0.local_size()), 
//           kp__->num_gkvec_row() * (s1.local_size() - s0.local_size()) * sizeof(double_complex));
//    
//    hphi__.gather(n__, N__);
//    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, nphi, hphi__.slice(), hphi__.slice());
//    hphi__.scatter(n__, N__);
//
//    log_function_exit(__func__);
}

/** \param [in] phi Input wave-function [storage: CPU && GPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU || GPU] 
 */
void Band::apply_h_parallel(K_point* kp__,
                            std::vector<double> const& effective_potential__,
                            std::vector<double> const& pw_ekin__,
                            int N__,
                            int n__,
                            dmatrix<double_complex>& phi__,
                            dmatrix<double_complex>& hphi__,
                            matrix<double_complex>& beta_gk__,
                            mdarray<int, 1> const& packed_mtrx_offset__,
                            mdarray<double_complex, 1>& d_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::apply_h_parallel", kp__->comm_row());

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        hphi__.copy_cols_to_device(N__, N__ + n__);
        #endif
    }

    add_non_local_contribution_parallel(kp__, N__, n__, phi__, hphi__, beta_gk__, packed_mtrx_offset__,
                                        d_mtrx_packed__, double_complex(1, 0));
    log_function_exit(__func__);
}

/** \param [in] phi Input wave-function [storage: CPU && GPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU || GPU].
 */
void Band::apply_h_o_parallel(K_point* kp__,
                              std::vector<double> const& effective_potential__,
                              std::vector<double> const& pw_ekin__,
                              int N__,
                              int n__,
                              dmatrix<double_complex>& phi__,
                              dmatrix<double_complex>& hphi__,
                              dmatrix<double_complex>& ophi__,
                              matrix<double_complex>& beta_gk__,
                              mdarray<int, 1>& packed_mtrx_offset__,
                              mdarray<double_complex, 1>& d_mtrx_packed__,
                              mdarray<double_complex, 1>& q_mtrx_packed__)
{
    LOG_FUNC_BEGIN();
    Timer t("sirius::Band::apply_h_o_parallel", kp__->comm_row());

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    if (parameters_.processing_unit() == CPU)
    {
        /* set intial ophi */
        memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        /* copy hphi do device */
        cuda_copy_to_device(hphi__.at<GPU>(0, s0.local_size()), hphi__.at<CPU>(0, s0.local_size()), 
                            kp__->num_gkvec_row() * nloc * sizeof(double_complex));

        /* set intial ophi */
        cuda_copy_device_to_device(ophi__.at<GPU>(0, s0.local_size()), phi__.at<GPU>(0, s0.local_size()), 
                                   kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }
    #endif

    /* allocate space for <beta|phi> array */
    int nbmax = 0;
    for (int ib = 0; ib < unit_cell_.num_beta_chunks(); ib++) nbmax = std::max(nbmax, unit_cell_.beta_chunk(ib).num_beta_);
    mdarray<double_complex, 1> beta_phi_tmp(nbmax * nloc);

    /* work space (result of Q or D multiplied by <beta|phi>) */
    matrix<double_complex> work(nbmax, nloc);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi_tmp.allocate_on_device();
        work.allocate_on_device();
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

        kp__->generate_beta_gk(natoms, unit_cell_.beta_chunk(ib).atom_pos_, unit_cell_.beta_chunk(ib).desc_, beta_gk__);
        
        kp__->generate_beta_phi(nbeta, phi__.panel(), nloc, (int)s0.local_size(), beta_gk__, beta_phi);

        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk__, d_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, hphi__.panel(), nloc, (int)s0.local_size(),
                                         complex_one, work);
        
        kp__->add_non_local_contribution(natoms, nbeta, unit_cell_.beta_chunk(ib).desc_, beta_gk__, q_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, ophi__.panel(), nloc, (int)s0.local_size(),
                                         complex_one, work);
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
    LOG_FUNC_END();
}

void Band::set_fv_h_o_parallel_simple(int N__,
                                      int n__,
                                      K_point* kp__,
                                      std::vector<double> const& veff_it_coarse__,
                                      std::vector<double> const& pw_ekin__,
                                      dmatrix<double_complex>& phi__,
                                      dmatrix<double_complex>& hphi__,
                                      dmatrix<double_complex>& ophi__,
                                      dmatrix<double_complex>& h__,
                                      dmatrix<double_complex>& o__,
                                      dmatrix<double_complex>& h_old__,
                                      dmatrix<double_complex>& o_old__,
                                      mdarray<double_complex, 2>& kappa__,
                                      mdarray<int, 1>& packed_mtrx_offset__,
                                      mdarray<double_complex, 1>& d_mtrx_packed__,
                                      mdarray<double_complex, 1>& q_mtrx_packed__)
{
    Timer t("sirius::Band::set_fv_h_o_parallel_simple", kp__->comm());

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

    /* apply Hamiltonian and overlap operators to the new basis functions */
    apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__, kappa__, 
                       packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__);
    
    Timer t2("sirius::Band::set_fv_h_o_parallel_simple|zgemm", kp__->comm());
    /* <{phi,res}|H|res> */
    linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, hphi__, 0, N__, complex_zero, h__, 0, N__);
    /* <{phi,res}|O|res> */
    linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, ophi__, 0, N__, complex_zero, o__, 0, N__);
    double tval = t2.stop();

    if (verbosity_level >= 10 && kp__->comm().rank() == 0)
    {
        printf("pzgemm #4&5 with M, N, K: %6i %6i %6i, offset in B&C: %6i, %12.4f sec, %12.4f GFlops/rank\n",
               N__ + n__, n__, kp__->num_gkvec(), N__,
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
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
    LOG_FUNC_BEGIN();

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

    LOG_FUNC_END();
}

void Band::set_fv_h_o_parallel(int N__,
                               int n__,
                               K_point* kp__,
                               std::vector<double>& veff_it_coarse__,
                               std::vector<double>& pw_ekin__,
                               dmatrix<double_complex>& phi__,
                               dmatrix<double_complex>& hphi__,
                               dmatrix<double_complex>& ophi__,
                               dmatrix<double_complex>& h__,
                               dmatrix<double_complex>& o__,
                               dmatrix<double_complex>& h_old__,
                               dmatrix<double_complex>& o_old__,
                               mdarray<double_complex, 2>& kappa__,
                               mdarray<int, 1>& packed_mtrx_offset__,
                               mdarray<double_complex, 1>& d_mtrx_packed__,
                               mdarray<double_complex, 1>& q_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::set_fv_h_o_parallel", kp__->comm());
    
    bool with_overlap = (parameters_.esm_type() == ultrasoft_pseudopotential);

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

    /* apply Hamiltonian and overlap operators to the new basis functions */
    if (with_overlap)
    {
        apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__,
                           kappa__, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__);
    }
    else
    {
        apply_h_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, kappa__,
                         packed_mtrx_offset__, d_mtrx_packed__);
    }
    
    #if defined(__GPU) && !defined(__GPU_DIRECT)
    if (parameters_.processing_unit() == GPU)
    {
        size_t panel_size = kp__->num_gkvec_row() * (s1_col.local_size() - s0_col.local_size()) * sizeof(double_complex);
        cuda_copy_to_host(hphi__.at<CPU>(0, s0_col.local_size()), hphi__.at<GPU>(0, s0_col.local_size()), panel_size);
        if (with_overlap) cuda_copy_to_host(ophi__.at<CPU>(0, s0_col.local_size()), ophi__.at<GPU>(0, s0_col.local_size()), panel_size);
    }
    #endif

    int max_num_phi_new = 0;
    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        max_num_phi_new = std::max(max_num_phi_new, (int)(s1_col.local_size(icol) - s0_col.local_size(icol)));
    
    int num_phi = (int)s1_col.local_size();

    assert(max_num_phi_new * 4 <= (int)kappa__.size(1));

    mdarray<double_complex, 3> hphi_tmp, ophi_tmp;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            hphi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0, 0),
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            ophi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0,  2 * max_num_phi_new),
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            break;
        }
        case GPU:
        {
            hphi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0, 0),
                                                  kappa__.at<GPU>(0, 0), 
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            ophi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0,  2 * max_num_phi_new), 
                                                  kappa__.at<GPU>(0,  2 * max_num_phi_new), 
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            break;
        }
    }
    
    mdarray<double_complex, 3> h_tmp(num_phi, max_num_phi_new, 2);
    mdarray<double_complex, 3> o_tmp(num_phi, max_num_phi_new, 2);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        h_tmp.allocate_on_device();
        o_tmp.allocate_on_device();
    }
    #endif

    std::array<std::atomic_bool, 2> lock_hphi;
    std::array<std::atomic_bool, 2> lock_ophi;
    std::array<std::atomic_bool, 2> lock_h;
    std::array<std::atomic_bool, 2> lock_o;
    for (int i = 0; i < 2; i++)
    {
        lock_hphi[i].store(false);
        lock_ophi[i].store(false);
        lock_h[i].store(false);
        lock_o[i].store(false);
    }
   
    Timer t1("sirius::Band::set_fv_h_o_parallel|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    //== struct col_to_row_scatter
    //== {
    //==     /* index (in h and o) of each vector for each row rank */
    //==     std::vector< std::vector<int> > ivec_row;

    //==     /* index in (|hphi> or |ophi>) of each vector for each row rank */
    //==     std::vector< std::vector<int> > idx_col;

    //==     /* offsets in the send buffer */
    //==     std::vector<int> offset;
    //== };

    //== std::vector<col_to_row_scatter> scatter_desc(kp__->num_ranks_col());


    //== for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    //== {
    //==     scatter_desc[icol].ivec_row = std::vector< std::vector<int> >(kp__->num_ranks_row());
    //==     scatter_desc[icol].idx_col = std::vector< std::vector<int> >(kp__->num_ranks_row());
    //==     
    //==     /* local number of new basis functions */
    //==     int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

    //==     for (int j = 0; j < nloc; j++)
    //==     {
    //==         int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
    //==         auto p = s1_row.location(idx_hphi_glob);

    //==         scatter_desc[icol].ivec_row[p.second].push_back((int)p.first);
    //==         scatter_desc[icol].idx_col[p.second].push_back((int)(s0_col.local_size(icol) + j));
    //==     }
    //==     scatter_desc[icol].offset = std::vector<int>(kp__->num_ranks_row());
    //==     int i = 0;
    //==     for (int j = 0; j < kp__->num_ranks_row(); j++)
    //==     {
    //==          scatter_desc[icol].offset[j] = i;
    //==          i += (int)scatter_desc[icol].idx_col.size();
    //==     }
    //== }

    
    /* auxiliary function to broadcast vectors (|hphi> or |ophi>) from a column icol
     *
     *    column ranks 
     *    0   1   2   3  ...
     *  +---+---+---+---+---
     *  |   | <-+-*-+-> |
     *  +---+---+---+---+---
     *  |   | <-+-*-+-> |
     *  +---+---+---+---+---
     *  |   | <-+-*-+-> |
     *  .................
     */
    auto bcast_column = [kp__, &s0_col, &s1_col, pu]
                        (int icol,
                         dmatrix<double_complex>& mtrx,
                         mdarray<double_complex, 3>& mtrx_tmp,
                         std::array<std::atomic_bool, 2>& lock_tmp) -> void
    {
        Timer t("sirius::Band::set_fv_h_o_parallel|bcast_column");

        #ifdef __GPU
        #ifdef __GPU_DIRECT
        bool gpu_direct = true;
        #else
        bool gpu_direct = false;
        #endif
        #endif

        /* wait for unlocking of this buffer */
        while (lock_tmp[icol % 2].load());
        
        /* number of vectors to broadcast */
        int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

        /* return if there is nothing to do */
        if (!nloc) return;
        
        /* total size of the panel to broadcast */
        size_t panel_size = kp__->num_gkvec_row() * nloc * sizeof(double_complex);
        
        if (pu == CPU)
        {
            if (kp__->rank_col() == icol)
            {
                /* this rank copies content of the vectors into temporary buffer */
                memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(0, s0_col.local_size(icol)), panel_size);
            }
            /* buffer is broadcasted between columns */
            kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
        }
        if (pu == GPU)
        {
            #ifdef __GPU
            if (gpu_direct)
            {
                if (kp__->rank_col() == icol)
                    cuda_copy_device_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx.at<GPU>(0, s0_col.local_size(icol)), panel_size);
                kp__->comm_col().bcast(mtrx_tmp.at<GPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
            } 
            else
            {
                if (kp__->rank_col() == icol)
                    memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(0, s0_col.local_size(icol)), panel_size);
                kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
                cuda_copy_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx_tmp.at<CPU>(0, 0, icol % 2), panel_size);
            }
            #else
            TERMINATE_NO_GPU
            #endif
        }

        /* lock temporary buffer; it can't be used for the next broadcast until <phi|tmp> is computed */
        lock_tmp[icol % 2].store(true);
    };

    /* broadcast |hphi> from the first column */
    bcast_column(0, hphi__, hphi_tmp, lock_hphi);

    /* same for |ophi> or |phi> */
    if (with_overlap)
    {
        bcast_column(0, ophi__, ophi_tmp, lock_ophi);
    }
    else
    {
        bcast_column(0, phi__, ophi_tmp, lock_ophi);
    }

    /* get maximum number of threads */
    int nthread = omp_get_max_threads();

    /* one thread will be doing communication, others will do local zgemm */
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    /* create communication thread */
    std::thread comm_thread([kp__, &s0_col, &s1_col, &s0_row, &s1_row, &lock_hphi, &lock_ophi, &lock_h, &lock_o, 
                             &hphi__, &ophi__, &hphi_tmp, &ophi_tmp, &h_tmp, &o_tmp, &h__, &o__, bcast_column, 
                             with_overlap, &phi__]()
    {
        int num_phi = (int)s1_col.local_size();
    
        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            /* local number of new basis functions */
            int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));
            
            Timer t1("sirius::Band::set_fv_h_o_parallel|comm_thread|1");
            /* broadcast next column */
            if (icol + 1 < kp__->num_ranks_col())
            {
                bcast_column(icol + 1, hphi__, hphi_tmp, lock_hphi);
                
                if (with_overlap)
                {
                    bcast_column(icol + 1, ophi__, ophi_tmp, lock_ophi);
                }
                else
                {
                    bcast_column(icol + 1, phi__, ophi_tmp, lock_ophi);
                }
            }
            t1.stop();
    
            Timer t2("sirius::Band::set_fv_h_o_parallel|comm_thread|2");
            if (nloc > 0)
            {
                /* wait for locking of h-buffer which happens after a local zgemm;
                 * when this is done the reduction between rows can be performed 
                 */
                while (!lock_h[icol % 2].load());
                kp__->comm_row().allreduce(&h_tmp(0, 0, icol % 2), num_phi * nloc);
                
                /* cycle through the local fraction of new basis functions */
                for (int j = 0; j < nloc; j++)
                {
                    /* compute global index of hphi by local index and column rank */
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    /* and now compute row rank and local index */
                    auto p = s1_row.location(idx_hphi_glob);
                    /* check if this rank stores <hphi_tmp_j|phi_i> */ 
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            h__(p.first, i) = conj(h_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* when the reduction is done and the result is saved in h__, h-buffer can be unlocked */
                lock_h[icol % 2].store(false);
                
                /* the same with o-buffer */
                while (!lock_o[icol % 2].load());
                kp__->comm_row().allreduce(&o_tmp(0, 0, icol % 2), num_phi * nloc);
    
                for (int j = 0; j < nloc; j++)
                {
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    auto p = s1_row.location(idx_hphi_glob);
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            o__(p.first, i) = conj(o_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* remove lock from o buffer */
                lock_o[icol % 2].store(false);
            }
            t2.stop();
        }
    });

    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    {
        int n = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

        if (n > 0)
        {
            /* wait for broadcast of this column */
            while (!lock_hphi[icol % 2].load());
            /* wait for unlock of h buffer */
            while (lock_h[icol % 2].load());

            Timer t2("sirius::Band::set_fv_h_o_parallel|zgemm_loc");
            if (pu == GPU)
            {
                #ifdef __GPU
                linalg<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<GPU>(), phi__.ld(),
                                  hphi_tmp.at<GPU>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<GPU>(0, 0, icol % 2), h_tmp.ld());
                cuda_copy_to_host(h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.at<GPU>(0, 0, icol % 2), num_phi * n * sizeof(double_complex));
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == CPU)
            {
                /* compute <phi|hphi_tmp> */
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<CPU>(), phi__.ld(),
                                  hphi_tmp.at<CPU>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.ld());
            }
            lock_h[icol % 2].store(true);
            lock_hphi[icol % 2].store(false);
        }
            
        if (n > 0)
        {
            while (!lock_ophi[icol % 2].load());
            while (lock_o[icol % 2].load());

            Timer t2("sirius::Band::set_fv_h_o_parallel|zgemm_loc");
            if (pu == GPU)
            {
                #ifdef __GPU
                linalg<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<GPU>(), phi__.ld(),
                                  ophi_tmp.at<GPU>(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.at<GPU>(0, 0, icol % 2), o_tmp.ld());
                cuda_copy_to_host(o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.at<GPU>(0, 0, icol % 2), num_phi * n * sizeof(double_complex));
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == CPU)
            {
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<CPU>(), phi__.ld(),
                                  ophi_tmp.at<CPU>(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.ld());
            }
            lock_o[icol % 2].store(true);
            lock_ophi[icol % 2].store(false);
        }
    }
    comm_thread.join();
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #4&5 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/rank\n",
               N__ + n__, n__, kp__->num_gkvec(),
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }

    /* restore right block of the matrix */
    if (N__ != 0)
    {
        Timer t1("sirius::Band::set_fv_h_o_parallel|transpose");
        linalg<CPU>::tranc(N__, n__, h__, N__, 0, h__, 0, N__);
        linalg<CPU>::tranc(N__, n__, o__, N__, 0, o__, 0, N__);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }
    
    log_function_exit(__func__);
}

void Band::precondition_and_normalize_residuals_parallel(int num_bands__,
                                                         K_point* kp__,
                                                         std::vector<double>& eval__,
                                                         dmatrix<double_complex>& hpsi__,
                                                         dmatrix<double_complex>& opsi__,
                                                         dmatrix<double_complex>& res__,
                                                         std::vector<double>& h_diag__,
                                                         std::vector<double>& o_diag__,
                                                         std::vector<double>& res_norm__)

{
    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(),
                                             parameters_.cyclic_block_size());

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
        double norm2 = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
        {
            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        }
        res_norm__[ires] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    
    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
    
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            double p = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];

            p *= 2; // QE formula is in Ry; here we convert to Ha
            p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
            res__(igk_row, i) /= p;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
        double d = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        norm2[ires] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
        double d = 1.0 / std::sqrt(norm2[ires]);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    }
}

void Band::residuals_parallel_simple(int N__,
                                     int num_bands__,
                                     K_point* kp__,
                                     std::vector<double>& eval__,
                                     dmatrix<double_complex>& evec__,
                                     dmatrix<double_complex>& hphi__,
                                     dmatrix<double_complex>& ophi__,
                                     dmatrix<double_complex>& hpsi__,
                                     dmatrix<double_complex>& opsi__,
                                     dmatrix<double_complex>& res__,
                                     std::vector<double>& h_diag__,
                                     std::vector<double>& o_diag__,
                                     std::vector<double>& res_norm__)
{
    Timer t("sirius::Band::residuals_parallel_simple");
    
    Timer t2("sirius::Band::residuals_parallel_simple|zgemm");
    /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, hphi__, evec__, complex_zero, hpsi__);
    /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, ophi__, evec__, complex_zero, opsi__);
    double tval = t2.stop();

    if (verbosity_level >= 10 && kp__->comm().rank() == 0)
    {
        printf("pzgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/rank\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }
    
    precondition_and_normalize_residuals_parallel(num_bands__, kp__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm__);
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
    LOG_FUNC_BEGIN();

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

    LOG_FUNC_END();
}

void Band::residuals_parallel(int N__,
                              int num_bands__,
                              K_point* kp__,
                              std::vector<double>& eval__,
                              dmatrix<double_complex>& evec__,
                              dmatrix<double_complex>& hphi__,
                              dmatrix<double_complex>& ophi__,
                              dmatrix<double_complex>& hpsi__,
                              dmatrix<double_complex>& opsi__,
                              dmatrix<double_complex>& res__,
                              std::vector<double>& h_diag__,
                              std::vector<double>& o_diag__,
                              std::vector<double>& res_norm__,
                              mdarray<double_complex, 2>& kappa__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::residuals_parallel", kp__->comm());

    Timer t1("sirius::Band::residuals_parallel|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(),
                                             parameters_.cyclic_block_size());
    splindex<block_cyclic> spl_num_bands_row(num_bands__, kp__->num_ranks_row(), kp__->rank_row(),
                                             parameters_.cyclic_block_size());
    
    /* transpose matrix of eigen-vectors;
     * row index of evec_t runs over bands, column index runs over basis functions 
     */ 
    dmatrix<double_complex> evec_t(num_bands__, N__, kp__->blacs_grid(), parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
    linalg<CPU>::tranu(num_bands__, N__, evec__, 0, 0, evec_t, 0, 0);
    
    /* local number of basis function |phi> */
    int num_phi_loc = evec_t.num_cols_local();

    mdarray<double_complex, 3> evec_tmp(num_phi_loc, spl_num_bands_col.local_size(0), 2);
    #ifdef __GPU
    if (pu == GPU) evec_tmp.allocate_on_device();
    #endif
    
    std::array<std::atomic_bool, 2> lock_evec_tmp;
    std::atomic_bool lock_hpsi_tmp;
    std::atomic_bool lock_opsi_tmp;
    for (int i = 0; i < 2; i++) lock_evec_tmp[i].store(false);
    lock_hpsi_tmp.store(false);
    lock_opsi_tmp.store(false);

    /* maximum local number of bands */
    int num_bnd_max = (int)spl_num_bands_col.local_size(0);

    matrix<double_complex> hpsi_tmp, opsi_tmp;

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            hpsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, 0),           kp__->num_gkvec_row(), num_bnd_max);
            opsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, num_bnd_max), kp__->num_gkvec_row(), num_bnd_max);
            break;
        }
        case GPU:
        {
            hpsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, 0),           kappa__.at<GPU>(0, 0), 
                                              kp__->num_gkvec_row(), num_bnd_max);
            opsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, num_bnd_max), kappa__.at<GPU>(0, num_bnd_max),
                                              kp__->num_gkvec_row(), num_bnd_max);
            break;
        }
    }
    
    /* get eigen-vectors for a specific column */
    auto get_evec = [kp__, &spl_num_bands_col, &spl_num_bands_row, &evec_t, &evec_tmp, num_phi_loc, pu]
                    (int icol) -> void 
    {
        /* number of bands for this column */
        int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
        
        /* zero temporary buffer */
        memset(&evec_tmp(0, 0, icol % 2), 0, num_phi_loc * num_bands_of_col * sizeof(double_complex));

        /* loop over local fraction of bands */
        for (int i = 0; i < num_bands_of_col; i++)
        {
            /* global index of band */
            int iglob = (int)spl_num_bands_col.global_index(i, icol);

            /* location of the global band index in the row of evec_t */
            auto p = spl_num_bands_row.location(iglob); 
            
            /* pick elements of evec_t from row ranks */
            if (p.second == kp__->rank_row())
            {
                for (int j = 0; j < num_phi_loc; j++) evec_tmp(j, i, icol % 2) = evec_t(p.first, j);
            }
        }

        /* reduce evec_tmp; now it contains fraction of the expansion coefficients of bands for the given column
         * over the basis function local to this column 
         */
        kp__->comm_row().allreduce(&evec_tmp(0, 0, icol % 2), num_phi_loc * num_bands_of_col);
        #ifdef __GPU
        if (pu == GPU)
        {
            /* send evec to gpu */
            cuda_copy_to_device(evec_tmp.at<GPU>(0, 0, icol % 2), evec_tmp.at<CPU>(0, 0, icol % 2), 
                                num_phi_loc * num_bands_of_col * sizeof(double_complex));
        }
        #endif
    };

    /* get evec for first column */
    get_evec(0);
    lock_evec_tmp[0].store(true);
    
    /* communication thread */
    std::thread comm_thread([kp__, &lock_evec_tmp, &lock_hpsi_tmp, &hpsi_tmp, &hpsi__, 
                             &lock_opsi_tmp, &opsi_tmp, &opsi__, &spl_num_bands_col, get_evec, pu]()
    {
        #ifdef __GPU
        #ifdef __GPU_DIRECT
        // gpu-direct is not working at the moment
        bool gpu_direct = false;
        #else
        bool gpu_direct = false;
        #endif
        #endif

        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
            if (icol + 1 < kp__->num_ranks_col())
            {
                while (lock_evec_tmp[(icol + 1) % 2].load());
                get_evec(icol + 1);
                lock_evec_tmp[(icol + 1) % 2].store(true);
            }
            
            while (!lock_hpsi_tmp.load());
            switch (pu)
            {
                case CPU:
                {
                    kp__->comm_col().reduce(hpsi_tmp.at<CPU>(), hpsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case GPU:
                {
                    #ifdef __GPU
                    if (gpu_direct)
                    {
                        kp__->comm_col().reduce(hpsi_tmp.at<GPU>(), hpsi__.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    } 
                    else
                    {
                        cuda_copy_to_host(hpsi_tmp.at<CPU>(), hpsi_tmp.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                        kp__->comm_col().reduce(hpsi_tmp.at<CPU>(), hpsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                        if (icol == kp__->rank_col())
                            cuda_copy_to_device(hpsi__.at<GPU>(), hpsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
                    break;
                }
            }
            lock_hpsi_tmp.store(false);

            while (!lock_opsi_tmp.load());
            switch (pu)
            {
                case CPU:
                {
                    kp__->comm_col().reduce(opsi_tmp.at<CPU>(), opsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case GPU:
                {
                    #ifdef __GPU
                    if (gpu_direct)
                    {
                        kp__->comm_col().reduce(opsi_tmp.at<GPU>(), opsi__.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    }
                    else
                    {
                        cuda_copy_to_host(opsi_tmp.at<CPU>(), opsi_tmp.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                        kp__->comm_col().reduce(opsi_tmp.at<CPU>(), opsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                        if (icol == kp__->rank_col())
                            cuda_copy_to_device(opsi__.at<GPU>(), opsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
                    break;
                }
            }
            lock_opsi_tmp.store(false);
        }
    });

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    for (int rank_col = 0; rank_col < kp__->num_ranks_col(); rank_col++)
    {
        int num_bands_of_rank = (int)spl_num_bands_col.local_size(rank_col);
        
        while (!lock_evec_tmp[rank_col % 2].load());
        
        while (lock_hpsi_tmp.load());
        switch (pu)
        {
            case CPU:
            {
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  hphi__.at<CPU>(), hphi__.ld(), evec_tmp.at<CPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  hpsi_tmp.at<CPU>(), hpsi_tmp.ld());
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  hphi__.at<GPU>(), hphi__.ld(), evec_tmp.at<GPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  hpsi_tmp.at<GPU>(), hpsi_tmp.ld());
                cuda_device_synchronize();
                break;
                #endif
            }
        }
        lock_hpsi_tmp.store(true);
       
        while (lock_opsi_tmp.load());
        switch (pu)
        {
            case CPU:
            {
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  ophi__.at<CPU>(), ophi__.ld(), evec_tmp.at<CPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  opsi_tmp.at<CPU>(), opsi_tmp.ld());
                break;
            }
            case GPU:
            {
                #ifdef __GPU
                linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  ophi__.at<GPU>(), ophi__.ld(), evec_tmp.at<GPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  opsi_tmp.at<GPU>(), opsi_tmp.ld());
                cuda_device_synchronize();
                break;
                #endif
            }
        }
        lock_opsi_tmp.store(true);

        lock_evec_tmp[rank_col % 2].store(false);
    }
    comm_thread.join();
    
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/rank\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    precondition_and_normalize_residuals_parallel(num_bands__, kp__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm__);

    //memset(&res_norm__[0], 0, num_bands__ * sizeof(double));

    //if (pu == CPU)
    //{
    //    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //        double norm2 = 0;
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
    //        {
    //            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
    //            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
    //        }
    //        res_norm__[ires] = norm2;
    //    }
    //    kp__->comm().allreduce(res_norm__);
    //    
    //    /* compute norm */
    //    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    //    
    //    /* apply preconditioner */
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //    
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    //        {
    //            double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
    //            if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
    //            res__(igk_row, i) /= z;
    //        }
    //    }
    //    
    //    std::vector<double> norm2(num_bands__, 0);
    //    /* normalize new basis functions */
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //        double d = 0;
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
    //            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
    //        norm2[ires] = d;
    //    }
    //    kp__->comm().allreduce(norm2);
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //        double d = 1.0 / std::sqrt(norm2[ires]);
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    //    }
    //}

    //if (pu == GPU)
    //{
    //    #ifdef __GPU
    //    mdarray<double, 1> res_norm_gpu(&res_norm__[0], num_bands__);
    //    res_norm_gpu.allocate_on_device();
    //    res_norm_gpu.zero_on_device();

    //    mdarray<double, 1> eval_gpu(&eval__[0], num_bands__);
    //    eval_gpu.allocate_on_device();
    //    eval_gpu.copy_to_device();

    //    mdarray<int, 1> res_idx_gpu(res__.num_cols_local());
    //    for (int i = 0; i < res__.num_cols_local(); i++) res_idx_gpu(i) = res__.icol(i);
    //    res_idx_gpu.allocate_on_device();
    //    res_idx_gpu.copy_to_device();

    //    compute_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
    //                          hpsi__.at<GPU>(), opsi__.at<GPU>(), res__.at<GPU>(), res_norm_gpu.at<GPU>());
    //    res_norm_gpu.copy_to_host();

    //    kp__->comm().allreduce(res_norm__);
    //    
    //    /* compute norm */
    //    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

    //    mdarray<double, 1> hdiag_gpu(&h_diag__[0], kp__->num_gkvec_row());
    //    hdiag_gpu.allocate_on_device();
    //    hdiag_gpu.copy_to_device();

    //    mdarray<double, 1> odiag_gpu(&o_diag__[0], kp__->num_gkvec_row());
    //    odiag_gpu.allocate_on_device();
    //    odiag_gpu.copy_to_device();

    //    mdarray<double, 1> norm2(num_bands__);
    //    norm2.allocate_on_device();
    //    norm2.zero_on_device();

    //    apply_preconditioner_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
    //                             hdiag_gpu.at<GPU>(), odiag_gpu.at<GPU>(), res__.at<GPU>(), norm2.at<GPU>());
    //    // TODO: test gpudirect here
    //    norm2.copy_to_host();
    //    kp__->comm().allreduce(norm2.at<CPU>(), num_bands__);
    //    norm2.copy_to_device();

    //    normalize_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(),
    //                            norm2.at<GPU>(), res__.at<GPU>());
    //    #else
    //    TERMINATE_NO_GPU
    //    #endif
    //}

    log_function_exit(__func__);
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
    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);

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
    log_function_enter(__func__);
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

    log_function_exit(__func__);
}

};


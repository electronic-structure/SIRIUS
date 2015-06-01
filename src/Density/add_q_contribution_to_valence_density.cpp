#include "density.h"

namespace sirius {

void Density::add_q_contribution_to_valence_density(K_set& ks)
{
    Timer t("sirius::Density::add_q_contribution_to_valence_density", ctx_.comm());

    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
     * because density matrix is symmetric.
     */
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    /* complex density matrix */
    mdarray<double_complex, 4> pp_complex_density_matrix(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(),
                                                         num_zdmat, unit_cell_.num_atoms());
    pp_complex_density_matrix.zero();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = (int)ks.spl_num_kpoints(ikloc);
        auto occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);

        add_k_point_contribution<CPU, ultrasoft_pseudopotential>(ks[ik], occupied_bands, pp_complex_density_matrix);
    }
    ctx_.comm().allreduce(pp_complex_density_matrix.at<CPU>(), (int)pp_complex_density_matrix.size());

    auto rl = ctx_.reciprocal_lattice();

    std::vector<double_complex> f_pw(rl->num_gvec(), complex_zero);

    /* split local fraction of G-vectors between threads */
    splindex<block> spl_ngv_loc(rl->spl_num_gvec().local_size(), Platform::max_num_threads(), 0);

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type->mt_basis_size();

        mdarray<double_complex, 2> d_mtrx_packed(atom_type->num_atoms(), nbf * nbf);
        for (int i = 0; i < atom_type->num_atoms(); i++)
        {
            int ia = atom_type->atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    d_mtrx_packed(i, xi2 * nbf + xi1) = pp_complex_density_matrix(xi2, xi1, 0, ia);
                }
            }
        }
        #pragma omp parallel
        {
            mdarray<double_complex, 2> phase_factors(spl_ngv_loc.local_size(), atom_type->num_atoms());
            
            mdarray<double_complex, 2> d_mtrx_pw(spl_ngv_loc.local_size(), nbf * nbf);
    
            int thread_id = Platform::thread_id();

            for (int i = 0; i < atom_type->num_atoms(); i++)
            {
                int ia = atom_type->atom_id(i);

                for (int igloc_t = 0; igloc_t < (int)spl_ngv_loc.local_size(thread_id); igloc_t++)
                {
                    int igloc = (int)spl_ngv_loc.global_index(igloc_t, thread_id);
                    phase_factors(igloc_t, i) = conj(rl->gvec_phase_factor<local>(igloc, ia));
                }
            }

            linalg<CPU>::gemm(0, 0, (int)spl_ngv_loc.local_size(thread_id), nbf * nbf, atom_type->num_atoms(),
                              &phase_factors(0, 0), phase_factors.ld(), &d_mtrx_packed(0, 0), d_mtrx_packed.ld(), 
                              &d_mtrx_pw(0, 0), d_mtrx_pw.ld());

            /* remember that d_mtrx_pw is not a Hermitian matrix in xi1,xi2 indices */
            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                int idx12 = xi2 * (xi2 + 1) / 2;

                /* add diagonal term */
                for (int igloc_t = 0; igloc_t < (int)spl_ngv_loc.local_size(thread_id); igloc_t++)
                {
                    int igloc = (int)spl_ngv_loc.global_index(igloc_t, thread_id);
                    /* D_{xi2,xi2} * Q(G)_{xi2, xi2} */
                    f_pw[rl->spl_num_gvec(igloc)] += d_mtrx_pw(igloc_t, xi2 * nbf + xi2) * 
                                                     atom_type->uspp().q_pw(igloc, idx12 + xi2);

                }
                /* add non-diagonal terms */
                for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
                {
                    for (int igloc_t = 0; igloc_t < (int)spl_ngv_loc.local_size(thread_id); igloc_t++)
                    {
                        int igloc = (int)spl_ngv_loc.global_index(igloc_t, thread_id);
                        
                        /* D_{xi2,xi1} * Q(G)_{xi1, xi2} */
                        f_pw[rl->spl_num_gvec(igloc)] += d_mtrx_pw(igloc_t, xi2 * nbf + xi1) * atom_type->uspp().q_pw(igloc, idx12);

                        /* D_{xi1,xi2} * Q(G)_{xix, xi1}^{+} */
                        f_pw[rl->spl_num_gvec(igloc)] += d_mtrx_pw(igloc_t, xi1 * nbf + xi2) * conj(atom_type->uspp().q_pw(igloc, idx12));
                    }
                }
            }
        }
    }
    
    ctx_.comm().allgather(&f_pw[0], (int)rl->spl_num_gvec().global_offset(), (int)rl->spl_num_gvec().local_size());

    for (int ig = 0; ig < rl->num_gvec(); ig++) rho_->f_pw(ig) += f_pw[ig];
}

#ifdef _GPU_

extern "C" void sum_q_pw_d_mtrx_pw_gpu(int num_gvec_loc,
                                       int num_beta,
                                       void* q_pw_t,
                                       void* dm_g,
                                       void* rho_pw);

extern "C" void generate_d_mtrx_pw_gpu(int num_atoms,
                                       int num_gvec_loc,
                                       int num_beta,
                                       double* atom_pos,
                                       int* gvec,
                                       void* d_mtrx_packed,
                                       void* d_mtrx_pw);

void Density::add_q_contribution_to_valence_density_gpu(K_set& ks)
{
    Timer t("sirius::Density::add_q_contribution_to_valence_density_gpu", ctx_.comm());

    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
     * because density matrix is symmetric.
     */
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    /* complex density matrix */
    mdarray<double_complex, 4> pp_complex_density_matrix(unit_cell_.max_mt_basis_size(), 
                                                         unit_cell_.max_mt_basis_size(),
                                                         num_zdmat, unit_cell_.num_atoms());
    pp_complex_density_matrix.zero();
    //pp_complex_density_matrix.allocate_on_device();
    //pp_complex_density_matrix.zero_on_device();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        std::vector< std::pair<int, double> > occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);

        add_k_point_contribution<GPU, ultrasoft_pseudopotential>(ks[ik], occupied_bands, pp_complex_density_matrix);
    }
    //pp_complex_density_matrix.copy_to_host();
    //pp_complex_density_matrix.deallocate_on_device();

    //parameters_.mpi_grid().communicator(1 << _dim_k_ | 1 << _dim_col_).allreduce(pp_complex_density_matrix.at<CPU>(), 
    //                                                                             (int)pp_complex_density_matrix.size());

    ctx_.comm().allreduce(pp_complex_density_matrix.at<CPU>(), (int)pp_complex_density_matrix.size());

    auto rl = ctx_.reciprocal_lattice();

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
         auto type = unit_cell_.atom_type(iat);
         type->uspp().q_pw.allocate_on_device();
         type->uspp().q_pw.copy_to_device();
    }

    mdarray<int, 2> gvec(3, rl->spl_num_gvec().local_size());
    for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
    {
        for (int x = 0; x < 3; x++) gvec(x, igloc) = rl->gvec(rl->spl_num_gvec(igloc))[x];
    }
    gvec.allocate_on_device();
    gvec.copy_to_device();

    std::vector<double_complex> rho_pw(rl->num_gvec(), double_complex(0, 0));
    mdarray<double_complex, 1> rho_pw_gpu(&rho_pw[rl->spl_num_gvec().global_offset()], rl->spl_num_gvec().local_size());
    rho_pw_gpu.allocate_on_device();
    rho_pw_gpu.zero_on_device();

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto type = unit_cell_.atom_type(iat);
        int nbf = type->mt_basis_size();

        mdarray<double_complex, 2> d_mtrx_packed(nbf * nbf, type->num_atoms());
        mdarray<double, 2> atom_pos(type->num_atoms(), 3);
        #pragma omp parallel for
        for (int i = 0; i < type->num_atoms(); i++)
        {
            int ia = type->atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    d_mtrx_packed(xi2 * nbf + xi1, i) = pp_complex_density_matrix(xi2, xi1, 0, ia);
                }
            }
            for (int x = 0; x < 3; x++) atom_pos(i, x) = unit_cell_.atom(ia)->position(x);
        }
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        atom_pos.allocate_on_device();
        atom_pos.copy_to_device();

        mdarray<double_complex, 2> d_mtrx_pw(nullptr, rl->spl_num_gvec().local_size(), nbf * nbf);
        d_mtrx_pw.allocate_on_device();
        d_mtrx_pw.zero_on_device();

        generate_d_mtrx_pw_gpu(type->num_atoms(),
                               (int)rl->spl_num_gvec().local_size(),
                               nbf,
                               atom_pos.at<GPU>(),
                               gvec.at<GPU>(),
                               d_mtrx_packed.at<GPU>(),
                               d_mtrx_pw.at<GPU>());

        sum_q_pw_d_mtrx_pw_gpu((int)rl->spl_num_gvec().local_size(), 
                               nbf,
                               type->uspp().q_pw.at<GPU>(),
                               d_mtrx_pw.at<GPU>(),
                               rho_pw_gpu.at<GPU>());
    }

    rho_pw_gpu.copy_to_host();

    ctx_.comm().allgather(&rho_pw[0], (int)rl->spl_num_gvec().global_offset(), (int)rl->spl_num_gvec().local_size());
    
    for (int ig = 0; ig < rl->num_gvec(); ig++) rho_->f_pw(ig) += rho_pw[ig];
    
    //fft_->input(rl->num_gvec(), rl->fft_index(), &rho_pw[0]);
    //fft_->transform(1);
    //for (int ir = 0; ir < fft_->size(); ir++) rho_->f_it<global>(ir) += real(fft_->buffer(ir));
    
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) unit_cell_.atom_type(iat)->uspp().q_pw.deallocate_on_device();
}
#endif

};

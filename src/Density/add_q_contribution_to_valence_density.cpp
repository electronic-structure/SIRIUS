#include "density.h"

#ifdef __GPU
extern "C" void generate_dm_pw_gpu(int num_atoms__,
                                   int num_gvec_loc__,
                                   int num_beta__,
                                   double const* atom_pos__,
                                   int const* gvec__,
                                   double_complex const* dm__,
                                   double_complex* dm_pw__);

extern "C" void sum_q_pw_dm_pw_gpu(int num_gvec_loc__,
                                   int nbf__,
                                   double_complex const* q_pw_t__,
                                   double_complex const* dm_pw__,
                                   double_complex* rho_pw__);
#endif

namespace sirius {

void Density::add_q_contribution_to_valence_density(K_set& ks)
{
    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
     * because density matrix is symmetric. */
    int ndm = (ctx_.num_mag_dims() == 3) ? 3 : ctx_.num_spins();

    /* complex density matrix */
    mdarray<double_complex, 4> density_matrix(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(),
                                              ndm, unit_cell_.num_atoms());
    density_matrix.zero();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        add_k_point_contribution<ultrasoft_pseudopotential>(ks[ik], density_matrix);
    }
    ctx_.comm().allreduce(density_matrix.at<CPU>(), static_cast<int>(density_matrix.size()));

    /* split G-vectors between ranks */
    splindex<block> spl_gvec(ctx_.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());

    std::vector<Periodic_function<double>*> rho_vec(ctx_.num_mag_dims() + 1);
    rho_vec[0] = rho_;
    for (int j = 0; j < ctx_.num_mag_dims(); j++) rho_vec[1 + j] = magnetization_[j];

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++)
        ctx_.augmentation_op(iat).prepare();

    #ifdef __GPU
    mdarray<int, 2> gvec;
    mdarray<double_complex, 2> rho_pw_gpu;
    if (ctx_.processing_unit() == GPU)
    {
        gvec = mdarray<int, 2>(3, spl_gvec.local_size());
        for (int igloc = 0; igloc < (int)spl_gvec.local_size(); igloc++)
        {
            for (int x = 0; x < 3; x++) gvec(x, igloc) = ctx_.gvec()[(int)spl_gvec[igloc]][x];
        }
        gvec.allocate_on_device();
        gvec.copy_to_device();
        
        rho_pw_gpu = mdarray<double_complex, 2>(spl_gvec.local_size(), ctx_.num_mag_dims() + 1);
        rho_pw_gpu.allocate_on_device();
        rho_pw_gpu.zero_on_device();
    }
    #endif

    //mdarray<double, 2> timers(3, Platform::max_num_threads());
    //timers.zero();

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type.mt_basis_size();

        mdarray<double_complex, 3> dm(nbf * nbf, atom_type.num_atoms(), ndm);
        #pragma omp parallel for
        for (int i = 0; i < atom_type.num_atoms(); i++)
        {
            int ia = atom_type.atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    switch (ctx_.num_mag_dims())
                    {
                        case 0:
                        {
                            dm(xi2 * nbf + xi1, i, 0) = density_matrix(xi2, xi1, 0, ia);
                            break;
                        }
                        case 1:
                        {
                            dm(xi2 * nbf + xi1, i, 0) = density_matrix(xi2, xi1, 0, ia) + density_matrix(xi2, xi1, 1, ia);
                            dm(xi2 * nbf + xi1, i, 1) = density_matrix(xi2, xi1, 0, ia) - density_matrix(xi2, xi1, 1, ia);
                            break;
                        }
                    }
                }
            }
        }

        if (ctx_.processing_unit() == CPU)
        {
            mdarray<double_complex, 2> phase_factors(atom_type.num_atoms(), spl_gvec.local_size());

            #pragma omp parallel for
            for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
            {
                int ig = spl_gvec[igloc];
                for (int i = 0; i < atom_type.num_atoms(); i++)
                {
                    int ia = atom_type.atom_id(i);
                    phase_factors(i, igloc) = std::conj(ctx_.gvec_phase_factor(ig, ia));
                }
            }

            mdarray<double_complex, 2> dm_pw(nbf * nbf, spl_gvec.local_size());

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++)
            {
                linalg<CPU>::gemm(0, 0, nbf * nbf, spl_gvec.local_size(), atom_type.num_atoms(),
                                  &dm(0, 0, iv), dm.ld(), &phase_factors(0, 0), phase_factors.ld(),
                                  &dm_pw(0, 0), dm.ld());

                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
                {
                    double_complex z(0, 0);

                    /* remember that dm_pw is not a Hermitian matrix in xi1,xi2 indices */
                    for (int xi2 = 0; xi2 < nbf; xi2++)
                    {
                        int idx12 = xi2 * (xi2 + 1) / 2;

                        /* add diagonal term */
                        /* D_{xi2,xi2} * Q(G)_{xi2, xi2} */
                        z += dm_pw(xi2 * nbf + xi2, igloc) * ctx_.augmentation_op(iat).q_pw(idx12 + xi2, igloc);

                        /* add non-diagonal terms */
                        for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
                        {
                            double_complex q = ctx_.augmentation_op(iat).q_pw(idx12, igloc);

                            /* D_{xi2,xi1} * Q(G)_{xi1, xi2} + D_{xi1,xi2} * Q(G)_{xix, xi1}^{+} */
                            z += (dm_pw(xi2 * nbf + xi1, igloc) * q + dm_pw(xi1 * nbf + xi2, igloc) * std::conj(q));
                        }
                    }
                    rho_vec[iv]->f_pw(igloc) += z;
                }
            }
        }

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU)
        {
            mdarray<double, 2> atom_pos(3, atom_type.num_atoms());
            #pragma omp parallel for
            for (int i = 0; i < atom_type.num_atoms(); i++)
            {
                int ia = atom_type.atom_id(i);
                auto pos = unit_cell_.atom(ia).position();
                for (int x: {0, 1, 2}) atom_pos(x, i) = pos[x];
            }
            atom_pos.allocate_on_device();
            atom_pos.copy_to_device();

            dm.allocate_on_device();
            dm.copy_to_device();

            mdarray<double_complex, 2> dm_pw_gpu(nullptr, nbf * nbf, spl_gvec.local_size());
            dm_pw_gpu.allocate_on_device();

            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++)
            {
                generate_dm_pw_gpu(atom_type.num_atoms(),
                                   spl_gvec.local_size(),
                                   nbf,
                                   atom_pos.at<GPU>(),
                                   gvec.at<GPU>(),
                                   dm.at<GPU>(0, 0, iv),
                                   dm_pw_gpu.at<GPU>());
                
                sum_q_pw_dm_pw_gpu(spl_gvec.local_size(), 
                                   nbf,
                                   ctx_.augmentation_op(iat).q_pw().at<GPU>(),
                                   dm_pw_gpu.at<GPU>(),
                                   rho_pw_gpu.at<GPU>(0, iv));
            }
        }
        #endif
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
    {
        rho_pw_gpu.copy_to_host();
        for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++)
        {
            #pragma omp parallel for
            for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
                rho_vec[iv]->f_pw(igloc) += rho_pw_gpu(igloc, iv);
        }
    }
    #endif

    //if (ctx_.comm().rank() == 0)
    //{
    //    std::cout << "-------------------------------------------" << std::endl;
    //    std::cout << "thread_id  | phase    | zgemm    | update  " << std::endl;
    //    std::cout << "-------------------------------------------" << std::endl;
    //    for (int i = 0; i < Platform::max_num_threads(); i++)
    //    {
    //        printf("   %2i      | %8.4f | %8.4f | %8.4f \n", i, timers(0, i), timers(1, i), timers(2, i));
    //    }
    //    std::cout << "-------------------------------------------" << std::endl;
    //}
    
    for (auto e: rho_vec)
        ctx_.comm().allgather(&e->f_pw(0), spl_gvec.global_offset(), spl_gvec.local_size());

    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++)
        ctx_.augmentation_op(iat).dismiss();
}

};

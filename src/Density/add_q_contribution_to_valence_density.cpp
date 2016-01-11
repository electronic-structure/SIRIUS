#include "density.h"

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
}

#ifdef __GPU

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
    STOP();

//    /* If we have ud and du spin blocks, don't compute one of them (du in this implementation)
//     * because density matrix is symmetric.  */
//    int num_zdmat = (ctx_.num_mag_dims() == 3) ? 3 : (ctx_.num_mag_dims() + 1);
//
//    /* complex density matrix */
//    mdarray<double_complex, 4> pp_complex_density_matrix(unit_cell_.max_mt_basis_size(), 
//                                                         unit_cell_.max_mt_basis_size(),
//                                                         num_zdmat, unit_cell_.num_atoms());
//    pp_complex_density_matrix.zero();
//    
//    /* add k-point contribution */
//    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
//    {
//        int ik = ks.spl_num_kpoints(ikloc);
//        auto kp = ks[ik];
//
//        add_k_point_contribution<ultrasoft_pseudopotential>(kp, pp_complex_density_matrix);
//    }
//
//    ctx_.comm().allreduce(pp_complex_density_matrix.at<CPU>(), (int)pp_complex_density_matrix.size());
//
//    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
//    {
//         auto& type = const_cast<Atom_type&>(unit_cell_.atom_type(iat));
//         type.uspp().q_pw.allocate_on_device();
//         type.uspp().q_pw.copy_to_device();
//    }
//
//    /* split G-vectors between ranks */
//    splindex<block> spl_gvec(ctx_.gvec().num_gvec(), ctx_.comm().size(), ctx_.comm().rank());
//
//    mdarray<int, 2> gvec(3, spl_gvec.local_size());
//    for (int igloc = 0; igloc < (int)spl_gvec.local_size(); igloc++)
//    {
//        for (int x = 0; x < 3; x++) gvec(x, igloc) = ctx_.gvec()[(int)spl_gvec[igloc]][x];
//    }
//    gvec.allocate_on_device();
//    gvec.copy_to_device();
//
//    mdarray<double_complex, 1> rho_pw_gpu(spl_gvec.local_size());
//    rho_pw_gpu.allocate_on_device();
//    rho_pw_gpu.zero_on_device();
//
//    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
//    {
//        auto& type = unit_cell_.atom_type(iat);
//        int nbf = type.mt_basis_size();
//
//        mdarray<double_complex, 2> d_mtrx_packed(nbf * nbf, type.num_atoms());
//        mdarray<double, 2> atom_pos(type.num_atoms(), 3);
//        #pragma omp parallel for
//        for (int i = 0; i < type.num_atoms(); i++)
//        {
//            int ia = type.atom_id(i);
//
//            for (int xi2 = 0; xi2 < nbf; xi2++)
//            {
//                for (int xi1 = 0; xi1 < nbf; xi1++)
//                {
//                    d_mtrx_packed(xi2 * nbf + xi1, i) = pp_complex_density_matrix(xi2, xi1, 0, ia);
//                }
//            }
//            auto pos = unit_cell_.atom(ia).position();
//            for (int x = 0; x < 3; x++) atom_pos(i, x) = pos[x];
//        }
//        d_mtrx_packed.allocate_on_device();
//        d_mtrx_packed.copy_to_device();
//        atom_pos.allocate_on_device();
//        atom_pos.copy_to_device();
//
//        mdarray<double_complex, 2> d_mtrx_pw(nullptr, spl_gvec.local_size(), nbf * nbf);
//        d_mtrx_pw.allocate_on_device();
//        d_mtrx_pw.zero_on_device();
//
//        generate_d_mtrx_pw_gpu(type.num_atoms(),
//                               (int)spl_gvec.local_size(),
//                               nbf,
//                               atom_pos.at<GPU>(),
//                               gvec.at<GPU>(),
//                               d_mtrx_packed.at<GPU>(),
//                               d_mtrx_pw.at<GPU>());
//
//        sum_q_pw_d_mtrx_pw_gpu((int)spl_gvec.local_size(), 
//                               nbf,
//                               const_cast<double_complex*>(type.uspp().q_pw.at<GPU>()),
//                               d_mtrx_pw.at<GPU>(),
//                               rho_pw_gpu.at<GPU>());
//    }
//
//    rho_pw_gpu.copy_to_host();
//    for (int igloc = 0; igloc < spl_gvec.local_size(); igloc++)
//        rho_->f_pw(spl_gvec[igloc]) += rho_pw_gpu(igloc);
//
//    ctx_.comm().allgather(&rho_->f_pw(0), (int)spl_gvec.global_offset(), (int)spl_gvec.local_size());
//    
//    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
//        const_cast<Atom_type&>(unit_cell_.atom_type(iat)).uspp().q_pw.deallocate_on_device();
}
#endif

};

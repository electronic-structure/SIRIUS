#include "density.h"

namespace sirius {

void Density::generate_valence_density_mt(K_set& ks)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence_density_mt");

    /* if we have ud and du spin blocks, don't compute one of them (du in this implementation)
       because density matrix is symmetric */
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    /* complex density matrix */
    mdarray<double_complex, 4> mt_complex_density_matrix(unit_cell_.max_mt_basis_size(), 
                                                         unit_cell_.max_mt_basis_size(),
                                                         num_zdmat, unit_cell_.num_atoms());
    mt_complex_density_matrix.zero();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        add_k_point_contribution<full_potential_lapwlo>(ks[ik], mt_complex_density_matrix);
    }
    
    mdarray<double_complex, 4> mt_complex_density_matrix_loc(unit_cell_.max_mt_basis_size(), 
                                                             unit_cell_.max_mt_basis_size(),
                                                             num_zdmat, unit_cell_.spl_num_atoms().local_size(0));
   
    for (int j = 0; j < num_zdmat; j++)
    {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int ialoc = (int)unit_cell_.spl_num_atoms().local_index(ia);
            int rank = unit_cell_.spl_num_atoms().local_rank(ia);

            ctx_.comm().reduce(&mt_complex_density_matrix(0, 0, j, ia), &mt_complex_density_matrix_loc(0, 0, j, ialoc),
                               unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size(), rank);
        }
    }
   
    /* compute occupation matrix */
    if (parameters_.uj_correction())
    {
        Timer t3("sirius::Density::generate:om");
        
        mdarray<double_complex, 4> occupation_matrix(16, 16, 2, 2); 
        
        for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            Atom_type* type = unit_cell_.atom(ia)->type();
            
            occupation_matrix.zero();
            for (int l = 0; l <= 3; l++)
            {
                int num_rf = type->indexr().num_rf(l);

                for (int j = 0; j < num_zdmat; j++)
                {
                    for (int order2 = 0; order2 < num_rf; order2++)
                    {
                    for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                    {
                        for (int order1 = 0; order1 < num_rf; order1++)
                        {
                        for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                        {
                            occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
                                mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
                                                              type->indexb_by_lm_order(lm2, order2), j, ialoc) *
                                unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
                        }
                        }
                    }
                    }
                }
            }
        
            // restore the du block
            for (int lm1 = 0; lm1 < 16; lm1++)
            {
                for (int lm2 = 0; lm2 < 16; lm2++)
                    occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
            }

            unit_cell_.atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        }

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int rank = unit_cell_.spl_num_atoms().local_rank(ia);
            unit_cell_.atom(ia)->sync_occupation_matrix(ctx_.comm(), rank);
        }
    }

    int max_num_rf_pairs = unit_cell_.max_mt_radial_basis_size() * 
                           (unit_cell_.max_mt_radial_basis_size() + 1) / 2;
    
    // real density matrix
    mdarray<double, 3> mt_density_matrix(parameters_.lmmax_rho(), max_num_rf_pairs, parameters_.num_mag_dims() + 1);
    
    mdarray<double, 2> rf_pairs(unit_cell_.max_num_mt_points(), max_num_rf_pairs);
    mdarray<double, 3> dlm(parameters_.lmmax_rho(), unit_cell_.max_num_mt_points(), 
                           parameters_.num_mag_dims() + 1);
    for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = (int)unit_cell_.spl_num_atoms(ialoc);
        Atom_type* atom_type = unit_cell_.atom(ia)->type();

        int nmtp = atom_type->num_mt_points();
        int num_rf_pairs = atom_type->mt_radial_basis_size() * (atom_type->mt_radial_basis_size() + 1) / 2;
        
        Timer t1("sirius::Density::generate|sum_zdens");
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                reduce_density_matrix<3>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 1:
            {
                reduce_density_matrix<1>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 0:
            {
                reduce_density_matrix<0>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
        }
        t1.stop();
        
        Timer t2("sirius::Density::generate|expand_lm");
        // collect radial functions
        for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
            {
                // off-diagonal pairs are taken two times: d_{12}*f_1*f_2 + d_{21}*f_2*f_1 = d_{12}*2*f_1*f_2
                int n = (idxrf1 == idxrf2) ? 1 : 2; 
                for (int ir = 0; ir < unit_cell_.atom(ia)->type()->num_mt_points(); ir++)
                {
                    rf_pairs(ir, offs + idxrf1) = n * unit_cell_.atom(ia)->symmetry_class()->radial_function(ir, idxrf1) * 
                                                      unit_cell_.atom(ia)->symmetry_class()->radial_function(ir, idxrf2); 
                }
            }
        }
        for (int j = 0; j < parameters_.num_mag_dims() + 1; j++)
        {
            linalg<CPU>::gemm(0, 1, parameters_.lmmax_rho(), nmtp, num_rf_pairs, 
                              &mt_density_matrix(0, 0, j), mt_density_matrix.ld(), 
                              &rf_pairs(0, 0), rf_pairs.ld(), &dlm(0, 0, j), dlm.ld());
        }

        int sz = parameters_.lmmax_rho() * nmtp * (int)sizeof(double);
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                memcpy(&magnetization_[1]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 2), sz); 
                memcpy(&magnetization_[2]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 3), sz);
            }
            case 1:
            {
                for (int ir = 0; ir < nmtp; ir++)
                {
                    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                    {
                        rho_->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        magnetization_[0]->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0:
            {
                memcpy(&rho_->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 0), sz);
            }
        }
        t2.stop();
    }
}

};

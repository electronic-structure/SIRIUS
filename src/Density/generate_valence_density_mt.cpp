#include "density.h"
#include <fstream>

using namespace std;

namespace sirius
{

void Density::generate_valence_density_mt(K_set& ks)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence_density_mt");

    /* if we have ud and du spin blocks, don't compute one of them (du in this implementation)
       because density matrix is symmetric */
    int num_zdmat = (ctx_.num_mag_dims() == 3) ? 3 : (ctx_.num_mag_dims() + 1);

    /* complex density matrix */
    mdarray<double_complex, 4> mt_complex_density_matrix(unit_cell_.max_mt_basis_size(), 
                                                         unit_cell_.max_mt_basis_size(),
                                                         num_zdmat, unit_cell_.num_atoms());
    mt_complex_density_matrix.zero();
    
    /* add k-point contribution */
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        add_k_point_contribution_mt(ks[ik], mt_complex_density_matrix);
    }
    
    mdarray<double_complex, 4> mt_complex_density_matrix_loc(unit_cell_.max_mt_basis_size(), 
                                                             unit_cell_.max_mt_basis_size(),
                                                             num_zdmat, unit_cell_.spl_num_atoms().local_size(0));
   
    for (int j = 0; j < num_zdmat; j++)
    {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int ialoc = unit_cell_.spl_num_atoms().local_index(ia);
            int rank = unit_cell_.spl_num_atoms().local_rank(ia);

            ctx_.comm().reduce(&mt_complex_density_matrix(0, 0, j, ia), &mt_complex_density_matrix_loc(0, 0, j, ialoc),
                               unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size(), rank);
        }
    }
   
    /* compute occupation matrix */
    if (ctx_.uj_correction())
    {
        STOP();

        // TODO: fix the way how occupation matrix is calculated

        //Timer t3("sirius::Density::generate:om");
        //
        //mdarray<double_complex, 4> occupation_matrix(16, 16, 2, 2); 
        //
        //for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
        //{
        //    int ia = unit_cell_.spl_num_atoms(ialoc);
        //    Atom_type* type = unit_cell_.atom(ia)->type();
        //    
        //    occupation_matrix.zero();
        //    for (int l = 0; l <= 3; l++)
        //    {
        //        int num_rf = type->indexr().num_rf(l);

        //        for (int j = 0; j < num_zdmat; j++)
        //        {
        //            for (int order2 = 0; order2 < num_rf; order2++)
        //            {
        //            for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
        //            {
        //                for (int order1 = 0; order1 < num_rf; order1++)
        //                {
        //                for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
        //                {
        //                    occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
        //                        mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
        //                                                      type->indexb_by_lm_order(lm2, order2), j, ialoc) *
        //                        unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
        //                }
        //                }
        //            }
        //            }
        //        }
        //    }
        //
        //    // restore the du block
        //    for (int lm1 = 0; lm1 < 16; lm1++)
        //    {
        //        for (int lm2 = 0; lm2 < 16; lm2++)
        //            occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
        //    }

        //    unit_cell_.atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        //}

        //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //{
        //    int rank = unit_cell_.spl_num_atoms().local_rank(ia);
        //    unit_cell_.atom(ia)->sync_occupation_matrix(ctx_.comm(), rank);
        //}
    }

    int max_num_rf_pairs = unit_cell_.max_mt_radial_basis_size() * 
                           (unit_cell_.max_mt_radial_basis_size() + 1) / 2;
    
    // real density matrix
    mdarray<double, 3> mt_density_matrix(ctx_.lmmax_rho(), max_num_rf_pairs, ctx_.num_mag_dims() + 1);
    
    mdarray<double, 2> rf_pairs(unit_cell_.max_num_mt_points(), max_num_rf_pairs);
    mdarray<double, 3> dlm(ctx_.lmmax_rho(), unit_cell_.max_num_mt_points(), 
                           ctx_.num_mag_dims() + 1);
    for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = (int)unit_cell_.spl_num_atoms(ialoc);
        auto& atom_type = unit_cell_.atom(ia).type();

        int nmtp = atom_type.num_mt_points();
        int num_rf_pairs = atom_type.mt_radial_basis_size() * (atom_type.mt_radial_basis_size() + 1) / 2;
        
        runtime::Timer t1("sirius::Density::generate|sum_zdens");
        switch (ctx_.num_mag_dims())
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
        
        runtime::Timer t2("sirius::Density::generate|expand_lm");
        /* collect radial functions */
        for (int idxrf2 = 0; idxrf2 < atom_type.mt_radial_basis_size(); idxrf2++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
            {
                /* off-diagonal pairs are taken two times: d_{12}*f_1*f_2 + d_{21}*f_2*f_1 = d_{12}*2*f_1*f_2 */
                int n = (idxrf1 == idxrf2) ? 1 : 2; 
                for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++)
                {
                    rf_pairs(ir, offs + idxrf1) = n * unit_cell_.atom(ia).symmetry_class().radial_function(ir, idxrf1) * 
                                                      unit_cell_.atom(ia).symmetry_class().radial_function(ir, idxrf2); 
                }
            }
        }
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++)
        {
            linalg<CPU>::gemm(0, 1, ctx_.lmmax_rho(), nmtp, num_rf_pairs, 
                              &mt_density_matrix(0, 0, j), mt_density_matrix.ld(), 
                              &rf_pairs(0, 0), rf_pairs.ld(), &dlm(0, 0, j), dlm.ld());
        }

        int sz = static_cast<int>(ctx_.lmmax_rho() * nmtp * sizeof(double));
        switch (ctx_.num_mag_dims())
        {
            case 3:
            {
                std::memcpy(&magnetization_[1]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 2), sz); 
                std::memcpy(&magnetization_[2]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 3), sz);
            }
            case 1:
            {
                for (int ir = 0; ir < nmtp; ir++)
                {
                    for (int lm = 0; lm < ctx_.lmmax_rho(); lm++)
                    {
                        rho_->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        magnetization_[0]->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0:
            {
                std::memcpy(&rho_->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 0), sz);
            }
        }
    }
}





void Density::generate_paw_loc_density()
{
//	ofstream of("loc_density.txt");
	//of<<"========================================"<<endl;

	for(int ia = 0; ia < unit_cell_.num_atoms(); ia++)
	{
		auto& atom = unit_cell_.atom(ia);

		auto& atom_type = atom.type();

		auto& paw = atom_type.get_PAW_descriptor();

		auto& uspp = atom_type.uspp();

		std::vector<int> l_by_lm = Utils::l_by_lm( 2 * atom_type.indexr().lmax_lo() );

		//TODO calculate not for every atom but for every atom type
		Gaunt_coefficients<double> GC(atom_type.indexr().lmax_lo(),
				2*atom_type.indexr().lmax_lo(),
				atom_type.indexr().lmax_lo(),
				SHT::gaunt_rlm);

		// get density for current atom
		mdarray<double,2> &ae_atom_density = paw_ae_local_density_[ia];
		mdarray<double,2> &ps_atom_density = paw_ps_local_density_[ia];

		// and magnetization
		auto &ae_atom_magnetization = paw_ae_local_magnetization_[ia];
		auto &ps_atom_magnetization = paw_ps_local_magnetization_[ia];

		// get radial grid to divide density over r^2
		auto &grid = atom_type.radial_grid();

		// iterate over spin components
//		for(int ispin = 0; ispin < (int)ae_atom_density.size(2); ispin++)
//		{
//			double spin_sgn = 1.0;
//
//			if(ispin==1) spin_sgn = -1.0;
//			if(ispin>1) spin_sgn = 0.0;

			// iterate over local basis functions (or over lm1 and lm2)
			for(int ib2 = 0; ib2 < (int)atom_type.indexb().size(); ib2++)
			{
				for(int ib1 = 0; ib1 <= ib2; ib1++)
				{
					// get lm quantum numbers (lm index) of the basis functions
					int lm2 = atom_type.indexb(ib2).lm;
					int lm1 = atom_type.indexb(ib1).lm;

					//get radial basis functions indices
					int irb2 = atom_type.indexb(ib2).idxrf;
					int irb1 = atom_type.indexb(ib1).idxrf;

					// index to iterate Qij,
					// TODO check indices
					int iqij = irb2 * (irb2 + 1) / 2 + irb1;

					// get num of non-zero GC
					int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

					// add nonzero coefficients
					for(int inz = 0; inz < num_non_zero_gk; inz++)
					{
						auto& lm3coef = GC.gaunt(lm1,lm2,inz);

						if(lm3coef.lm3 >= (int)ae_atom_density.size(1))
						{
							TERMINATE("PAW: lm3 index out of range of lm part of density array");
						}

						// iterate over radial points
						// size of ps and ae must be equal TODO: if not?
						// this part in fortran looks better, is there the same for c++?
						for(int irad = 0; irad < (int)grid.num_points(); irad++)
						{
							// we need to divide density over r^2 since wave functions are stored multiplied by r
							double inv_r2 = 1./(grid[irad] * grid[irad]);

							// TODO for 3 spin dimensions 3th density spin component must be complex
							// replace order of indices for density from {irad,lm} to {lm,irad}
							// to be in according with ELK and other SIRIUS code
							double ae_part = inv_r2 * lm3coef.coef * paw.all_elec_wfc(irad,irb1) * paw.all_elec_wfc(irad,irb2);
							double ps_part = inv_r2 * lm3coef.coef *
									( paw.pseudo_wfc(irad,irb1) * paw.pseudo_wfc(irad,irb2)  + uspp.q_radial_functions_l(irad,iqij,l_by_lm[lm3coef.lm3]));

							// calculate UP density (or total in case of nonmagnetic)
							double ae_dens_u = density_matrix_(ib1,ib2,0,ia).real() * ae_part;
							double ps_dens_u = density_matrix_(ib1,ib2,0,ia).real() * ps_part;

							// add density UP to the total density
							ae_atom_density(lm3coef.lm3,irad) += ae_dens_u;
							ps_atom_density(lm3coef.lm3,irad) += ps_dens_u;

							switch(ctx_.num_spins())
							{
								case 2:
								{
									double ae_dens_d = density_matrix_(ib1,ib2,1,ia).real() * ae_part;
									double ps_dens_d = density_matrix_(ib1,ib2,1,ia).real() * ps_part;

									// add density DOWN to the total density
									ae_atom_density(lm3coef.lm3,irad) += ae_dens_d;
									ps_atom_density(lm3coef.lm3,irad) += ps_dens_d;

									// add magnetization to 2nd components (0th and 1st are always zero )
									ae_atom_magnetization(lm3coef.lm3,irad,2)= ae_dens_u - ae_dens_d;
									ps_atom_magnetization(lm3coef.lm3,irad,2)= ps_dens_u - ps_dens_d;
								}break;

								case 3:
								{
									TERMINATE("PAW: non collinear is not implemented");
								}break;

								default:break;
							}


						}
					}
				}
			}
//		}


//		ofstream of("loc_density.txt");
//
//
//
//		for(int j = 0; j< ae_atom_density.size(0); j++)
//		{
//			for(int i = 0; i< ae_atom_density.size(1); i++)
//			{
//				of<< ae_atom_density(j,i) << " " << ps_atom_density(j,i) << endl;
//			}
//		}
//
//
//		of.close();


	}

//
//
//	TERMINATE("terminated");
}

};

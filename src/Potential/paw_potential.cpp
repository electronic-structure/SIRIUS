/*
 * paw_potential.cpp
 *
 *  Created on: May 3, 2016
 *      Author: isivkov
 */

#include "Potential.h"

namespace sirius
{

void Potential::init_PAW()
{
	//--- allocate PAW potential array ---
	paw_hartree_energies_.resize(unit_cell_.num_atoms());
	paw_xc_energies_.resize(unit_cell_.num_atoms());
	paw_core_energies_.resize(unit_cell_.num_atoms());

	for(int ia = 0; ia < unit_cell_.num_atoms(); ia++)
	{
		auto& atom = unit_cell_.atom(ia);

		auto& atype = atom.type();

		int n_mt_points = atype.num_mt_points();

		int rad_func_lmax = atype.indexr().lmax_lo();

		// TODO am I right?
		int n_rho_lm_comp = (2 * rad_func_lmax + 1) * (2 * rad_func_lmax + 1);

		// allocate potential
		mdarray<double, 3> ae_atom_potential(n_rho_lm_comp, n_mt_points, ctx_.num_spins());
		mdarray<double, 3> ps_atom_potential(n_rho_lm_comp, n_mt_points, ctx_.num_spins());

		ae_paw_local_potential_.push_back(std::move(ae_atom_potential));
		ps_paw_local_potential_.push_back(std::move(ps_atom_potential));

		// allocate Dij
		//		mdarray<double, 2> atom_Dij( (atype.indexb().size() * (atype.indexb().size()+1)) / 2, ctx_.num_spins());

		//		paw_local_Dij_matrix_.push_back(std::move(atom_Dij)); valence_eval_sum

		paw_core_energies_[ia] = atype.get_PAW_descriptor().core_energy;
	}

	// separate because I can
	paw_total_core_energy_ = 0.0;

	for(int ia = 0; ia < unit_cell_.num_atoms(); ia++)
	{
		paw_total_core_energy_ += paw_core_energies_[ia];
	}
	paw_total_core_energy_ *=0.5;
}



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::generate_PAW_effective_potential(std::vector< mdarray<double, 2> > *paw_ae_local_density,
												 std::vector< mdarray<double, 2> > *paw_ps_local_density,
												 std::vector< mdarray<double, 3> > *paw_ae_local_magnetization,
												 std::vector< mdarray<double, 3> > *paw_ps_local_magnetization)
{
	for(int ia = 0; ia < unit_cell_.num_atoms(); ia++)
	{
		calc_PAW_local_potential(ia, paw_ae_local_density->at(ia),
								 paw_ps_local_density->at(ia),
								 paw_ae_local_magnetization->at(ia),
								 paw_ps_local_magnetization->at(ia));

		calc_PAW_local_Dij(ia);
	}

	// separate because I can
	paw_hartree_total_energy_ = 0.0;
	paw_xc_total_energy_ = 0.0;

	for(int ia = 0; ia < unit_cell_.num_atoms(); ia++)
	{
		paw_hartree_total_energy_ += paw_hartree_energies_[ia];
		paw_xc_total_energy_ += paw_xc_energies_[ia];
	}
}




//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
double Potential::xc_mt_PAW_nonmagnetic(const Radial_grid& rgrid,
									  mdarray<double, 3> &out_atom_pot,
									  mdarray<double, 2> &full_rho_lm,
									  const std::vector<double> &rho_core)
{
	Spheric_function<spectral,double> out_atom_pot_sf(&out_atom_pot(0,0,0), full_rho_lm.size(0), rgrid);

	Spheric_function<spectral,double> full_rho_lm_sf(&full_rho_lm(0,0), full_rho_lm.size(0), rgrid);

	Spheric_function<spectral,double> full_rho_lm_sf_new(full_rho_lm.size(0), rgrid);

	full_rho_lm_sf_new.zero();
	full_rho_lm_sf_new += full_rho_lm_sf;

	double invY00 = 1. / y00 ;

	for(int ir = 0; ir < rgrid.num_points(); ir++ )
	{
		full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
	}

	Spheric_function<spatial,double> full_rho_tp_sf = transform(sht_,full_rho_lm_sf_new);

	// create potential in theta phi
	Spheric_function<spatial,double> vxc_tp_sf(sht_->num_points(), rgrid);

	// create energy in theta phi
	Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

	xc_mt_nonmagnetic(rgrid, xc_func_, full_rho_lm_sf_new, full_rho_tp_sf, vxc_tp_sf, exc_tp_sf);

	out_atom_pot_sf += transform(sht_,vxc_tp_sf);

	//------------------------
	//--- calculate energy ---
	//------------------------
	Spheric_function<spectral,double> exc_lm_sf = transform( sht_, exc_tp_sf );

	return inner(exc_lm_sf, full_rho_lm_sf_new);
}



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
double Potential::xc_mt_PAW_collinear(const Radial_grid& rgrid,
									mdarray<double,3> &out_atom_pot,
									mdarray<double,2> &full_rho_lm,
									mdarray<double,3> &magnetization_lm,
									const std::vector<double> &rho_core)
{
	int lmsize_rho = full_rho_lm.size(0);

	// make spherical functions for input density
	Spheric_function<spectral,double> full_rho_lm_sf(&full_rho_lm(0,0),lmsize_rho,rgrid);

	Spheric_function<spectral,double> full_rho_lm_sf_new(full_rho_lm.size(0), rgrid);

	full_rho_lm_sf_new.zero();
	full_rho_lm_sf_new += full_rho_lm_sf;

	double invY00 = 1. / y00 ;

	for(int ir = 0; ir < rgrid.num_points(); ir++ )
	{
		full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
	}

	// make spherical functions for output potential
	Spheric_function<spectral,double> out_atom_pot_up_sf(&out_atom_pot(0,0,0),lmsize_rho,rgrid);
	Spheric_function<spectral,double> out_atom_pot_dn_sf(&out_atom_pot(0,0,1),lmsize_rho,rgrid);

	// make magnetization from z component in lm components
	Spheric_function<spectral,double> magnetization_Z_lm(&magnetization_lm(0,0,2), lmsize_rho, rgrid );

	// calculate spin up spin down density components in lm components
	// up = 1/2 ( rho + magn );  down = 1/2 ( rho - magn )
	Spheric_function<spectral,double> rho_u_lm_sf =  0.5 * (full_rho_lm_sf_new + magnetization_Z_lm);
	Spheric_function<spectral,double> rho_d_lm_sf =  0.5 * (full_rho_lm_sf_new - magnetization_Z_lm);

	// transform density to theta phi components
	Spheric_function<spatial,double> rho_u_tp_sf = transform( sht_, rho_u_lm_sf );
	Spheric_function<spatial,double> rho_d_tp_sf = transform( sht_, rho_d_lm_sf );

	// create potential in theta phi
	Spheric_function<spatial,double> vxc_u_tp_sf(sht_->num_points(), rgrid);
	Spheric_function<spatial,double> vxc_d_tp_sf(sht_->num_points(), rgrid);

	// create energy in theta phi
	Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

	// calculate XC
	xc_mt_magnetic(rgrid, xc_func_,
				   rho_u_lm_sf, rho_u_tp_sf,
				   rho_d_lm_sf, rho_d_tp_sf,
				   vxc_u_tp_sf, vxc_d_tp_sf,
				   exc_tp_sf);

	// transform back in lm
	out_atom_pot_up_sf += transform( sht_, vxc_u_tp_sf);
	out_atom_pot_dn_sf += transform( sht_, vxc_d_tp_sf);
//////////////////////////////////////////////////////////////
//	std::cout<<"\n UP "<<std::endl;
//	for(int lm=0; lm<out_atom_pot_up_sf.angular_domain_size(); lm++)
//	{
//		for(int ir = 0; ir < rgrid.num_points(); ir++ )
//		{
//			if(full_rho_lm_sf(lm,ir) > 50)
//			std::cout<<full_rho_lm_sf(lm,ir)<<" ";
//		}
//	}
//	std::cout<<"\n POT "<<std::endl;
//	for(int lm=0; lm<out_atom_pot_up_sf.angular_domain_size(); lm++)
//	{
//		for(int ir = 0; ir < rgrid.num_points(); ir++ )
//		{
//			if(out_atom_pot_dn_sf(lm,ir) > 50)
//			std::cout<<out_atom_pot_dn_sf(lm,ir)<<" ";
//		}
//	}
//////////////////////////////////////////////////////////////

	//------------------------
	//--- calculate energy ---
	//------------------------
	Spheric_function<spectral,double> exc_lm_sf = transform( sht_, exc_tp_sf );

	return inner(exc_lm_sf, full_rho_lm_sf_new);
}





//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
double Potential::calc_PAW_hartree_potential(Atom& atom, const Radial_grid& grid,
											 mdarray<double, 2> &full_density,
											 mdarray<double, 3> &out_atom_pot)
{
	//---------------------
	//-- calc potential --
	//---------------------
	int lmsize_rho = full_density.size(0);

	Spheric_function<function_domain_t::spectral,double> dens_sf(&full_density(0,0), lmsize_rho, grid);

	// make spher funcs from arrays
	std::vector< Spheric_function<spectral,double> > atom_pot_sfs;

	for(int i=0;i<ctx_.num_spins();i++)
	{
		Spheric_function<spectral,double> atom_pot_sf(&out_atom_pot(0,0,i), lmsize_rho, grid);

		atom_pot_sfs.push_back(std::move(atom_pot_sf));
	}

	// array passed to poisson solver
	Spheric_function<spectral,double> atom_pot_sf(lmsize_rho, grid);
	atom_pot_sf.zero();

	// create qmt to store multipoles
	mdarray<double_complex,1> qmt(lmsize_rho);

	// solve poisson eq and fill 0th spin component of hartree array (in nonmagnetic we have only this)
	qmt.zero();
	poisson_atom_vmt(dens_sf, atom_pot_sf, qmt, atom);

	// if we have collinear megnetic states we need to add the same Hartree potential to DOWN-DOWN channel
	atom_pot_sfs[0] += atom_pot_sf;

	if(ctx_.num_spins() == 2)
	{
		atom_pot_sfs[1] += atom_pot_sf;
	}

	//---------------------
	//--- calc energy ---
	//---------------------
	std::vector<int> l_by_lm = Utils::l_by_lm( Utils::lmax_by_lmmax(lmsize_rho) );

	// create array for integration
	std::vector<double> intdata(grid.num_points(),0);

	double hartree_energy=0.0;

	for(int lm=0; lm < lmsize_rho; lm++)
	{
		// fill data to integrate
		for(int irad = 0; irad < grid.num_points(); irad++)
		{
			intdata[irad] = full_density(lm,irad) * out_atom_pot(lm, irad, 0) * grid[irad] * grid[irad];
		}

		// create spline from the data
		Spline<double> h_spl(grid,intdata);

		hartree_energy += 0.5 * h_spl.integrate(0);
	}

	return hartree_energy;
}



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::calc_PAW_local_potential(int atom_index,
										 mdarray<double, 2> &ae_full_density,
										 mdarray<double, 2> &ps_full_density,
										 mdarray<double, 3> &ae_local_magnetization,
										 mdarray<double, 3> &ps_local_magnetization)
{
	auto& atom = unit_cell_.atom(atom_index);

	auto& atom_type = atom.type();

	auto& paw = atom_type.get_PAW_descriptor();
	auto& uspp = atom_type.uspp();

	//-----------------------------------------
	//---- Calculation of Hartree potential ---
	//-----------------------------------------

	ae_paw_local_potential_[atom_index].zero();
	ps_paw_local_potential_[atom_index].zero();

	double ae_hartree_energy = calc_PAW_hartree_potential(atom,
														  atom.radial_grid(),
														  ae_full_density,
														  ae_paw_local_potential_[atom_index]);

	double ps_hartree_energy = calc_PAW_hartree_potential(atom,
														  atom.radial_grid(),
														  ps_full_density,
														  ps_paw_local_potential_[atom_index]);

	paw_hartree_energies_[atom_index] = ae_hartree_energy - ps_hartree_energy;

	//-----------------------------------------
	//---- Calculation of XC potential ---
	//-----------------------------------------
	double ae_xc_energy = 0.0;
	double ps_xc_energy = 0.0;

	switch(ctx_.num_spins())
	{
		case 1:
		{
			ae_xc_energy = xc_mt_PAW_nonmagnetic(atom.radial_grid(), ae_paw_local_potential_[atom_index],
														ae_full_density ,paw.all_elec_core_charge);

			ps_xc_energy = xc_mt_PAW_nonmagnetic(atom.radial_grid(), ps_paw_local_potential_[atom_index],
														ps_full_density,uspp.core_charge_density);
		}break;

		case 2:
		{
			ae_xc_energy = xc_mt_PAW_collinear(atom.radial_grid(), ae_paw_local_potential_[atom_index], ae_full_density,
								ae_local_magnetization, paw.all_elec_core_charge);

			ps_xc_energy = xc_mt_PAW_collinear(atom.radial_grid(), ps_paw_local_potential_[atom_index], ps_full_density,
								ps_local_magnetization, uspp.core_charge_density);
		}break;

		case 3:
		{
			xc_mt_PAW_noncollinear();
			TERMINATE("PAW potential ERROR! Non-collinear is not implemented");
		}break;

		default:
		{
			TERMINATE("PAW local potential error! Wrong number of spins!")
		}break;
	}

	////////////////////////////////////////////////////////////////
//		for(int i=0;i<ps_paw_local_potential_[atom_index].size(1);i++)
//			for(int j=0;j<ps_paw_local_potential_[atom_index].size(0);j++)
//				std::cout<<ps_paw_local_potential_[atom_index](j,i,0) - ps_paw_local_potential_[atom_index](j,i,1) << " ";
//
//		std::cout<<std::endl;
	////////////////////////////////////////////////////////////////

	paw_xc_energies_[atom_index] = ae_xc_energy - ps_xc_energy;
}




//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::calc_PAW_local_Dij(int atom_index)
{
	auto& atom = unit_cell_.atom(atom_index);

	auto& atom_type = atom.type();

	auto& paw = atom_type.get_PAW_descriptor();

	auto& uspp = atom_type.uspp();

	// get lm size for density
	int lmax = atom_type.indexr().lmax_lo();
	int lmsize_rho = ( lmax * 2 + 1) * ( lmax * 2 + 1);

	std::vector<int> l_by_lm = Utils::l_by_lm( 2 * lmax );

	//TODO calculate not for every atom but for every atom type
	Gaunt_coefficients<double> GC(lmax, 2*lmax, lmax, SHT::gaunt_rlm);

	auto &ae_atom_pot = ae_paw_local_potential_[atom_index];
	auto &ps_atom_pot = ps_paw_local_potential_[atom_index];

	//---- precalc integrals ----
	mdarray<double,3> integrals( lmsize_rho , uspp.num_beta_radial_functions * (uspp.num_beta_radial_functions + 1) / 2, ctx_.num_spins() );

	for(int ispin = 0; ispin < ctx_.num_spins(); ispin++ )
	{
		for(int irb2 = 0; irb2 < uspp.num_beta_radial_functions; irb2++)
		{
			for(int irb1 = 0; irb1 <= irb2; irb1++)
			{
				// common index
				int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

				Radial_grid newgrid = atom_type.radial_grid().segment(paw.cutoff_radius_index);

				// create array for integration
				std::vector<double> intdata(newgrid.num_points(),0);

				for(int lm3 = 0; lm3 < lmsize_rho; lm3++ )
				{
					// fill array
					for(int irad=0; irad< intdata.size(); irad++)
					{
						double ae_part = paw.all_elec_wfc(irad,irb1) * paw.all_elec_wfc(irad,irb2);
						double ps_part = paw.pseudo_wfc(irad,irb1) * paw.pseudo_wfc(irad,irb2)  + uspp.q_radial_functions_l(irad,iqij,l_by_lm[lm3]);

						intdata[irad] = ae_atom_pot(lm3,irad,ispin) * ae_part - ps_atom_pot(lm3,irad,ispin) * ps_part;
					}

					// create spline from data arrays
					Spline<double> dij_spl(newgrid,intdata);

					// integrate
					integrals(lm3, iqij, ispin) = dij_spl.integrate(0);
				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////
//	std::cout<<"INT"<<std::endl;
//	for(int irb2 = 0; irb2 < uspp.num_beta_radial_functions; irb2++)
//		for(int irb1 = 0; irb1 <= irb2; irb1++)
//			for(int lm3 = 0; lm3 < lmsize_rho; lm3++ )
//			{
//				int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;
//				std::cout<<integrals(lm3, iqij, 0) - integrals(lm3, iqij, 1)<<"   ";
//			}
//
//	std::cout<<std::endl;
	///////////////////////////////////////////////////////////////////////

	//---- calc Dij ----
	for(int ib2 = 0; ib2 < (int)atom_type.mt_lo_basis_size(); ib2++)
	{
		for(int ib1 = 0; ib1 <= ib2; ib1++)
		{
			//int idij = (ib2 * (ib2 + 1)) / 2 + ib1;

			// get lm quantum numbers (lm index) of the basis functions
			int lm1 = atom_type.indexb(ib1).lm;
			int lm2 = atom_type.indexb(ib2).lm;

			//get radial basis functions indices
			int irb1 = atom_type.indexb(ib1).idxrf;
			int irb2 = atom_type.indexb(ib2).idxrf;

			// common index
			int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

			// get num of non-zero GC
			int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

			for(int ispin = 0; ispin < ctx_.num_spins(); ispin++)
			{
				// add nonzero coefficients
				for(int inz = 0; inz < num_non_zero_gk; inz++)
				{
					auto& lm3coef = GC.gaunt(lm1,lm2,inz);

					// add to atom Dij an integral of dij array
					//atom.d_mtrx(ib1,ib2,ispin) += lm3coef.coef * integrals(lm3coef.lm3, iqij, ispin);
					//atom.d_mtrx(ib2,ib1,ispin) += lm3coef.coef * integrals(lm3coef.lm3, iqij, ispin);
				}
			}
		}
	}
//
//
//	std::cout <<" DIJ"<<std::endl;
//	for(int ib2 = 0; ib2 < (int)atom_type.mt_lo_basis_size(); ib2++)
//	{
//		for(int ib1 = 0; ib1 <= ib2; ib1++)
//		{
//			std::cout << atom.d_mtrx(ib2,ib1,0) - atom.d_mtrx(ib2,ib1,1) << " ";
//		}
//	}
//	std::cout <<std::endl;
//	std::cout<<std::endl;
	//////////////////////////////////////////////////////////////////
	//	std::ofstream ofxc("atom_dij.dat");
	//
	//	for(int ib2 = 0; ib2 < (int)atom_type.mt_lo_basis_size(); ib2++)
	//	{
	//		for(int ib1 = 0; ib1 < (int)atom_type.mt_lo_basis_size(); ib1++)
	//		{
	//			ofxc<< atom.d_mtrx(ib1,ib2,0).real() << std::endl;
	//		}
	//	}
	//
	//	ofxc.close();
	//////////////////////////////////////////////////////////////////

	//	TERMINATE("ololo");
}

}

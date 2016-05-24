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

//		paw_local_Dij_matrix_.push_back(std::move(atom_Dij));
	}
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
}




//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::xc_mt_PAW_nonmagnetic(Radial_grid const& rgrid,
									  Spheric_function<spectral,double> &out_atom_pot_sf,
									  Spheric_function<spectral,double> &full_rho_lm,
									  const std::vector<double> &rho_core)
{
	Spheric_function<spatial,double> full_rho_tp_sf = sht_->transform(full_rho_lm);

	Spheric_function<spatial,double> vxc_tp_sf(sht_->num_points(), rgrid);

	Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

	for(int itp = 0; itp < sht_->num_points(); itp++)
	{
		for(int ir=0; ir < rgrid.num_points(); ir++)
		{
			full_rho_tp_sf(itp,ir) += rho_core[ir];
		}
	}

	xc_mt_nonmagnetic(rgrid, xc_func_, full_rho_lm, full_rho_tp_sf, vxc_tp_sf, exc_tp_sf);

	out_atom_pot_sf += sht_->transform(vxc_tp_sf);

	sht_->transform(exc_tp_sf);
}



//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::xc_mt_PAW_collinear(Radial_grid const& rgrid,
									 Spheric_function<spectral,double>  &out_atom_pot_up_sf,
									 Spheric_function<spectral,double>  &out_atom_pot_dn_sf,
									 Spheric_function<spectral,double> &full_rho_lm,
									 mdarray<double,3> &magnetization_lm,
									 const std::vector<double> &rho_core)
{
	int lmsize_rho = full_rho_lm.angular_domain_size();

	// make magnetization from z component in lm components
	Spheric_function<spectral,double> magnetization_Z_lm(&magnetization_lm(0,0,2), lmsize_rho, rgrid );

	// calculate spin up spin down density components in lm components
	// up = 1/2 ( rho + magn );  down = 1/2 ( rho - magn )
	Spheric_function<spectral,double> rho_u_lm_sf =  0.5 * (full_rho_lm + magnetization_Z_lm);
	Spheric_function<spectral,double> rho_d_lm_sf =  0.5 * (full_rho_lm - magnetization_Z_lm);

	// transform density to theta phi components
	Spheric_function<spatial,double> rho_u_tp_sf = sht_->transform( rho_u_lm_sf );
	Spheric_function<spatial,double> rho_d_tp_sf = sht_->transform( rho_d_lm_sf );

	for(int itp = 0; itp < sht_->num_points(); itp++)
	{
		for(int ir=0; ir < rgrid.num_points(); ir++)
		{
			rho_u_tp_sf(itp,ir) += 0.5 * rho_core[ir];
			rho_d_tp_sf(itp,ir) += 0.5 * rho_core[ir];
		}
	}

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
	out_atom_pot_up_sf += sht_->transform(vxc_u_tp_sf);
	out_atom_pot_dn_sf += sht_->transform(vxc_d_tp_sf);

	// TODO add parameter to store this
	//sht_->transform(exc_tp_sf,exc_lm);
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

	//std::cout<< ae_full_density(0,0);

	int lmax = atom_type.indexr().lmax_lo();
	int lmsize_rho = ( lmax * 2 + 1) * ( lmax * 2 + 1);

	// make spheric functions from data arrays
	Spheric_function<function_domain_t::spectral,double> ae_dens_sf(&ae_full_density(0,0), lmsize_rho, atom.radial_grid());
	Spheric_function<function_domain_t::spectral,double> ps_dens_sf(&ps_full_density(0,0), lmsize_rho, atom.radial_grid());


	//-----------------------------------------
	//---- Calculation of Hartree potential ---
	//-----------------------------------------
	auto &ae_atom_pot = ae_paw_local_potential_[atom_index];
	auto &ps_atom_pot = ps_paw_local_potential_[atom_index];

	ae_atom_pot.zero();
	ps_atom_pot.zero();

	// make spher funcs from arrays
	std::vector< Spheric_function<spectral,double> > ae_atom_pot_sfs;
	std::vector< Spheric_function<spectral,double> > ps_atom_pot_sfs;

	for(int i=0;i<ctx_.num_spins();i++)
	{
		Spheric_function<spectral,double> ae_atom_pot_sf(&ae_atom_pot(0,0,i), lmsize_rho, atom.radial_grid());
		Spheric_function<spectral,double> ps_atom_pot_sf(&ps_atom_pot(0,0,i), lmsize_rho, atom.radial_grid());

		ae_atom_pot_sfs.push_back(std::move(ae_atom_pot_sf));
		ps_atom_pot_sfs.push_back(std::move(ps_atom_pot_sf));
	}

	// create qmt to store multipoles
	mdarray<double_complex,1> qmt(lmsize_rho);

	// solve poisson eq and fill 0th spin component of hartree array (in nonmagnetic we have only this)
	qmt.zero();
	poisson_atom_vmt(ae_dens_sf, ae_atom_pot_sfs[0], qmt, atom);

	qmt.zero();
	poisson_atom_vmt(ps_dens_sf, ps_atom_pot_sfs[0], qmt, atom);

	// if we have collinear megnetic states we need to add the same Hartree potential to DOWN-DOWN channel
	if(ctx_.num_spins() == 2)
	{
		ae_atom_pot_sfs[1] += ae_atom_pot_sfs[0];
		ps_atom_pot_sfs[1] += ps_atom_pot_sfs[0];
	}

	//// DEBUG //////////////////////////////////////////////
//		std::cout<<"hartree done"<<std::endl;
//
//		std::stringstream s,sd;
//		s<<"hartree_2_"<<atom_index<<".dat";
//		sd<<"density_2_"<<atom_index<<".dat";
//
//		std::ofstream of(s.str());
//		std::ofstream ofd(sd.str());
//		std::ofstream ofg("grid.dat");
//
//
//		for(int j = 0; j< ae_full_density.size(0); j++)
//		{
//			for(int i = 0; i< ae_full_density.size(1); i++)
//			{
//				of<< ae_atom_pot_sfs[0](j,i) << " " << ps_atom_pot_sfs[0](j,i) << std::endl;
//				ofd<< ae_full_density(j,i) << " " << ps_full_density(j,i) << std::endl;
//			}
//		}
//
//		for(int i = 0; i< ae_full_density.size(1); i++)
//		{
//			ofg << atom.radial_grid(i)<< " ";
//		}
//
//		std::cout<<"hartree out done"<<std::endl;
//
//		of.close();
//		ofd.close();
//		ofg.close();

	/////////////////////////////////////////////////////////////////


	//-----------------------------------------
	//---- Calculation of XC potential ---
	//-----------------------------------------
	switch(ctx_.num_spins())
	{
		case 1:
		{
			xc_mt_PAW_nonmagnetic(atom.radial_grid(), ae_atom_pot_sfs[0], ae_dens_sf ,paw.all_elec_core_charge);
			xc_mt_PAW_nonmagnetic(atom.radial_grid(), ps_atom_pot_sfs[0], ps_dens_sf ,uspp.core_charge_density);

			//////// DEBUG ///////////////////////////////////
//				std::ofstream ofxc("tot.dat");
//
//				for(int j = 0; j< ae_atom_pot_sfs[0].angular_domain_size(); j++)
//				{
//					for(int i = 0; i< ae_atom_pot_sfs[0].radial_grid().num_points(); i++)
//					{
//						ofxc<< ae_atom_pot_sfs[0](j,i) << " " << ps_atom_pot_sfs[0](j,i) << std::endl;
//
//					}
//				}
//
//				ofxc.close();
			////////////////////////////////////////////////////
		}break;

		case 2:
		{


			xc_mt_PAW_collinear(atom.radial_grid(), ae_atom_pot_sfs[0],ae_atom_pot_sfs[1], ae_dens_sf,
								ae_local_magnetization, paw.all_elec_core_charge);

			xc_mt_PAW_collinear(atom.radial_grid(), ps_atom_pot_sfs[0], ps_atom_pot_sfs[1], ps_dens_sf,
								ps_local_magnetization, uspp.core_charge_density);

			//////// DEBUG ///////////////////////////////////
//				std::ofstream ofxc_up("tot_up.dat");
//				std::ofstream ofxc_dn("tot_dn.dat");
//
//				for(int j = 0; j< ae_atom_pot_sfs[0].angular_domain_size(); j++)
//				{
//					for(int i = 0; i< ae_atom_pot_sfs[0].radial_grid().num_points(); i++)
//					{
//						ofxc_up<< ae_atom_pot_sfs[0](j,i) << " " << ae_atom_pot_sfs[0](j,i) << std::endl;
//						ofxc_dn<< ps_atom_pot_sfs[0](j,i) << " " << ps_atom_pot_sfs[0](j,i) << std::endl;
//
//					}
//				}
//
//				ofxc_up.close();
//				ofxc_dn.close();
			////////////////////////////////////////////////////
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
}




//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::calc_PAW_local_Dij(int atom_index)
{
	auto& atom = unit_cell_.atom(atom_index);

	auto& atom_type = atom.type();

	auto& paw = atom_type.get_PAW_descriptor();

	auto& uspp = atom_type.uspp();

	std::vector<int> l_by_lm = Utils::l_by_lm( 2 * atom_type.indexr().lmax_lo() );

	//TODO calculate not for every atom but for every atom type
	Gaunt_coefficients<double> GC(atom_type.indexr().lmax_lo(),
			2*atom_type.indexr().lmax_lo(),
			atom_type.indexr().lmax_lo(),
			SHT::gaunt_rlm);

	auto &ae_atom_pot = ae_paw_local_potential_[atom_index];
	auto &ps_atom_pot = ps_paw_local_potential_[atom_index];


	//--- λ ------------------------
	auto integrate = [&] (int ispin, int irb1, int irb2, int iqij, int lm3 )
			{
				//create array for integration
				std::vector<double> intdata(atom_type.radial_grid().num_points(),0);

				// fill array
				for(int irad=0; irad< paw.cutoff_radius_index-1; irad++)
				{
					double ae_part = paw.all_elec_wfc(irad,irb1) * paw.all_elec_wfc(irad,irb2);
					double ps_part = paw.pseudo_wfc(irad,irb1) * paw.pseudo_wfc(irad,irb2)  + uspp.q_radial_functions_l(irad,iqij,l_by_lm[lm3]);

					intdata[irad] = ae_atom_pot(lm3,irad,ispin) * ae_part - ps_atom_pot(lm3,irad,ispin) * ps_part;

					//////////////////////////////////////////////////////////////////////////////
//					if(std::abs(intdata[irad]) > 300)
//					{
//						std::cout<<irb1<<" "<< irb2<< " "<< iqij <<" "<< lm3 <<" | "<< irad<<" | "<<intdata[irad] <<" || "<<ae_part << " " << ae_atom_pot(lm3,irad,ispin)<< " "<< ps_part << " "<<ps_atom_pot(lm3,irad,ispin) << std::endl;
//					}
					//////////////////////////////////////////////////////////////////////////////
				}


				// create spline from data arrays
				Spline<double> dij_spl(atom_type.radial_grid(),intdata);

				//////////////////////////////////////////////////////////////////
//				std::stringstream s;
//				s<<"dij_arr_"<<irb1<<"_"<<irb2<<".dat";
//
//				std::ofstream of(s.str());
//
//
//				for(int j = 0; j< intdata.size(); j++)
//				{
//
//					of<< intdata[j] <<" ";
//
//				}
//
//				of.close();
				//////////////////////////////////////////////////////////////////
				// integrate
				return dij_spl.integrate(0);
			};
	//---------------------------------------

//	paw_local_Dij_matrix_[atom_index].zero();

		// iterate over spin components
//		for(int ispin = 0; ispin < (int)ae_atom_density.size(2); ispin++)
//		{
	// iterate over local basis functions (or over lm1 and lm2)

	//////////////////////////////////////////////////////////////////////////////
//	std::ofstream of("deeqs.dat");
//
//	for(int ib2 = 0; ib2 < (int)atom_type.mt_lo_basis_size(); ib2++)
//	{
//		for(int ib1 = 0; ib1 < (int)atom_type.mt_lo_basis_size(); ib1++)
//		{
//			of<<atom.d_mtrx(ib1,ib2,0).real()<<std::endl;
//		}
//	}
//
//	of.close();


	//////////////////////////////////////////////////////////////////////////////

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

			// index to iterate Qij,
			// TODO check indices
			int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

			// get num of non-zero GC
			int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

			for(int ispin = 0; ispin < ctx_.num_spins(); ispin++)
			{
				//atom.d_mtrx(ib1,ib2,ispin)=0.0;

				// add nonzero coefficients
				for(int inz = 0; inz < num_non_zero_gk; inz++)
				{
					auto& lm3coef = GC.gaunt(lm1,lm2,inz);

					/////////////////////////////////////////////////////////////
//					std::cout<<integrate(ispin,irb1,irb2,iqij,lm3coef.lm3)<<std::endl;
					//////////////////////////////////////////////////////////////

					// add to atom Dij an integral of dij array
					atom.d_mtrx(ib1,ib2,ispin) += lm3coef.coef * integrate(ispin,irb1,irb2,iqij,lm3coef.lm3);

//					paw_local_Dij_matrix_[atom_index]( idij, ispin) += lm3coef.coef * integrate(ispin,irb1,irb2,iqij,lm3coef.lm3);
				}

				atom.d_mtrx(ib2,ib1,ispin) = atom.d_mtrx(ib1,ib2,ispin);
			}
		}
	}


	//////////////////////////////////////////////////////////////////
	std::ofstream ofxc("atom_dij.dat");

	for(int ib2 = 0; ib2 < (int)atom_type.mt_lo_basis_size(); ib2++)
	{
		for(int ib1 = 0; ib1 < (int)atom_type.mt_lo_basis_size(); ib1++)
		{
			ofxc<< atom.d_mtrx(ib1,ib2,0).real() << std::endl;
		}
	}

	ofxc.close();
	//////////////////////////////////////////////////////////////////

//	TERMINATE("ololo");
}

}

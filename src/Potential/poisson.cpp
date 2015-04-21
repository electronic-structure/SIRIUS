#include "potential.h"

namespace sirius {

void Potential::poisson(Periodic_function<double>* rho, Periodic_function<double>* vh)
{
    Timer t("sirius::Potential::poisson");

    /* get plane-wave coefficients of the charge density */
    //fft_->input(&rho->f_it<global>(0));
    //fft_->transform(-1);
    //fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho->f_pw(0));

    std::vector< Spheric_function<spectral, double_complex> > rho_ylm(parameters_.unit_cell()->spl_num_atoms().local_size());
    std::vector< Spheric_function<spectral, double_complex> > vh_ylm(parameters_.unit_cell()->spl_num_atoms().local_size());

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (parameters_.unit_cell()->full_potential())
    {
        for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);

            rho_ylm[ialoc] = sht_->convert(rho->f_mt(ialoc));
            vh_ylm[ialoc] = Spheric_function<spectral, double_complex>(parameters_.lmmax_rho(), parameters_.unit_cell()->atom(ia)->type()->radial_grid());
        }
        
        /* true multipole moments */
        mdarray<double_complex, 2> qmt(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
        poisson_vmt(rho_ylm, vh_ylm, qmt);
        
        /* compute multipoles of interstitial density in MT region */
        mdarray<double_complex, 2> qit(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
        poisson_sum_G(parameters_.lmmax_rho(), &rho->f_pw(0), sbessel_mom_, qit);
        
        /* add contribution from the pseudo-charge */
        poisson_add_pseudo_pw(qmt, qit, &rho->f_pw(0));

        if (check_pseudo_charge)
        {
            poisson_sum_G(parameters_.lmmax_rho(), &rho->f_pw(0), sbessel_mom_, qit);

            double d = 0.0;
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                for (int lm = 0; lm < parameters_.lmmax_rho(); lm++) d += abs(qmt(lm, ia) - qit(lm, ia));
            }
        }
    }

    /* compute pw coefficients of Hartree potential */
    vh->f_pw(0) = 0.0;
    #pragma omp parallel for schedule(static)
    for (int ig = 1; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        vh->f_pw(ig) = (fourpi * rho->f_pw(ig) / pow(parameters_.reciprocal_lattice()->gvec_len(ig), 2));
    
    /* boundary condition for muffin-tins */
    if (parameters_.unit_cell()->full_potential())
    {
        /* compute V_lm at the MT boundary */
        mdarray<double_complex, 2> vmtlm(parameters_.lmmax_pot(), parameters_.unit_cell()->num_atoms());
        poisson_sum_G(parameters_.lmmax_pot(), &vh->f_pw(0), sbessel_mt_, vmtlm);
        
        /* add boundary condition and convert to Rlm */
        Timer t1("sirius::Potential::poisson|bc");
        mdarray<double, 2> rRl(parameters_.unit_cell()->max_num_mt_points(), parameters_.lmax_pot() + 1);
        int type_id_prev = -1;

        for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();

            if (parameters_.unit_cell()->atom(ia)->type_id() != type_id_prev)
            {
                type_id_prev = parameters_.unit_cell()->atom(ia)->type_id();
            
                double R = parameters_.unit_cell()->atom(ia)->mt_radius();

                #pragma omp parallel for default(shared)
                for (int l = 0; l <= parameters_.lmax_pot(); l++)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        rRl(ir, l) = pow(parameters_.unit_cell()->atom(ia)->type()->radial_grid(ir) / R, l);
                }
            }
            
            #pragma omp parallel for default(shared)
            for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
            {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++)
                    vh_ylm[ialoc](lm, ir) += vmtlm(lm, ia) * rRl(ir, l);
            }
            sht_->convert(vh_ylm[ialoc], vh->f_mt(ialoc));
            
            /* save electronic part of potential at point of origin */
            vh_el_(ia) = vh->f_mt<local>(0, 0, ialoc);
        }
        parameters_.comm().allgather(vh_el_.at<CPU>(), (int)parameters_.unit_cell()->spl_num_atoms().global_offset(),
                                     (int)parameters_.unit_cell()->spl_num_atoms().local_size());

    }
    
    /* transform Hartree potential to real space */
    fft_->input(fft_->num_gvec(), fft_->index_map(), &vh->f_pw(0));
    fft_->transform(1);
    fft_->output(&vh->f_it<global>(0));
    
    /* compute contribution from the smooth part of Hartree potential */
    energy_vha_ = inner(parameters_, rho, vh);
        
    /* add nucleus potential and contribution to Hartree energy */
    if (parameters_.unit_cell()->full_potential())
    {
        double evha_nuc_ = 0;
        for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            auto atom = parameters_.unit_cell()->atom(ia);
            Spline<double> srho(atom->radial_grid());
            for (int ir = 0; ir < atom->num_mt_points(); ir++)
            {
                double r = atom->radial_grid(ir);
                hartree_potential_->f_mt<local>(0, ir, ialoc) -= atom->zn() / r / y00;
                srho[ir] = rho->f_mt<local>(0, ir, ialoc);
            }
            evha_nuc_ -= atom->zn() * srho.interpolate().integrate(1) / y00;
        }
        parameters_.comm().allreduce(&evha_nuc_, 1);
        energy_vha_ += evha_nuc_;
    }
}

};

#include "potential.h"

namespace sirius {

void Potential::poisson_vmt(Periodic_function<double>* rho__, 
                            Periodic_function<double>* vh__,
                            mdarray<double_complex, 2>& qmt__)
{
    Timer t("sirius::Potential::poisson_vmt");

    qmt__.zero();
    
    for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);

        double R = parameters_.unit_cell()->atom(ia)->mt_radius();
        int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();

        std::vector<double> qmt(parameters_.lmmax_rho(), 0);
       
        #pragma omp parallel default(shared)
        {
            std::vector<double> g1;
            std::vector<double> g2;

            #pragma omp for
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
            {
                int l = l_by_lm_[lm];

                auto rholm = rho__->f_mt(ialoc).component(lm);

                /* save multipole moment */
                qmt[lm] = rholm.integrate(g1, l + 2);
                
                if (lm < parameters_.lmmax_pot())
                {
                    rholm.integrate(g2, 1 - l);
                    
                    double d1 = 1.0 / pow(R, 2 * l + 1); 
                    double d2 = 1.0 / double(2 * l + 1); 
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double r = parameters_.unit_cell()->atom(ia)->radial_grid(ir);

                        double vlm = (1.0 - std::pow(r / R, 2 * l + 1)) * g1[ir] / std::pow(r, l + 1) +
                                      (g2[nmtp - 1] - g2[ir]) * std::pow(r, l) - (g1[nmtp - 1] - g1[ir]) * std::pow(r, l) * d1;

                        vh__->f_mt(ialoc)(lm, ir) = fourpi * vlm * d2;
                    }
                }
            }
        }

        SHT::convert(parameters_.lmax_rho(), &qmt[0], &qmt__(0, ia));

        /* fixed part of nuclear potential */
        for (int ir = 0; ir < nmtp; ir++)
        {
            //double r = parameters_.unit_cell()->atom(ia)->radial_grid(ir);
            //vh_ylm[ialoc](0, ir) -= parameters_.unit_cell()->atom(ia)->zn() * (1 / r - 1 / R) / y00;
            vh__->f_mt(ialoc)(0, ir) += parameters_.unit_cell()->atom(ia)->zn() / R / y00;
        }

        /* nuclear multipole moment */
        qmt__(0, ia) -= parameters_.unit_cell()->atom(ia)->zn() * y00;
    }

    parameters_.comm().allreduce(&qmt__(0, 0), (int)qmt__.size());
}


void Potential::poisson(Periodic_function<double>* rho, Periodic_function<double>* vh)
{
    Timer t("sirius::Potential::poisson");

    /* get plane-wave coefficients of the charge density */
    //fft_->input(&rho->f_it<global>(0));
    //fft_->transform(-1);
    //fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho->f_pw(0));

    //std::vector< Spheric_function<spectral, double_complex> > rho_ylm(parameters_.unit_cell()->spl_num_atoms().local_size());
    //std::vector< Spheric_function<spectral, double_complex> > vh_ylm(parameters_.unit_cell()->spl_num_atoms().local_size());

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (parameters_.unit_cell()->full_potential())
    {
        /* true multipole moments */
        mdarray<double_complex, 2> qmt(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
        poisson_vmt(rho, vh, qmt);
        
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
                        rRl(ir, l) = std::pow(parameters_.unit_cell()->atom(ia)->type()->radial_grid(ir) / R, l);
                }
            }

            std::vector<double> vlm(parameters_.lmmax_pot());
            SHT::convert(parameters_.lmax_pot(), &vmtlm(0, ia), &vlm[0]);
            
            #pragma omp parallel for default(shared)
            for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
            {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) vh->f_mt(ialoc)(lm, ir) += vlm[lm] * rRl(ir, l);
            }
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

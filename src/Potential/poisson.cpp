#include "potential.h"

namespace sirius {

/** The following operation is performed:
 *  \f[
 *    q_{\ell m}^{\alpha} = \sum_{\bf G} 4\pi \rho({\bf G}) e^{i{\bf G}{\bf r}_{\alpha}}i^{\ell}f_{\ell}^{\alpha}(G) Y_{\ell m}^{*}(\hat{\bf G})
 *  \f]
 */
void Potential::poisson_sum_G(int lmmax__,
                              double_complex* fpw__,
                              mdarray<double, 3>& fl__,
                              matrix<double_complex>& flm__)
{
    Timer t("sirius::Potential::poisson_sum_G");
    
    auto rl = parameters_.reciprocal_lattice();
    auto uc = parameters_.unit_cell();
    int ngv_loc = (int)rl->spl_num_gvec().local_size();
    //int lmax = Utils::lmax_by_lmmax(lmmax__);

    int na_max = 0;
    for (int iat = 0; iat < uc->num_atom_types(); iat++) na_max = std::max(na_max, uc->atom_type(iat)->num_atoms());
    
    matrix<double_complex> phase_factors(ngv_loc, na_max);
    matrix<double_complex> zm(lmmax__, ngv_loc);
    matrix<double_complex> tmp(lmmax__, na_max);

    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
        int na = uc->atom_type(iat)->num_atoms();
        #pragma omp parallel for
        for (int igloc = 0; igloc < ngv_loc; igloc++)
        {
            int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
            for (int i = 0; i < na; i++)
            {
                int ia = uc->atom_type(iat)->atom_id(i);
                phase_factors(igloc, i) = rl->gvec_phase_factor<local>(igloc, ia);
            }
            for (int lm = 0; lm < lmmax__; lm++)
            {
                int l = l_by_lm_[lm];
                zm(lm, igloc) = fourpi * fpw__[rl->spl_num_gvec(igloc)] * zilm_[lm] *
                                fl__(l, iat, rl->gvec_shell(ig)) * std::conj(rl->gvec_ylm(lm, igloc));
            }
        }
        linalg<CPU>::gemm(0, 0, lmmax__, na, ngv_loc, zm.at<CPU>(), zm.ld(), phase_factors.at<CPU>(), phase_factors.ld(),
                          tmp.at<CPU>(), tmp.ld());
        for (int i = 0; i < na; i++)
        {
            int ia = uc->atom_type(iat)->atom_id(i);
            for (int lm = 0; lm < lmmax__; lm++) flm__(lm, ia) = tmp(lm, i);
        }
    }


    //matrix<double_complex> zm1(ngv_loc, lmmax__);
    //#pragma omp parallel for default(shared)
    //for (int lm = 0; lm < lmmax__; lm++)
    //{
    //    for (int igloc = 0; igloc < ngv_loc; igloc++)
    //    {
    //        zm1(igloc, lm) = parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc) * 
    //                         conj(fpw__[parameters_.reciprocal_lattice()->spl_num_gvec(igloc)] * zilm_[lm]);
    //    }
    //}

    //matrix<double_complex> zm2(ngv_loc, parameters_.unit_cell()->num_atoms());

    //for (int l = 0; l <= lmax; l++)
    //{
    //    #pragma omp parallel for default(shared)
    //    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    //    {
    //        int iat = parameters_.unit_cell()->atom(ia)->type_id();
    //        for (int igloc = 0; igloc < ngv_loc; igloc++)
    //        {
    //            int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
    //            zm2(igloc, ia) = fourpi * parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia) *  
    //                             fl__(l, iat, parameters_.reciprocal_lattice()->gvec_shell(ig));
    //        }
    //    }

    //    linalg<CPU>::gemm(2, 0, 2 * l + 1, parameters_.unit_cell()->num_atoms(), ngv_loc, 
    //                      &zm1(0, Utils::lm_by_l_m(l, -l)), zm1.ld(), &zm2(0, 0), zm2.ld(), 
    //                      &flm__(Utils::lm_by_l_m(l, -l), 0), flm__.ld());
    //}

    //== #pragma omp parallel
    //== {
    //==     mdarray<double_complex, 2> zm(lmmax__, ngv_loc);
    //==     #pragma omp for
    //==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    //==     {
    //==         int iat = parameters_.unit_cell()->atom(ia)->type_id();
    //==         for (int igloc = 0; igloc < ngv_loc; igloc++)
    //==         {
    //==             int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
    //==             for (int lm = 0; lm < lmmax__; lm++)
    //==             {
    //==                 int l = l_by_lm_[lm];
    //==                 zm(lm, igloc) = fourpi * parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia) * 
    //==                                 zilm_[lm] * conj(parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc)) * 
    //==                                 fl__(l, iat, parameters_.reciprocal_lattice()->gvec_shell(ig));
    //==             }
    //==         }
    //==         blas<CPU>::gemv(0, lmmax__, ngv_loc, complex_one, zm.at<CPU>(), zm.ld(), 
    //==                         &fpw__[parameters_.reciprocal_lattice()->spl_num_gvec().global_offset()], 1, complex_zero, 
    //==                         &flm__(0, ia), 1);
    //==     }
    //== }
    
    parameters_.comm().allreduce(&flm__(0, 0), (int)flm__.size());
}

void Potential::poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt, mdarray<double_complex, 2>& qit, double_complex* rho_pw)
{
    Timer t("sirius::Potential::poisson_add_pseudo_pw");
    std::vector<double_complex> pseudo_pw(parameters_.reciprocal_lattice()->num_gvec());
    memset(&pseudo_pw[0], 0, parameters_.reciprocal_lattice()->num_gvec() * sizeof(double_complex));
    
    /* The following term is added to the plane-wave coefficients of the charge density:
     * Integrate[SphericalBesselJ[l,a*x]*p[x,R]*x^2,{x,0,R},Assumptions->{l>=0,n>=0,R>0,a>0}] / 
     *  Integrate[p[x,R]*x^(2+l),{x,0,R},Assumptions->{h>=0,n>=0,R>0}]
     * i.e. contributon from pseudodensity to l-th channel of plane wave expansion multiplied by 
     * the difference bethween true and interstitial-in-the-mt multipole moments and divided by the 
     * moment of the pseudodensity.
     */
    #pragma omp parallel default(shared)
    {
        std::vector<double_complex> pseudo_pw_pt(parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), double_complex(0, 0));

        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int iat = parameters_.unit_cell()->atom(ia)->type_id();

            double R = parameters_.unit_cell()->atom(ia)->mt_radius();

            /* compute G-vector independent prefactor */
            std::vector<double_complex> zp(parameters_.lmmax_rho());
            for (int l = 0, lm = 0; l <= parameters_.lmax_rho(); l++)
            {
                for (int m = -l; m <= l; m++, lm++)
                    zp[lm] = (qmt(lm, ia) - qit(lm, ia)) * conj(zil_[l]) * gamma_factors_R_(l, iat);
            }

            for (int igloc = 0; igloc < (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
            {
                int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
                
                double gR = parameters_.reciprocal_lattice()->gvec_len(ig) * R;
                
                double_complex zt = fourpi * conj(parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia)) / parameters_.unit_cell()->omega();

                if (ig)
                {
                    double_complex zt2(0, 0);
                    for (int l = 0, lm = 0; l <= parameters_.lmax_rho(); l++)
                    {
                        double_complex zt1(0, 0);
                        for (int m = -l; m <= l; m++, lm++) zt1 += parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc) * zp[lm];

                        zt2 += zt1 * sbessel_mt_(l + pseudo_density_order + 1, iat, parameters_.reciprocal_lattice()->gvec_shell(ig));
                    }

                    pseudo_pw_pt[igloc] += zt * zt2 * pow(2.0 / gR, pseudo_density_order + 1);
                }
                else // for |G|=0
                {
                    pseudo_pw_pt[igloc] += zt * y00 * (qmt(0, ia) - qit(0, ia));
                }
            }
        }
        #pragma omp critical
        for (int igloc = 0; igloc < (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++) 
            pseudo_pw[parameters_.reciprocal_lattice()->spl_num_gvec(igloc)] += pseudo_pw_pt[igloc];
    }

    parameters_.comm().allgather(&pseudo_pw[0], (int)parameters_.reciprocal_lattice()->spl_num_gvec().global_offset(), 
                                 (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size());
        
    // add pseudo_density to interstitial charge density; now rho(G) has the correct multipole moments in the muffin-tins
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++) rho_pw[ig] += pseudo_pw[ig];
}

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

        /* constant part of nuclear potential */
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

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (parameters_.unit_cell()->full_potential())
    {
        /* true multipole moments */
        mdarray<double_complex, 2> qmt(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
        poisson_vmt(rho, vh, qmt);

        #ifdef _PRINT_OBJECT_CHECKSUM_
        double_complex z1 = qmt.checksum();
        DUMP("checksum(qmt): %18.10f %18.10f", std::real(z1), std::imag(z1));
        #endif
        #ifdef _PRINT_OBJECT_HASH_
        DUMP("hash(qmt): %16llX", qmt.hash());
        #endif

        /* compute multipoles of interstitial density in MT region */
        mdarray<double_complex, 2> qit(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
        poisson_sum_G(parameters_.lmmax_rho(), &rho->f_pw(0), sbessel_mom_, qit);

        #ifdef _PRINT_OBJECT_CHECKSUM_
        double_complex z2 = qit.checksum();
        DUMP("checksum(qit): %18.10f %18.10f", std::real(z2), std::imag(z2));
        #endif
        #ifdef _PRINT_OBJECT_HASH_
        DUMP("hash(rhopw): %16llX", rho->f_pw().hash());
        DUMP("hash(qit): %16llX", qit.hash());
        #endif

        /* add contribution from the pseudo-charge */
        poisson_add_pseudo_pw(qmt, qit, &rho->f_pw(0));
        
        #ifdef _PRINT_OBJECT_CHECKSUM_
        double_complex z3 = mdarray<double_complex, 1>(&rho->f_pw(0), fft_->num_gvec()).checksum();
        DUMP("checksum(rho_ps_pw): %18.10f %18.10f", std::real(z3), std::imag(z3));
        #endif
        #ifdef _PRINT_OBJECT_HASH_
        DUMP("hash(rho_ps_pw): %16llX", mdarray<double_complex, 1>(&rho->f_pw(0), fft_->num_gvec()).hash());
        #endif

        if (check_pseudo_charge)
        {
            poisson_sum_G(parameters_.lmmax_rho(), &rho->f_pw(0), sbessel_mom_, qit);

            double d = 0.0;
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                for (int lm = 0; lm < parameters_.lmmax_rho(); lm++) d += abs(qmt(lm, ia) - qit(lm, ia));
            }
            printf("pseudocharge error: %18.10f\n", d);
        }
    }

    /* compute pw coefficients of Hartree potential */
    vh->f_pw(0) = 0.0;
    #pragma omp parallel for schedule(static)
    for (int ig = 1; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        vh->f_pw(ig) = (fourpi * rho->f_pw(ig) / std::pow(parameters_.reciprocal_lattice()->gvec_len(ig), 2));

    #ifdef _PRINT_OBJECT_CHECKSUM_
    double_complex z4 = mdarray<double_complex, 1>(&vh->f_pw(0), fft_->num_gvec()).checksum();
    DUMP("checksum(vh_pw): %20.14f %20.14f", std::real(z4), std::imag(z4));
    #endif
    #ifdef _PRINT_OBJECT_HASH_
    DUMP("hash(vh_pw): %16llX", mdarray<double_complex, 1>(&vh->f_pw(0), fft_->num_gvec()).hash());
    #endif
    
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
    vh->fft_transform(1);

    #ifdef _PRINT_OBJECT_CHECKSUM_
    DUMP("checksum(vha_it): %20.14f", vh->f_it().checksum());
    #endif
    #ifdef _PRINT_OBJECT_HASH_
    DUMP("hash(vha_it): %16llX", vh->f_it().hash());
    #endif
    
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

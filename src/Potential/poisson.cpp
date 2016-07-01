#include "potential.h"

namespace sirius {

#ifdef __GPU
extern "C" void generate_phase_factors_gpu(int num_gvec_loc__,
                                           int num_atoms__,
                                           int const* gvec__,
                                           double const* atom_pos__,
                                           cuDoubleComplex* phase_factors__);
#endif

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
    PROFILE_WITH_TIMER("sirius::Potential::poisson_sum_G");
    
    int ngv_loc = spl_num_gvec_.local_size();

    int na_max = 0;
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) na_max = std::max(na_max, unit_cell_.atom_type(iat).num_atoms());
    
    matrix<double_complex> phase_factors(ngv_loc, na_max);
    matrix<double_complex> zm(lmmax__, ngv_loc);
    matrix<double_complex> tmp(lmmax__, na_max);

    if (ctx_.processing_unit() == CPU)
    {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            int na = unit_cell_.atom_type(iat).num_atoms();
            #pragma omp parallel for
            for (int igloc = 0; igloc < ngv_loc; igloc++)
            {
                int ig = spl_num_gvec_[igloc];
                for (int i = 0; i < na; i++)
                {
                    int ia = unit_cell_.atom_type(iat).atom_id(i);
                    phase_factors(igloc, i) = ctx_.gvec_phase_factor(ig, ia);
                }
                for (int lm = 0; lm < lmmax__; lm++)
                {
                    int l = l_by_lm_[lm];
                    zm(lm, igloc) = fourpi * fpw__[ig] * zilm_[lm] *
                                    fl__(l, iat, ctx_.gvec().shell(ig)) * std::conj(gvec_ylm_(lm, igloc));
                }
            }
            linalg<CPU>::gemm(0, 0, lmmax__, na, ngv_loc, zm.at<CPU>(), zm.ld(), phase_factors.at<CPU>(), phase_factors.ld(),
                              tmp.at<CPU>(), tmp.ld());
            for (int i = 0; i < na; i++)
            {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                for (int lm = 0; lm < lmmax__; lm++) flm__(lm, ia) = tmp(lm, i);
            }
        }
    }

    if (ctx_.processing_unit() == GPU)
    {
        #ifdef __GPU
        auto gvec = mdarray<int, 2>(3, ngv_loc);
        for (int igloc = 0; igloc < ngv_loc; igloc++)
        {
            for (int x = 0; x < 3; x++) gvec(x, igloc) = ctx_.gvec()[spl_num_gvec_[igloc]][x];
        }
        gvec.allocate_on_device();
        gvec.copy_to_device();

        phase_factors.allocate_on_device();
        zm.allocate_on_device();
        tmp.allocate_on_device();

        double_complex alpha(1, 0);
        double_complex beta(0, 0);

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            int na = unit_cell_.atom_type(iat).num_atoms();
            
            mdarray<double, 2> atom_pos(3, na);
            for (int i = 0; i < na; i++)
            {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                auto pos = unit_cell_.atom(ia).position();
                for (int x = 0; x < 3; x++) atom_pos(x, i) = pos[x];
            }
            atom_pos.allocate_on_device();
            atom_pos.copy_to_device();

            generate_phase_factors_gpu(ngv_loc, na, gvec.at<GPU>(), atom_pos.at<GPU>(), phase_factors.at<GPU>());

            #pragma omp parallel for
            for (int igloc = 0; igloc < ngv_loc; igloc++)
            {
                int ig = spl_num_gvec_[igloc];
                for (int lm = 0; lm < lmmax__; lm++)
                {
                    int l = l_by_lm_[lm];
                    zm(lm, igloc) = fourpi * fpw__[ig] * zilm_[lm] *
                                    fl__(l, iat, ctx_.gvec().shell(ig)) * std::conj(gvec_ylm_(lm, igloc));
                }
            }
            zm.copy_to_device();
            linalg<GPU>::gemm(0, 0, lmmax__, na, ngv_loc, &alpha, zm.at<GPU>(), zm.ld(), phase_factors.at<GPU>(), phase_factors.ld(),
                              &beta, tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            for (int i = 0; i < na; i++)
            {
                int ia = unit_cell_.atom_type(iat).atom_id(i);
                for (int lm = 0; lm < lmmax__; lm++) flm__(lm, ia) = tmp(lm, i);
            }
        }
        #else
        TERMINATE_NO_GPU
        #endif
    }
    
    ctx_.comm().allreduce(&flm__(0, 0), (int)flm__.size());
}

void Potential::poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt, mdarray<double_complex, 2>& qit, double_complex* rho_pw)
{
    PROFILE_WITH_TIMER("sirius::Potential::poisson_add_pseudo_pw");
    
    /* The following term is added to the plane-wave coefficients of the charge density:
     * Integrate[SphericalBesselJ[l,a*x]*p[x,R]*x^2,{x,0,R},Assumptions->{l>=0,n>=0,R>0,a>0}] / 
     *  Integrate[p[x,R]*x^(2+l),{x,0,R},Assumptions->{h>=0,n>=0,R>0}]
     * i.e. contributon from pseudodensity to l-th channel of plane wave expansion multiplied by 
     * the difference bethween true and interstitial-in-the-mt multipole moments and divided by the 
     * moment of the pseudodensity.
     */
    #pragma omp parallel default(shared)
    {
        int tid = omp_get_thread_num();
        splindex<block> spl_gv_t(spl_num_gvec_.local_size(), omp_get_num_threads(), tid);
        std::vector<double_complex> pseudo_pw_t(spl_gv_t.local_size(), complex_zero); 

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            int iat = unit_cell_.atom(ia).type_id();

            double R = unit_cell_.atom(ia).mt_radius();

            /* compute G-vector independent prefactor */
            std::vector<double_complex> zp(ctx_.lmmax_rho());
            for (int l = 0, lm = 0; l <= ctx_.lmax_rho(); l++)
            {
                for (int m = -l; m <= l; m++, lm++)
                    zp[lm] = (qmt(lm, ia) - qit(lm, ia)) * std::conj(zil_[l]) * gamma_factors_R_(l, iat);
            }

            for (int igloc_t = 0; igloc_t < (int)spl_gv_t.local_size(); igloc_t++)
            {
                int igloc = (int)spl_gv_t[igloc_t];
                int ig = (int)spl_num_gvec_[igloc];

                double gR = ctx_.gvec().gvec_len(ig) * R;
                
                double_complex zt = fourpi * std::conj(ctx_.gvec_phase_factor(ig, ia)) / unit_cell_.omega();

                if (ig)
                {
                    double_complex zt2(0, 0);
                    for (int l = 0, lm = 0; l <= ctx_.lmax_rho(); l++)
                    {
                        double_complex zt1(0, 0);
                        for (int m = -l; m <= l; m++, lm++) zt1 += gvec_ylm_(lm, igloc) * zp[lm];

                        zt2 += zt1 * sbessel_mt_(l + pseudo_density_order + 1, iat, ctx_.gvec().shell(ig));
                    }

                    pseudo_pw_t[igloc_t] += zt * zt2 * std::pow(2.0 / gR, pseudo_density_order + 1);
                }
                else // for |G|=0
                {
                    pseudo_pw_t[igloc_t] += zt * y00 * (qmt(0, ia) - qit(0, ia));
                }
            }
        }

        /* add pseudo_density to interstitial charge density;
         * now rho(G) has the correct multipole moments in the muffin-tins */
        for (int igloc_t = 0; igloc_t < spl_gv_t.local_size(); igloc_t++)
        {
            int igloc = spl_gv_t[igloc_t];
            int ig = spl_num_gvec_[igloc];
            rho_pw[ig] += pseudo_pw_t[igloc_t];
        }
    }

    ctx_.comm().allgather(&rho_pw[0], (int)spl_num_gvec_.global_offset(), (int)spl_num_gvec_.local_size());
}




//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::poisson_atom_vmt(Spheric_function<function_domain_t::spectral,double> &rho_mt,
                            Spheric_function<function_domain_t::spectral,double> &vh_mt,
                            mdarray<double_complex, 1>& qmt_ext,
                            Atom &atom)
{
    double R = atom.mt_radius();
    int nmtp = atom.num_mt_points();

    // passed size of qmt_mt must be equal to
    int lmsize = rho_mt.angular_domain_size();

    int lmax_rho = 2*atom.type().indexr().lmax_lo();



    std::vector<double> qmt(lmsize, 0);

    std::vector<int> l_by_lm = Utils::l_by_lm(lmax_rho);

//  std::cout<<lmsize<<" "<<lmax_rho<<" "<< R <<std::endl;

    #pragma omp parallel default(shared)
    {
        std::vector<double> g1;
        std::vector<double> g2;

        #pragma omp for
        for (int lm = 0; lm < lmsize; lm++)
        {
            int l = l_by_lm[lm];

            auto rholm = rho_mt.component(lm);

            /* save multipole moment */
            qmt[lm] = rholm.integrate(g1, l + 2);

            if (lm < lmsize)
            {
                rholm.integrate(g2, 1 - l);

                double d1 = 1.0 / std::pow(R, 2 * l + 1);
                double d2 = 1.0 / double(2 * l + 1);
                for (int ir = 0; ir < nmtp; ir++)
                {


                    double r = atom.radial_grid(ir);

                    double vlm = (1.0 - std::pow(r / R, 2 * l + 1)) * g1[ir] / std::pow(r, l + 1) +
                                  (g2[nmtp - 1] - g2[ir]) * std::pow(r, l) - (g1[nmtp - 1] - g1[ir]) * std::pow(r, l) * d1;

                    vh_mt(lm, ir) = fourpi * vlm * d2;

                    ////////////////////////////////////////////////////////////////////
//                  if(lm==0 && ir> 930)
//                  {
//                      std::cout<<"g2 " << g2[ir]<<" vh "<<vh_mt(lm, ir)<<std::endl;
//                  }
                    //////////////////////////////////////////////////////////////////////
                }
            }
        }
    }

//  std::cout<<"pvmt done"<<std::endl;

    SHT::convert(lmax_rho, &qmt[0], &qmt_ext(0));

    /* constant part of nuclear potential -z*(1/r - 1/R) */
    for (int ir = 0; ir < nmtp; ir++)
        vh_mt(0, ir) += atom.zn() / R / y00;

    /* nuclear multipole moment */
    qmt_ext(0) -= atom.zn() * y00;
}



//TODO insert function above into this
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void Potential::poisson_vmt(Periodic_function<double>* rho__, 
                            Periodic_function<double>* vh__,
                            mdarray<double_complex, 2>& qmt__)
{
    PROFILE_WITH_TIMER("sirius::Potential::poisson_vmt");

    qmt__.zero();
    
    for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = unit_cell_.spl_num_atoms(ialoc);

        double R = unit_cell_.atom(ia).mt_radius();
        int nmtp = unit_cell_.atom(ia).num_mt_points();

        std::vector<double> qmt(ctx_.lmmax_rho(), 0);
       
        #pragma omp parallel default(shared)
        {
            std::vector<double> g1;
            std::vector<double> g2;

            #pragma omp for
            for (int lm = 0; lm < ctx_.lmmax_rho(); lm++)
            {
                int l = l_by_lm_[lm];

                auto rholm = rho__->f_mt(ialoc).component(lm);

                /* save multipole moment */
                qmt[lm] = rholm.integrate(g1, l + 2);
                
                if (lm < ctx_.lmmax_pot())
                {
                    rholm.integrate(g2, 1 - l);
                    
                    double d1 = 1.0 / std::pow(R, 2 * l + 1); 
                    double d2 = 1.0 / double(2 * l + 1); 
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double r = unit_cell_.atom(ia).radial_grid(ir);

                        double vlm = (1.0 - std::pow(r / R, 2 * l + 1)) * g1[ir] / std::pow(r, l + 1) +
                                      (g2[nmtp - 1] - g2[ir]) * std::pow(r, l) - (g1[nmtp - 1] - g1[ir]) * std::pow(r, l) * d1;

                        vh__->f_mt(ialoc)(lm, ir) = fourpi * vlm * d2;
                    }
                }
            }
        }

        SHT::convert(ctx_.lmax_rho(), &qmt[0], &qmt__(0, ia));

        /* constant part of nuclear potential -z*(1/r - 1/R) */
        for (int ir = 0; ir < nmtp; ir++)
            vh__->f_mt(ialoc)(0, ir) += unit_cell_.atom(ia).zn() / R / y00;

        /* nuclear multipole moment */
        qmt__(0, ia) -= unit_cell_.atom(ia).zn() * y00;
    }

    ctx_.comm().allreduce(&qmt__(0, 0), (int)qmt__.size());
}


void Potential::poisson(Periodic_function<double>* rho, Periodic_function<double>* vh)
{
    PROFILE_WITH_TIMER("sirius::Potential::poisson");

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (ctx_.full_potential())
    {
        /* true multipole moments */
        mdarray<double_complex, 2> qmt(ctx_.lmmax_rho(), unit_cell_.num_atoms());
        poisson_vmt(rho, vh, qmt);

        #ifdef __PRINT_OBJECT_CHECKSUM
        double_complex z1 = qmt.checksum();
        DUMP("checksum(qmt): %18.10f %18.10f", std::real(z1), std::imag(z1));
        #endif
        #ifdef __PRINT_OBJECT_HASH
        DUMP("hash(qmt): %16llX", qmt.hash());
        #endif

        /* compute multipoles of interstitial density in MT region */
        mdarray<double_complex, 2> qit(ctx_.lmmax_rho(), unit_cell_.num_atoms());
        poisson_sum_G(ctx_.lmmax_rho(), &rho->f_pw(0), sbessel_mom_, qit);

        #ifdef __PRINT_OBJECT_CHECKSUM
        double_complex z2 = qit.checksum();
        DUMP("checksum(qit): %18.10f %18.10f", std::real(z2), std::imag(z2));
        #endif
        #ifdef __PRINT_OBJECT_HASH
        DUMP("hash(rhopw): %16llX", rho->f_pw().hash());
        DUMP("hash(qit): %16llX", qit.hash());
        #endif

        /* add contribution from the pseudo-charge */
        poisson_add_pseudo_pw(qmt, qit, &rho->f_pw(0));
        
        #ifdef __PRINT_OBJECT_CHECKSUM
        double_complex z3 = mdarray<double_complex, 1>(&rho->f_pw(0), ctx_.gvec().num_gvec()).checksum();
        DUMP("checksum(rho_ps_pw): %18.10f %18.10f", std::real(z3), std::imag(z3));
        #endif
        #ifdef __PRINT_OBJECT_HASH
        DUMP("hash(rho_ps_pw): %16llX", mdarray<double_complex, 1>(&rho->f_pw(0), ctx_.gvec().num_gvec()).hash());
        #endif

        if (check_pseudo_charge)
        {
            poisson_sum_G(ctx_.lmmax_rho(), &rho->f_pw(0), sbessel_mom_, qit);

            double d = 0.0;
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
            {
                for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) d += abs(qmt(lm, ia) - qit(lm, ia));
            }
            printf("pseudocharge error: %18.10f\n", d);
        }
    }

    /* compute pw coefficients of Hartree potential */
    vh->f_pw(0) = 0.0;
    #pragma omp parallel for
    for (int ig = 1; ig < ctx_.gvec().num_gvec(); ig++)
        vh->f_pw(ig) = (fourpi * rho->f_pw(ig) / std::pow(ctx_.gvec().gvec_len(ig), 2));

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z4 = mdarray<double_complex, 1>(&vh->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    DUMP("checksum(vh_pw): %20.14f %20.14f", std::real(z4), std::imag(z4));
    #endif
    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(vh_pw): %16llX", mdarray<double_complex, 1>(&vh->f_pw(0), ctx_.gvec().num_gvec()).hash());
    #endif
    
    /* boundary condition for muffin-tins */
    if (ctx_.full_potential())
    {
        /* compute V_lm at the MT boundary */
        mdarray<double_complex, 2> vmtlm(ctx_.lmmax_pot(), unit_cell_.num_atoms());
        poisson_sum_G(ctx_.lmmax_pot(), &vh->f_pw(0), sbessel_mt_, vmtlm);
        
        /* add boundary condition and convert to Rlm */
        runtime::Timer t1("sirius::Potential::poisson|bc");
        mdarray<double, 2> rRl(unit_cell_.max_num_mt_points(), ctx_.lmax_pot() + 1);
        int type_id_prev = -1;

        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            int nmtp = unit_cell_.atom(ia).num_mt_points();

            if (unit_cell_.atom(ia).type_id() != type_id_prev)
            {
                type_id_prev = unit_cell_.atom(ia).type_id();
            
                double R = unit_cell_.atom(ia).mt_radius();

                #pragma omp parallel for default(shared)
                for (int l = 0; l <= ctx_.lmax_pot(); l++)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        rRl(ir, l) = std::pow(unit_cell_.atom(ia).type().radial_grid(ir) / R, l);
                }
            }

            std::vector<double> vlm(ctx_.lmmax_pot());
            SHT::convert(ctx_.lmax_pot(), &vmtlm(0, ia), &vlm[0]);
            
            #pragma omp parallel for default(shared)
            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++)
            {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) vh->f_mt(ialoc)(lm, ir) += vlm[lm] * rRl(ir, l);
            }
            /* save electronic part of potential at point of origin */
            vh_el_(ia) = vh->f_mt<local>(0, 0, ialoc);
        }
        ctx_.comm().allgather(vh_el_.at<CPU>(), unit_cell_.spl_num_atoms().global_offset(),
                              unit_cell_.spl_num_atoms().local_size());

    }
    
    /* transform Hartree potential to real space */
    vh->fft_transform(1);

    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(vha_rg): %20.14f", vh->checksum_rg());
    #endif
    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(vha_it): %16llX", vh->f_it().hash());
    #endif
    
    /* compute contribution from the smooth part of Hartree potential */
    energy_vha_ = rho->inner(vh);
        
    /* add nucleus potential and contribution to Hartree energy */
    if (ctx_.full_potential())
    {
        double evha_nuc_ = 0;
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
        {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            auto& atom = unit_cell_.atom(ia);
            Spline<double> srho(atom.radial_grid());
            for (int ir = 0; ir < atom.num_mt_points(); ir++)
            {
                double r = atom.radial_grid(ir);
                hartree_potential_->f_mt<local>(0, ir, ialoc) -= atom.zn() / r / y00;
                srho[ir] = rho->f_mt<local>(0, ir, ialoc);
            }
            evha_nuc_ -= atom.zn() * srho.interpolate().integrate(1) / y00;
        }
        ctx_.comm().allreduce(&evha_nuc_, 1);
        energy_vha_ += evha_nuc_;
    }
}

};

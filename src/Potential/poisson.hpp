/** The following operation is performed:
 *  \f[
 *    q_{\ell m}^{\alpha} = \sum_{\bf G} 4\pi \rho({\bf G}) e^{i{\bf G}{\bf r}_{\alpha}}i^{\ell}f_{\ell}^{\alpha}(G) Y_{\ell m}^{*}(\hat{\bf G})
 *  \f]
 */
inline void Potential::poisson_sum_G(int lmmax__,
                                     double_complex const* fpw__,
                                     mdarray<double, 3>& fl__,
                                     matrix<double_complex>& flm__)
{
    PROFILE("sirius::Potential::poisson_sum_G");

    int ngv_loc = ctx_.gvec().count();

    int na_max{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        na_max = std::max(na_max, unit_cell_.atom_type(iat).num_atoms());
    }

    matrix<double_complex> phase_factors(ngv_loc, na_max, ctx_.main_memory_t());
    matrix<double_complex> zm(lmmax__, ngv_loc, ctx_.dual_memory_t());
    matrix<double_complex> tmp(lmmax__, na_max, ctx_.dual_memory_t());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        double t = -omp_get_wtime();
        int na = unit_cell_.atom_type(iat).num_atoms();
        ctx_.generate_phase_factors(iat, phase_factors);
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ngv_loc; igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            for (int lm = 0; lm < lmmax__; lm++) {
                int l = l_by_lm_[lm];
                zm(lm, igloc) = fourpi * fpw__[igloc] * zilm_[lm] *
                                fl__(l, iat, ctx_.gvec().shell(ig)) * std::conj(gvec_ylm_(lm, igloc));
            }
        }
        switch (ctx_.processing_unit()) {
            case CPU: {
                linalg<CPU>::gemm(0, 0, lmmax__, na, ngv_loc,
                                  zm.at<CPU>(), zm.ld(),
                                  phase_factors.at<CPU>(), phase_factors.ld(),
                                  tmp.at<CPU>(), tmp.ld());
                break;
            }
            case GPU: {
                #ifdef __GPU
                zm.copy<memory_t::host, memory_t::device>();
                linalg<GPU>::gemm(0, 0, lmmax__, na, ngv_loc,
                                  zm.at<GPU>(), zm.ld(),
                                  phase_factors.at<GPU>(), phase_factors.ld(),
                                  tmp.at<GPU>(), tmp.ld());
                tmp.copy_to_host();
                #endif
                break;
            }
        }

        if (ctx_.control().print_performance_) {
            t += omp_get_wtime();
            if (comm_.rank() == 0) {
                printf("poisson_sum_G() performance: %12.6f GFlops/rank, [m,n,k=%i %i %i, time=%f (sec)]\n",
                       8e-9 * lmmax__ * na * ctx_.gvec().num_gvec() / t / comm_.size(), lmmax__, na, ctx_.gvec().num_gvec(), t);
            }
        }
        for (int i = 0; i < na; i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            for (int lm = 0; lm < lmmax__; lm++) {
                flm__(lm, ia) = tmp(lm, i);
            }
        }
    }
    
    ctx_.comm().allreduce(&flm__(0, 0), (int)flm__.size());
}

inline void Potential::poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt__,
                                             mdarray<double_complex, 2>& qit__,
                                             double_complex* rho_pw__)
{
    PROFILE("sirius::Potential::poisson_add_pseudo_pw");

    int lmmax = ctx_.lmmax_rho();
    int ngv = ctx_.gvec().count();
    
    /* The following term is added to the plane-wave coefficients of the charge density:
     * Integrate[SphericalBesselJ[l,a*x]*p[x,R]*x^2,{x,0,R},Assumptions->{l>=0,n>=0,R>0,a>0}] / 
     *  Integrate[p[x,R]*x^(2+l),{x,0,R},Assumptions->{h>=0,n>=0,R>0}]
     * i.e. contributon from pseudodensity to l-th channel of plane wave expansion multiplied by 
     * the difference bethween true and interstitial-in-the-mt multipole moments and divided by the 
     * moment of the pseudodensity.
     */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        double R = unit_cell_.atom_type(iat).mt_radius();
        int na = unit_cell_.atom_type(iat).num_atoms();
        
        mdarray<double_complex, 2> pf;
        mdarray<double_complex, 2> qa;
        mdarray<double_complex, 2> qapf;

        switch (ctx_.processing_unit()) {
            case CPU: {
                double_complex* buf_ptr = (double_complex*)ctx_.memory_buffer(sizeof(double_complex) * (ngv * (lmmax + na) + lmmax * na));
                pf = mdarray<double_complex, 2>(buf_ptr, ngv, na);
                qa = mdarray<double_complex, 2>(buf_ptr + pf.size(), lmmax, na);
                qapf = mdarray<double_complex, 2>(buf_ptr + pf.size() + qa.size(), lmmax, ngv);
                break;
            }
            case GPU: {
                /* allocate on GPU */
                pf = mdarray<double_complex, 2>(ngv, na, memory_t::device);
                /* allocate on CPU & GPU */
                qa = mdarray<double_complex, 2>(lmmax, na, ctx_.dual_memory_t());
                double_complex* buf_ptr = (double_complex*)ctx_.memory_buffer(sizeof(double_complex) * ngv * lmmax);
                /* create on CPU and allocate on GPU */
                qapf = mdarray<double_complex, 2>(buf_ptr, lmmax, ngv);
                qapf.allocate(memory_t::device);
                break;
            }
        }

        ctx_.generate_phase_factors(iat, pf);

        for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                qa(lm, i) = qmt__(lm, ia) - qit__(lm, ia);
            }
        }

        switch (ctx_.processing_unit()) {
            case CPU: {
                linalg<CPU>::gemm(0, 2, ctx_.lmmax_rho(), ctx_.gvec().count(), unit_cell_.atom_type(iat).num_atoms(),
                                  qa, pf, qapf);
                break;
            }
            case GPU: {
                #ifdef __GPU
                qa.copy<memory_t::host, memory_t::device>();
                linalg<GPU>::gemm(0, 2, ctx_.lmmax_rho(), ctx_.gvec().count(), unit_cell_.atom_type(iat).num_atoms(),
                                  qa.at<GPU>(), qa.ld(),
                                  pf.at<GPU>(), pf.ld(),
                                  qapf.at<GPU>(), qapf.ld());
                qapf.copy_to_host();
                #endif
                break;
            }
        }

        /* add pseudo_density to interstitial charge density so that rho(G) has the correct 
         * multipole moments in the muffin-tins */
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;

            double gR = ctx_.gvec().gvec_len(ig) * R;
            double gRn = std::pow(2.0 / gR, pseudo_density_order_ + 1);

            double_complex rho_G(0, 0);
            if (ig) { // G!=0
                /* loop over atoms of the same type */
                for (int l = 0, lm = 0; l <= ctx_.lmax_rho(); l++) {
                    double_complex zt1(0, 0);
                    for (int m = -l; m <= l; m++, lm++) {
                        zt1 += gvec_ylm_(lm, igloc) * qapf(lm, igloc);
                    }
                    rho_G += (fourpi / unit_cell_.omega()) * std::conj(zil_[l]) * zt1 * gamma_factors_R_(l, iat) * 
                             sbessel_mt_(l + pseudo_density_order_ + 1, iat, ctx_.gvec().shell(ig)) * gRn;
                } // l
            } else { // G=0
                for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
                    int ia = unit_cell_.atom_type(iat).atom_id(i);
                    rho_G += (fourpi / unit_cell_.omega()) * y00 * (qmt__(0, ia) - qit__(0, ia));
                }
            }
            rho_pw__[igloc] += rho_G;
        }
    }
}

inline void Potential::poisson(Periodic_function<double> const& rho)
{
    PROFILE("sirius::Potential::poisson");

    /* in case of full potential we need to do pseudo-charge multipoles */
    if (ctx_.full_potential()) {

        /* true multipole moments */
        mdarray<double_complex, 2> qmt(ctx_.lmmax_rho(), unit_cell_.num_atoms());
        poisson_vmt(rho, qmt);
        
        //== for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        //==     for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
        //==         printf("qmt(%2i, %2i) = %18.12f %18.12f\n", lm, ia, qmt(lm, ia).real(), qmt(lm, ia).imag());
        //==     }
        //==     printf("\n");
        //== }

        if (ctx_.control().print_checksum_) {
            if (ctx_.comm().rank() == 0) {
                print_checksum("qmt", qmt.checksum());
            }
        }

        /* compute multipoles of interstitial density in MT region */
        mdarray<double_complex, 2> qit(ctx_.lmmax_rho(), unit_cell_.num_atoms());
        poisson_sum_G(ctx_.lmmax_rho(), &rho.f_pw_local(0), sbessel_mom_, qit);

        //== for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        //==     for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
        //==         printf("qi(%2i, %2i) = %18.12f %18.12f\n", lm, ia, qit(lm, ia).real(), qit(lm, ia).imag());
        //==     }
        //==     printf("\n");
        //== }

        if (ctx_.control().print_checksum_) {
            if (ctx_.comm().rank() == 0) {
                print_checksum("qit", qit.checksum());
            }
        }

        /* add contribution from the pseudo-charge */
        poisson_add_pseudo_pw(qmt, qit, const_cast<double_complex*>(&rho.f_pw_local(0)));
        
        if (check_pseudo_charge) {
            poisson_sum_G(ctx_.lmmax_rho(), &rho.f_pw_local(0), sbessel_mom_, qit);

            double d = 0.0;
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                    d += abs(qmt(lm, ia) - qit(lm, ia));
                }
            }
            printf("pseudocharge error: %18.10f\n", d);
        }
    }

    /* compute pw coefficients of Hartree potential */
    int ig0{0};
    if (ctx_.gvec().comm().rank() == 0) {
        hartree_potential_->f_pw_local(0) = 0.0;
        ig0 = 1;
    }
    if (!ctx_.molecule()) {
        #pragma omp parallel for schedule(static)
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            hartree_potential_->f_pw_local(igloc) = (fourpi * rho.f_pw_local(igloc) / std::pow(ctx_.gvec().gvec_len(ig), 2));
        }
    } else {
        double R_cut = 0.5 * std::pow(unit_cell_.omega(), 1.0 / 3);
        #pragma omp parallel for schedule(static)
        for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
            int ig = ctx_.gvec().offset() + igloc;
            hartree_potential_->f_pw_local(igloc) = (fourpi * rho.f_pw_local(igloc) / std::pow(ctx_.gvec().gvec_len(ig), 2)) *
                                                    (1.0 - std::cos(ctx_.gvec().gvec_len(ig) * R_cut));
        }
    }
    
    //if (ctx_.control().print_checksum_) {
    //    auto z4 = mdarray<double_complex, 1>(&vh->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    //    if (ctx_.comm().rank() == 0) {
    //        DUMP("checksum(vh_pw): %20.14f %20.14f", z4.real(), z4.imag());
    //    }
    //}
    
    /* boundary condition for muffin-tins */
    if (ctx_.full_potential()) {
        /* compute V_lm at the MT boundary */
        mdarray<double_complex, 2> vmtlm(ctx_.lmmax_pot(), unit_cell_.num_atoms());
        poisson_sum_G(ctx_.lmmax_pot(), &hartree_potential_->f_pw_local(0), sbessel_mt_, vmtlm);
        
        /* add boundary condition and convert to Rlm */
        sddk::timer t1("sirius::Potential::poisson|bc");
        mdarray<double, 2> rRl(unit_cell_.max_num_mt_points(), ctx_.lmax_pot() + 1);
        int type_id_prev = -1;

        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            int nmtp = unit_cell_.atom(ia).num_mt_points();

            if (unit_cell_.atom(ia).type_id() != type_id_prev) {
                type_id_prev = unit_cell_.atom(ia).type_id();
            
                double R = unit_cell_.atom(ia).mt_radius();

                #pragma omp parallel for default(shared)
                for (int l = 0; l <= ctx_.lmax_pot(); l++) {
                    for (int ir = 0; ir < nmtp; ir++) {
                        rRl(ir, l) = std::pow(unit_cell_.atom(ia).type().radial_grid(ir) / R, l);
                    }
                }
            }

            std::vector<double> vlm(ctx_.lmmax_pot());
            SHT::convert(ctx_.lmax_pot(), &vmtlm(0, ia), &vlm[0]);
            
            #pragma omp parallel for default(shared)
            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) {
                    hartree_potential_->f_mt<index_domain_t::local>(lm, ir, ialoc) += vlm[lm] * rRl(ir, l);
                }
            }
            /* save electronic part of the potential at the point of origin */
            #ifdef __VHA_AUX
            vh_el_(ia) = y00 * hartree_potential_->f_mt<index_domain_t::local>(0, 0, ialoc) + unit_cell_.atom(ia).zn() / unit_cell_.atom(ia).radial_grid(0);
            #else
            vh_el_(ia) = y00 * hartree_potential_->f_mt<index_domain_t::local>(0, 0, ialoc);
            #endif
        }
        ctx_.comm().allgather(vh_el_.at<CPU>(), unit_cell_.spl_num_atoms().global_offset(),
                              unit_cell_.spl_num_atoms().local_size());
    }

    /* transform Hartree potential to real space */
    hartree_potential_->fft_transform(1);

    if (ctx_.control().print_checksum_) {
        auto cs = hartree_potential_->checksum_rg();
        if (ctx_.comm().rank() == 0) {
            print_checksum("vha_rg", cs);
        }
    }
    
    /* compute contribution from the smooth part of Hartree potential */
    energy_vha_ = rho.inner(hartree_potential_.get());
        
    #ifndef __VHA_AUX
    /* add nucleus potential and contribution to Hartree energy */
    if (ctx_.full_potential()) {
        double evha_nuc{0};
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            auto& atom = unit_cell_.atom(ia);
            Spline<double> srho(atom.radial_grid());
            for (int ir = 0; ir < atom.num_mt_points(); ir++) {
                double r = atom.radial_grid(ir);
                hartree_potential_->f_mt<index_domain_t::local>(0, ir, ialoc) -= atom.zn() / r / y00;
                srho[ir] = rho.f_mt<index_domain_t::local>(0, ir, ialoc);
            }
            evha_nuc -= atom.zn() * srho.interpolate().integrate(1) / y00;
        }
        ctx_.comm().allreduce(&evha_nuc, 1);
        energy_vha_ += evha_nuc;
    }
    #endif
}

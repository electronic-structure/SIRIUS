#include "potential.h"
#include "smooth_periodic_function.h"

namespace sirius {

void Potential::xc_mt_nonmagnetic(Radial_grid const& rgrid,
                                  std::vector<XC_functional*>& xc_func,
                                  Spheric_function<spectral, double>& rho_lm, 
                                  Spheric_function<spatial, double>& rho_tp, 
                                  Spheric_function<spatial, double>& vxc_tp, 
                                  Spheric_function<spatial, double>& exc_tp)
{
    Timer t("sirius::Potential::xc_mt_nonmagnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) if (ixc->gga()) is_gga = true;

    Spheric_function_gradient<spatial, double> grad_rho_tp(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> lapl_rho_tp;
    Spheric_function<spatial, double> grad_rho_grad_rho_tp;

    if (is_gga)
    {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_lm = gradient(rho_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) grad_rho_tp[x] = sht_->transform(grad_rho_lm[x]);

        /* compute density gradient product */
        grad_rho_grad_rho_tp = grad_rho_tp * grad_rho_tp;
        
        /* compute Laplacian in Rlm spherical harmonics */
        auto lapl_rho_lm = laplacian(rho_lm);

        /* backward transform Laplacian from Rlm to (theta, phi) */
        lapl_rho_tp = sht_->transform(lapl_rho_lm);
    }

    exc_tp.zero();
    vxc_tp.zero();

    Spheric_function<spatial, double> vsigma_tp;
    if (is_gga)
    {
        vsigma_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_tp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func)
    {
        /* if this is an LDA functional */
        if (ixc->lda())
        {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vxc_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++)
                {
                    ixc->get_lda(sht_->num_points(), &rho_tp(0, ir), &vxc_t[0], &exc_t[0]);
                    for (int itp = 0; itp < sht_->num_points(); itp++)
                    {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc */
                        vxc_tp(itp, ir) += vxc_t[itp];
                    }
                }
            }
        }
        if (ixc->gga())
        {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vrho_t(sht_->num_points());
                std::vector<double> vsigma_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++)
                {
                    ixc->get_gga(sht_->num_points(), &rho_tp(0, ir), &grad_rho_grad_rho_tp(0, ir), &vrho_t[0], &vsigma_t[0], &exc_t[0]);
                    for (int itp = 0; itp < sht_->num_points(); itp++)
                    {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc available contributions */
                        vxc_tp(itp, ir) += (vrho_t[itp] - 2 * vsigma_t[itp] * lapl_rho_tp(itp, ir));

                        /* save the sigma derivative */
                        vsigma_tp(itp, ir) += vsigma_t[itp]; 
                    }
                }
            }
        }
    }

    if (is_gga)
    {
        /* forward transform vsigma to Rlm */
        auto vsigma_lm = sht_->transform(vsigma_tp);

        /* compute gradient of vsgima in spherical harmonics */
        auto grad_vsigma_lm = gradient(vsigma_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        Spheric_function_gradient<spatial, double> grad_vsigma_tp(sht_->num_points(), rgrid);
        for (int x = 0; x < 3; x++) grad_vsigma_tp[x] = sht_->transform(grad_vsigma_lm[x]);

        /* compute scalar product of two gradients */
        auto grad_vsigma_grad_rho_tp = grad_vsigma_tp * grad_rho_tp;

        /* add remaining term to Vxc */
        for (int ir = 0; ir < rgrid.num_points(); ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++)
            {
                vxc_tp(itp, ir) -= 2 * grad_vsigma_grad_rho_tp(itp, ir);
            }
        }
    }
}

void Potential::xc_mt_magnetic(Radial_grid const& rgrid,
                               std::vector<XC_functional*>& xc_func,
                               Spheric_function<spectral, double>& rho_up_lm, 
                               Spheric_function<spatial, double>& rho_up_tp, 
                               Spheric_function<spectral, double>& rho_dn_lm, 
                               Spheric_function<spatial, double>& rho_dn_tp, 
                               Spheric_function<spatial, double>& vxc_up_tp, 
                               Spheric_function<spatial, double>& vxc_dn_tp, 
                               Spheric_function<spatial, double>& exc_tp)
{
    Timer t("sirius::Potential::xc_mt_magnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) if (ixc->gga()) is_gga = true;

    Spheric_function_gradient<spatial, double> grad_rho_up_tp(sht_->num_points(), rgrid);
    Spheric_function_gradient<spatial, double> grad_rho_dn_tp(sht_->num_points(), rgrid);

    Spheric_function<spatial, double> lapl_rho_up_tp(sht_->num_points(), rgrid);
    Spheric_function<spatial, double> lapl_rho_dn_tp(sht_->num_points(), rgrid);

    Spheric_function<spatial, double> grad_rho_up_grad_rho_up_tp;
    Spheric_function<spatial, double> grad_rho_dn_grad_rho_dn_tp;
    Spheric_function<spatial, double> grad_rho_up_grad_rho_dn_tp;

    assert(rho_up_lm.radial_grid().hash() == rho_dn_lm.radial_grid().hash());

    vxc_up_tp.zero();
    vxc_dn_tp.zero();
    exc_tp.zero();

    if (is_gga)
    {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_up_lm = gradient(rho_up_lm);
        auto grad_rho_dn_lm = gradient(rho_dn_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++)
        {
            grad_rho_up_tp[x] = sht_->transform(grad_rho_up_lm[x]);
            grad_rho_dn_tp[x] = sht_->transform(grad_rho_dn_lm[x]);
        }

        /* compute density gradient products */
        grad_rho_up_grad_rho_up_tp = grad_rho_up_tp * grad_rho_up_tp;
        grad_rho_up_grad_rho_dn_tp = grad_rho_up_tp * grad_rho_dn_tp;
        grad_rho_dn_grad_rho_dn_tp = grad_rho_dn_tp * grad_rho_dn_tp;
        
        /* compute Laplacians in Rlm spherical harmonics */
        auto lapl_rho_up_lm = laplacian(rho_up_lm);
        auto lapl_rho_dn_lm = laplacian(rho_dn_lm);

        /* backward transform Laplacians from Rlm to (theta, phi) */
        lapl_rho_up_tp = sht_->transform(lapl_rho_up_lm);
        lapl_rho_dn_tp = sht_->transform(lapl_rho_dn_lm);
    }

    Spheric_function<spatial, double> vsigma_uu_tp;
    Spheric_function<spatial, double> vsigma_ud_tp;
    Spheric_function<spatial, double> vsigma_dd_tp;
    if (is_gga)
    {
        vsigma_uu_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_uu_tp.zero();

        vsigma_ud_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_ud_tp.zero();

        vsigma_dd_tp = Spheric_function<spatial, double>(sht_->num_points(), rgrid);
        vsigma_dd_tp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func)
    {
        /* if this is an LDA functional */
        if (ixc->lda())
        {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vxc_up_t(sht_->num_points());
                std::vector<double> vxc_dn_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++)
                {
                    ixc->get_lda(sht_->num_points(), &rho_up_tp(0, ir), &rho_dn_tp(0, ir), &vxc_up_t[0], &vxc_dn_t[0], &exc_t[0]);
                    for (int itp = 0; itp < sht_->num_points(); itp++)
                    {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc */
                        vxc_up_tp(itp, ir) += vxc_up_t[itp];
                        vxc_dn_tp(itp, ir) += vxc_dn_t[itp];
                    }
                }
            }
        }
        if (ixc->gga())
        {
            #pragma omp parallel
            {
                std::vector<double> exc_t(sht_->num_points());
                std::vector<double> vrho_up_t(sht_->num_points());
                std::vector<double> vrho_dn_t(sht_->num_points());
                std::vector<double> vsigma_uu_t(sht_->num_points());
                std::vector<double> vsigma_ud_t(sht_->num_points());
                std::vector<double> vsigma_dd_t(sht_->num_points());
                #pragma omp for
                for (int ir = 0; ir < rgrid.num_points(); ir++)
                {
                    ixc->get_gga(sht_->num_points(), 
                                 &rho_up_tp(0, ir), 
                                 &rho_dn_tp(0, ir), 
                                 &grad_rho_up_grad_rho_up_tp(0, ir), 
                                 &grad_rho_up_grad_rho_dn_tp(0, ir), 
                                 &grad_rho_dn_grad_rho_dn_tp(0, ir),
                                 &vrho_up_t[0], 
                                 &vrho_dn_t[0],
                                 &vsigma_uu_t[0], 
                                 &vsigma_ud_t[0],
                                 &vsigma_dd_t[0],
                                 &exc_t[0]);

                    for (int itp = 0; itp < sht_->num_points(); itp++)
                    {
                        /* add Exc contribution */
                        exc_tp(itp, ir) += exc_t[itp];

                        /* directly add to Vxc available contributions */
                        vxc_up_tp(itp, ir) += (vrho_up_t[itp] - 2 * vsigma_uu_t[itp] * lapl_rho_up_tp(itp, ir) - vsigma_ud_t[itp] * lapl_rho_dn_tp(itp, ir));
                        vxc_dn_tp(itp, ir) += (vrho_dn_t[itp] - 2 * vsigma_dd_t[itp] * lapl_rho_dn_tp(itp, ir) - vsigma_ud_t[itp] * lapl_rho_up_tp(itp, ir));

                        /* save the sigma derivatives */
                        vsigma_uu_tp(itp, ir) += vsigma_uu_t[itp]; 
                        vsigma_ud_tp(itp, ir) += vsigma_ud_t[itp]; 
                        vsigma_dd_tp(itp, ir) += vsigma_dd_t[itp]; 
                    }
                }
            }
        }
    }

    if (is_gga)
    {
        /* forward transform vsigma to Rlm */
        auto vsigma_uu_lm = sht_->transform(vsigma_uu_tp);
        auto vsigma_ud_lm = sht_->transform(vsigma_ud_tp);
        auto vsigma_dd_lm = sht_->transform(vsigma_dd_tp);

        /* compute gradient of vsgima in spherical harmonics */
        auto grad_vsigma_uu_lm = gradient(vsigma_uu_lm);
        auto grad_vsigma_ud_lm = gradient(vsigma_ud_lm);
        auto grad_vsigma_dd_lm = gradient(vsigma_dd_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        Spheric_function_gradient<spatial, double> grad_vsigma_uu_tp(sht_->num_points(), rgrid);
        Spheric_function_gradient<spatial, double> grad_vsigma_ud_tp(sht_->num_points(), rgrid);
        Spheric_function_gradient<spatial, double> grad_vsigma_dd_tp(sht_->num_points(), rgrid);
        for (int x = 0; x < 3; x++)
        {
            grad_vsigma_uu_tp[x] = sht_->transform(grad_vsigma_uu_lm[x]);
            grad_vsigma_ud_tp[x] = sht_->transform(grad_vsigma_ud_lm[x]);
            grad_vsigma_dd_tp[x] = sht_->transform(grad_vsigma_dd_lm[x]);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_uu_grad_rho_up_tp = grad_vsigma_uu_tp * grad_rho_up_tp;
        auto grad_vsigma_dd_grad_rho_dn_tp = grad_vsigma_dd_tp * grad_rho_dn_tp;
        auto grad_vsigma_ud_grad_rho_up_tp = grad_vsigma_ud_tp * grad_rho_up_tp;
        auto grad_vsigma_ud_grad_rho_dn_tp = grad_vsigma_ud_tp * grad_rho_dn_tp;

        /* add remaining terms to Vxc */
        for (int ir = 0; ir < rgrid.num_points(); ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++)
            {
                vxc_up_tp(itp, ir) -= (2 * grad_vsigma_uu_grad_rho_up_tp(itp, ir) + grad_vsigma_ud_grad_rho_dn_tp(itp, ir));
                vxc_dn_tp(itp, ir) -= (2 * grad_vsigma_dd_grad_rho_dn_tp(itp, ir) + grad_vsigma_ud_grad_rho_up_tp(itp, ir));
            }
        }
    }
}

void Potential::xc_mt(Periodic_function<double>* rho, 
                      Periodic_function<double>* magnetization[3],
                      std::vector<XC_functional*>& xc_func,
                      Periodic_function<double>* vxc, 
                      Periodic_function<double>* bxc[3], 
                      Periodic_function<double>* exc)
{
    Timer t2("sirius::Potential::xc_mt");

    for (int ialoc = 0; ialoc < (int)unit_cell_.spl_num_atoms().local_size(); ialoc++)
    {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        auto& rgrid = unit_cell_.atom(ia)->radial_grid();
        int nmtp = unit_cell_.atom(ia)->num_mt_points();

        /* backward transform density from Rlm to (theta, phi) */
        auto rho_tp = sht_->transform(rho->f_mt(ialoc));

        /* backward transform magnetization from Rlm to (theta, phi) */
        std::vector< Spheric_function<spatial, double> > vecmagtp(parameters_.num_mag_dims());
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            vecmagtp[j] = sht_->transform(magnetization[j]->f_mt(ialoc));
       
        /* "up" component of the density */
        Spheric_function<spectral, double> rho_up_lm;
        Spheric_function<spatial, double> rho_up_tp(sht_->num_points(), rgrid);

        /* "dn" component of the density */
        Spheric_function<spectral, double> rho_dn_lm;
        Spheric_function<spatial, double> rho_dn_tp(sht_->num_points(), rgrid);

        /* check if density has negative values */
        double rhomin = 0.0;
        for (int ir = 0; ir < nmtp; ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++) rhomin = std::min(rhomin, rho_tp(itp, ir));
        }

        if (rhomin < 0.0)
        {
            std::stringstream s;
            s << "Charge density for atom " << ia << " has negative values" << std::endl
              << "most negatve value : " << rhomin << std::endl
              << "current Rlm expansion of the charge density may be not sufficient, try to increase lmax_rho";
            warning_local(__FILE__, __LINE__, s);
        }

        if (parameters_.num_spins() == 1)
        {
            for (int ir = 0; ir < nmtp; ir++)
            {
                /* fix negative density */
                for (int itp = 0; itp < sht_->num_points(); itp++) 
                {
                    if (rho_tp(itp, ir) < 0.0) rho_tp(itp, ir) = 0.0;
                }
            }
        }
        else
        {
            for (int ir = 0; ir < nmtp; ir++)
            {
                for (int itp = 0; itp < sht_->num_points(); itp++)
                {
                    /* compute magnitude of the magnetization vector */
                    double mag = 0.0;
                    for (int j = 0; j < parameters_.num_mag_dims(); j++) mag += pow(vecmagtp[j](itp, ir), 2);
                    mag = std::sqrt(mag);

                    /* in magnetic case fix both density and magnetization */
                    for (int itp = 0; itp < sht_->num_points(); itp++) 
                    {
                        if (rho_tp(itp, ir) < 0.0)
                        {
                            rho_tp(itp, ir) = 0.0;
                            mag = 0.0;
                        }
                        /* fix numerical noise at high values of magnetization */
                        mag = std::min(mag, rho_tp(itp, ir));
                    
                        /* compute "up" and "dn" components */
                        rho_up_tp(itp, ir) = 0.5 * (rho_tp(itp, ir) + mag);
                        rho_dn_tp(itp, ir) = 0.5 * (rho_tp(itp, ir) - mag);
                    }
                }
            }

            /* transform from (theta, phi) to Rlm */
            rho_up_lm = sht_->transform(rho_up_tp);
            rho_dn_lm = sht_->transform(rho_dn_tp);
        }

        Spheric_function<spatial, double> exc_tp(sht_->num_points(), rgrid);
        Spheric_function<spatial, double> vxc_tp(sht_->num_points(), rgrid);

        if (parameters_.num_spins() == 1)
        {
            xc_mt_nonmagnetic(rgrid, xc_func, rho->f_mt(ialoc), rho_tp, vxc_tp, exc_tp);
        }
        else
        {
            Spheric_function<spatial, double> vxc_up_tp(sht_->num_points(), rgrid);
            Spheric_function<spatial, double> vxc_dn_tp(sht_->num_points(), rgrid);

            xc_mt_magnetic(rgrid, xc_func, rho_up_lm, rho_up_tp, rho_dn_lm, rho_dn_tp, vxc_up_tp, vxc_dn_tp, exc_tp);

            for (int ir = 0; ir < nmtp; ir++)
            {
                for (int itp = 0; itp < sht_->num_points(); itp++)
                {
                    /* align magnetic filed parallel to magnetization */
                    /* use vecmagtp as temporary vector */
                    double mag =  rho_up_tp(itp, ir) - rho_dn_tp(itp, ir);
                    if (mag > 1e-8)
                    {
                        /* |Bxc| = 0.5 * (V_up - V_dn) */
                        double b = 0.5 * (vxc_up_tp(itp, ir) - vxc_dn_tp(itp, ir));
                        for (int j = 0; j < parameters_.num_mag_dims(); j++)
                            vecmagtp[j](itp, ir) = b * vecmagtp[j](itp, ir) / mag;
                    }
                    else
                    {
                        for (int j = 0; j < parameters_.num_mag_dims(); j++) vecmagtp[j](itp, ir) = 0.0;
                    }
                    /* Vxc = 0.5 * (V_up + V_dn) */
                    vxc_tp(itp, ir) = 0.5 * (vxc_up_tp(itp, ir) + vxc_dn_tp(itp, ir));
                }       
            }
            /* convert magnetic field back to Rlm */
            for (int j = 0; j < parameters_.num_mag_dims(); j++) sht_->transform(vecmagtp[j], bxc[j]->f_mt(ialoc));
        }

        /* forward transform from (theta, phi) to Rlm */
        sht_->transform(vxc_tp, vxc->f_mt(ialoc));
        sht_->transform(exc_tp, exc->f_mt(ialoc));
    }
}

void Potential::xc_it_nonmagnetic(Periodic_function<double>* rho, 
                                  std::vector<XC_functional*>& xc_func,
                                  Periodic_function<double>* vxc, 
                                  Periodic_function<double>* exc)
{
    Timer t("sirius::Potential::xc_it_nonmagnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) if (ixc->gga()) is_gga = true;

    splindex<block> spl_fft_size(fft_->size(), ctx_.comm().size(), ctx_.comm().rank());
    int num_loc_points = (int)spl_fft_size.local_size();
    
    /* check for negative values */
    double rhomin = 0.0;
    for (int ir = 0; ir < fft_->size(); ir++)
    {
        rhomin = std::min(rhomin, rho->f_it<global>(ir));
        if (rho->f_it<global>(ir) < 0.0)  rho->f_it<global>(ir) = 0.0;
    }
    if (rhomin < 0.0)
    {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        warning_global(__FILE__, __LINE__, s);
    }
    
    Smooth_periodic_function_gradient<spatial, double> grad_rho_it;
    Smooth_periodic_function<spatial, double> lapl_rho_it;
    Smooth_periodic_function<spatial, double> grad_rho_grad_rho_it;
    
    if (is_gga) 
    {
        Smooth_periodic_function<spatial, double> rho_it(&rho->f_it<global>(0), fft_, &ctx_.gvec());

        /* get plane-wave coefficients of the density */
        Smooth_periodic_function<spectral> rho_pw = transform(rho_it);

        /* generate pw coeffs of the gradient and laplacian */
        auto grad_rho_pw = gradient(rho_pw);
        auto lapl_rho_pw = laplacian(rho_pw);

        /* gradient in real space */
        for (int x = 0; x < 3; x++) 
            grad_rho_it[x] = transform<double>(grad_rho_pw[x], spl_fft_size);

        /* product of gradients */
        grad_rho_grad_rho_it = grad_rho_it * grad_rho_it;
        
        /* Laplacian in real space */
        lapl_rho_it = transform<double>(lapl_rho_pw, spl_fft_size);
    }

    mdarray<double, 1> exc_tmp(num_loc_points);
    exc_tmp.zero();

    mdarray<double, 1> vxc_tmp(num_loc_points);
    vxc_tmp.zero();

    mdarray<double, 1> vsigma_tmp;
    if (is_gga)
    {
        vsigma_tmp = mdarray<double, 1>(num_loc_points);
        vsigma_tmp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func)
    {
        #pragma omp parallel
        {
            /* split local size between threads */
            splindex<block> spl_t(num_loc_points, Platform::num_threads(), Platform::thread_id());

            std::vector<double> exc_t(spl_t.local_size());

            /* if this is an LDA functional */
            if (ixc->lda())
            {
                std::vector<double> vxc_t(spl_t.local_size());

                ixc->get_lda((int)spl_t.local_size(), &rho->f_it<local>((int)spl_t.global_offset()), &vxc_t[0], &exc_t[0]);

                for (int i = 0; i < (int)spl_t.local_size(); i++)
                {
                    /* add Exc contribution */
                    exc_tmp(spl_t[i]) += exc_t[i];

                    /* directly add to Vxc */
                    vxc_tmp(spl_t[i]) += vxc_t[i];
                }
            }
            if (ixc->gga())
            {
                std::vector<double> vrho_t(spl_t.local_size());
                std::vector<double> vsigma_t(spl_t.local_size());
                
                ixc->get_gga((int)spl_t.local_size(), 
                             &rho->f_it<local>((int)spl_t.global_offset()), 
                             &grad_rho_grad_rho_it((int)spl_t.global_offset()), 
                             &vrho_t[0], 
                             &vsigma_t[0], 
                             &exc_t[0]);


                for (int i = 0; i < (int)spl_t.local_size(); i++)
                {
                    /* add Exc contribution */
                    exc_tmp(spl_t[i]) += exc_t[i];

                    /* directly add to Vxc available contributions */
                    vxc_tmp(spl_t[i]) += (vrho_t[i] - 2 * vsigma_t[i] * lapl_rho_it((int)spl_t[i]));

                    /* save the sigma derivative */
                    vsigma_tmp(spl_t[i]) += vsigma_t[i]; 
                }
            }
        }
    }

    if (is_gga)
    {
        /* gather vsigma */
        Smooth_periodic_function<spatial, double> vsigma_it(fft_, &ctx_.gvec());
        ctx_.comm().allgather(&vsigma_tmp(0), &vsigma_it(0), (int)spl_fft_size.global_offset(), num_loc_points);

        /* forward transform vsigma to plane-wave domain */
        Smooth_periodic_function<spectral> vsigma_pw = transform(vsigma_it);
        
        /* gradient of vsigma in plane-wave domain */
        auto grad_vsigma_pw = gradient(vsigma_pw);

        /* backward transform gradient from pw to real space */
        Smooth_periodic_function_gradient<spatial, double> grad_vsigma_it;
        for (int x = 0; x < 3; x++) grad_vsigma_it[x] = transform<double>(grad_vsigma_pw[x], spl_fft_size);

        /* compute scalar product of two gradients */
        auto grad_vsigma_grad_rho_it = grad_vsigma_it * grad_rho_it;

        /* add remaining term to Vxc */
        for (int ir = 0; ir < num_loc_points; ir++)
        {
            vxc_tmp(ir) -= 2 * grad_vsigma_grad_rho_it(ir);
        }
    }

    for (int irloc = 0; irloc < num_loc_points; irloc++)
    {
        vxc->f_it<local>(irloc) = vxc_tmp(irloc);
        exc->f_it<local>(irloc) = exc_tmp(irloc);
    }
    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(vxc_tmp): %18.10f", vxc_tmp.checksum());
    DUMP("checksum(exc_tmp): %18.10f", exc_tmp.checksum());
    #endif
    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(vxc_tmp): %16llX", vxc_tmp.hash());
    DUMP("hash(exc_tmp): %16llX", exc_tmp.hash());
    #endif
}

void Potential::xc_it_magnetic(Periodic_function<double>* rho, 
                               Periodic_function<double>* magnetization[3], 
                               std::vector<XC_functional*>& xc_func,
                               Periodic_function<double>* vxc, 
                               Periodic_function<double>* bxc[3], 
                               Periodic_function<double>* exc)
{
    Timer t("sirius::Potential::xc_it_magnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) if (ixc->gga()) is_gga = true;

    splindex<block> spl_fft_size(fft_->size(), ctx_.comm().size(), ctx_.comm().rank());
    int num_loc_points = (int)spl_fft_size.local_size();
    
    Smooth_periodic_function<spatial, double> rho_up_it(fft_, &ctx_.gvec());
    Smooth_periodic_function<spatial, double> rho_dn_it(fft_, &ctx_.gvec());

    /* compute "up" and "dn" components and also check for negative values of density */
    double rhomin = 0.0;

    for (int ir = 0; ir < fft_->size(); ir++)
    {
        double mag = 0.0;
        for (int j = 0; j < parameters_.num_mag_dims(); j++) mag += pow(magnetization[j]->f_it<global>(ir), 2);
        mag = std::sqrt(mag);

        /* remove numerical noise at high values of magnetization */
        mag = std::min(mag, rho->f_it<global>(ir));

        rhomin = std::min(rhomin, rho->f_it<global>(ir));
        if (rho->f_it<global>(ir) < 0.0)
        {
            rho->f_it<global>(ir) = 0.0;
            mag = 0.0;
        }
        
        rho_up_it(ir) = 0.5 * (rho->f_it<global>(ir) + mag);
        rho_dn_it(ir) = 0.5 * (rho->f_it<global>(ir) - mag);
    }

    if (rhomin < 0.0)
    {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        warning_global(__FILE__, __LINE__, s);
    }

    Smooth_periodic_function_gradient<spatial, double> grad_rho_up_it;
    Smooth_periodic_function_gradient<spatial, double> grad_rho_dn_it;
    Smooth_periodic_function<spatial, double> lapl_rho_up_it;
    Smooth_periodic_function<spatial, double> lapl_rho_dn_it;
    Smooth_periodic_function<spatial, double> grad_rho_up_grad_rho_up_it;
    Smooth_periodic_function<spatial, double> grad_rho_up_grad_rho_dn_it;
    Smooth_periodic_function<spatial, double> grad_rho_dn_grad_rho_dn_it;
    
    if (is_gga) 
    {
        /* get plane-wave coefficients of the density */
        Smooth_periodic_function<spectral> rho_up_pw = transform(rho_up_it);
        Smooth_periodic_function<spectral> rho_dn_pw = transform(rho_dn_it);

        /* generate pw coeffs of the gradient and laplacian */
        auto grad_rho_up_pw = gradient(rho_up_pw);
        auto grad_rho_dn_pw = gradient(rho_dn_pw);
        auto lapl_rho_up_pw = laplacian(rho_up_pw);
        auto lapl_rho_dn_pw = laplacian(rho_dn_pw);

        /* gradient in real space */
        for (int x = 0; x < 3; x++)
        {
            grad_rho_up_it[x] = transform<double>(grad_rho_up_pw[x], spl_fft_size);
            grad_rho_dn_it[x] = transform<double>(grad_rho_dn_pw[x], spl_fft_size);
        }

        /* product of gradients */
        grad_rho_up_grad_rho_up_it = grad_rho_up_it * grad_rho_up_it;
        grad_rho_up_grad_rho_dn_it = grad_rho_up_it * grad_rho_dn_it;
        grad_rho_dn_grad_rho_dn_it = grad_rho_dn_it * grad_rho_dn_it;
        
        /* Laplacian in real space */
        lapl_rho_up_it = transform<double>(lapl_rho_up_pw, spl_fft_size);
        lapl_rho_dn_it = transform<double>(lapl_rho_dn_pw, spl_fft_size);
    }

    mdarray<double, 1> exc_tmp(num_loc_points);
    exc_tmp.zero();

    mdarray<double, 1> vxc_up_tmp(num_loc_points);
    vxc_up_tmp.zero();

    mdarray<double, 1> vxc_dn_tmp(num_loc_points);
    vxc_dn_tmp.zero();

    mdarray<double, 1> vsigma_uu_tmp;
    mdarray<double, 1> vsigma_ud_tmp;
    mdarray<double, 1> vsigma_dd_tmp;

    if (is_gga)
    {
        vsigma_uu_tmp = mdarray<double, 1>(num_loc_points);
        vsigma_uu_tmp.zero();
        
        vsigma_ud_tmp = mdarray<double, 1>(num_loc_points);
        vsigma_ud_tmp.zero();
        
        vsigma_dd_tmp = mdarray<double, 1>(num_loc_points);
        vsigma_dd_tmp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func)
    {
        #pragma omp parallel
        {
            /* split local size between threads */
            splindex<block> spl_t(num_loc_points, Platform::num_threads(), Platform::thread_id());

            std::vector<double> exc_t(spl_t.local_size());

            /* if this is an LDA functional */
            if (ixc->lda())
            {
                std::vector<double> vxc_up_t(spl_t.local_size());
                std::vector<double> vxc_dn_t(spl_t.local_size());

                ixc->get_lda((int)spl_t.local_size(), 
                             &rho_up_it(spl_fft_size.global_offset() + spl_t.global_offset()), 
                             &rho_dn_it(spl_fft_size.global_offset() + spl_t.global_offset()), 
                             &vxc_up_t[0], 
                             &vxc_dn_t[0], 
                             &exc_t[0]);

                for (int i = 0; i < (int)spl_t.local_size(); i++)
                {
                    /* add Exc contribution */
                    exc_tmp(spl_t[i]) += exc_t[i];

                    /* directly add to Vxc */
                    vxc_up_tmp(spl_t[i]) += vxc_up_t[i];
                    vxc_dn_tmp(spl_t[i]) += vxc_dn_t[i];
                }
            }
            if (ixc->gga())
            {
                std::vector<double> vrho_up_t(spl_t.local_size());
                std::vector<double> vrho_dn_t(spl_t.local_size());
                std::vector<double> vsigma_uu_t(spl_t.local_size());
                std::vector<double> vsigma_ud_t(spl_t.local_size());
                std::vector<double> vsigma_dd_t(spl_t.local_size());
                
                ixc->get_gga((int)spl_t.local_size(), 
                             &rho_up_it(spl_fft_size.global_offset() + spl_t.global_offset()), 
                             &rho_dn_it(spl_fft_size.global_offset() + spl_t.global_offset()), 
                             &grad_rho_up_grad_rho_up_it(spl_t.global_offset()), 
                             &grad_rho_up_grad_rho_dn_it(spl_t.global_offset()), 
                             &grad_rho_dn_grad_rho_dn_it(spl_t.global_offset()), 
                             &vrho_up_t[0], 
                             &vrho_dn_t[0], 
                             &vsigma_uu_t[0], 
                             &vsigma_ud_t[0], 
                             &vsigma_dd_t[0], 
                             &exc_t[0]);

                for (int i = 0; i < (int)spl_t.local_size(); i++)
                {
                    /* add Exc contribution */
                    exc_tmp(spl_t[i]) += exc_t[i];

                    /* directly add to Vxc available contributions */
                    vxc_up_tmp(spl_t[i]) += (vrho_up_t[i] - 2 * vsigma_uu_t[i] * lapl_rho_up_it(spl_t[i]) - vsigma_ud_t[i] * lapl_rho_dn_it(spl_t[i]));
                    vxc_dn_tmp(spl_t[i]) += (vrho_dn_t[i] - 2 * vsigma_dd_t[i] * lapl_rho_dn_it(spl_t[i]) - vsigma_ud_t[i] * lapl_rho_up_it(spl_t[i]));

                    /* save the sigma derivative */
                    vsigma_uu_tmp(spl_t[i]) += vsigma_uu_t[i]; 
                    vsigma_ud_tmp(spl_t[i]) += vsigma_ud_t[i]; 
                    vsigma_dd_tmp(spl_t[i]) += vsigma_dd_t[i]; 
                }
            }
        }
    }

    if (is_gga)
    {
        /* gather vsigma */
        Smooth_periodic_function<spatial, double> vsigma_uu_it(fft_, &ctx_.gvec());
        Smooth_periodic_function<spatial, double> vsigma_ud_it(fft_, &ctx_.gvec());
        Smooth_periodic_function<spatial, double> vsigma_dd_it(fft_, &ctx_.gvec());
        int global_offset = (int)spl_fft_size.global_offset();
        ctx_.comm().allgather(&vsigma_uu_tmp(0), &vsigma_uu_it(0), global_offset, num_loc_points);
        ctx_.comm().allgather(&vsigma_ud_tmp(0), &vsigma_ud_it(0), global_offset, num_loc_points);
        ctx_.comm().allgather(&vsigma_dd_tmp(0), &vsigma_dd_it(0), global_offset, num_loc_points);

        /* forward transform vsigma to plane-wave domain */
        Smooth_periodic_function<spectral> vsigma_uu_pw = transform(vsigma_uu_it);
        Smooth_periodic_function<spectral> vsigma_ud_pw = transform(vsigma_ud_it);
        Smooth_periodic_function<spectral> vsigma_dd_pw = transform(vsigma_dd_it);
        
        /* gradient of vsigma in plane-wave domain */
        auto grad_vsigma_uu_pw = gradient(vsigma_uu_pw);
        auto grad_vsigma_ud_pw = gradient(vsigma_ud_pw);
        auto grad_vsigma_dd_pw = gradient(vsigma_dd_pw);

        /* backward transform gradient from pw to real space */
        Smooth_periodic_function_gradient<spatial, double> grad_vsigma_uu_it;
        Smooth_periodic_function_gradient<spatial, double> grad_vsigma_ud_it;
        Smooth_periodic_function_gradient<spatial, double> grad_vsigma_dd_it;
        for (int x = 0; x < 3; x++)
        {
            grad_vsigma_uu_it[x] = transform<double>(grad_vsigma_uu_pw[x], spl_fft_size);
            grad_vsigma_ud_it[x] = transform<double>(grad_vsigma_ud_pw[x], spl_fft_size);
            grad_vsigma_dd_it[x] = transform<double>(grad_vsigma_dd_pw[x], spl_fft_size);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_uu_grad_rho_up_it = grad_vsigma_uu_it * grad_rho_up_it;
        auto grad_vsigma_dd_grad_rho_dn_it = grad_vsigma_dd_it * grad_rho_dn_it;
        auto grad_vsigma_ud_grad_rho_up_it = grad_vsigma_ud_it * grad_rho_up_it;
        auto grad_vsigma_ud_grad_rho_dn_it = grad_vsigma_ud_it * grad_rho_dn_it;

        /* add remaining term to Vxc */
        for (int ir = 0; ir < num_loc_points; ir++)
        {
            vxc_up_tmp(ir) -= (2 * grad_vsigma_uu_grad_rho_up_it(ir) + grad_vsigma_ud_grad_rho_dn_it(ir)); 
            vxc_dn_tmp(ir) -= (2 * grad_vsigma_dd_grad_rho_dn_it(ir) + grad_vsigma_ud_grad_rho_up_it(ir)); 
        }
    }

    for (int irloc = 0; irloc < num_loc_points; irloc++)
    {
        exc->f_it<local>(irloc) = exc_tmp(irloc);
        vxc->f_it<local>(irloc) = 0.5 * (vxc_up_tmp(irloc) + vxc_dn_tmp(irloc));
        double m = rho_up_it(spl_fft_size.global_offset() + irloc) - rho_dn_it(spl_fft_size.global_offset() + irloc);

        if (m > 1e-8)
        {
            double b = 0.5 * (vxc_up_tmp(irloc) - vxc_dn_tmp(irloc));
            for (int j = 0; j < parameters_.num_mag_dims(); j++)
               bxc[j]->f_it<local>(irloc) = b * magnetization[j]->f_it<local>(irloc) / m;
       }
       else
       {
           for (int j = 0; j < parameters_.num_mag_dims(); j++) bxc[j]->f_it<local>(irloc) = 0.0;
       }
    }
}


void Potential::xc(Periodic_function<double>* rho, 
                   Periodic_function<double>* magnetization[3], 
                   Periodic_function<double>* vxc, 
                   Periodic_function<double>* bxc[3], 
                   Periodic_function<double>* exc)
{
    Timer t("sirius::Potential::xc", ctx_.comm());

    if (parameters_.xc_functionals_input_section().xc_functional_names_.size() == 0)
    {
        vxc->zero();
        exc->zero();
        for (int i = 0; i < parameters_.num_mag_dims(); i++) bxc[i]->zero();
        return;
    }

    /* create list of XC functionals */
    std::vector<XC_functional*> xc_func;
    for (int i = 0; i < (int)parameters_.xc_functionals_input_section().xc_functional_names_.size(); i++)
    {
        std::string xc_label = parameters_.xc_functionals_input_section().xc_functional_names_[i];
        xc_func.push_back(new XC_functional(xc_label, parameters_.num_spins()));
    }
   
    if (parameters_.full_potential()) xc_mt(rho, magnetization, xc_func, vxc, bxc, exc);
    
    if (parameters_.num_spins() == 1)
    {
        xc_it_nonmagnetic(rho, xc_func, vxc, exc);
    }
    else
    {
        xc_it_magnetic(rho, magnetization, xc_func, vxc, bxc, exc);
    }

    for (auto& ixc: xc_func) delete ixc;
}

};

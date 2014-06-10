// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file potential.cpp
 *   
 *  \brief Contains remaining implementation of sirius::Potential class.
 */

#include "potential.h"
#include "smooth_periodic_function.h"

namespace sirius {

// TODO: everything here must be documented
// TODO: better naming convention: q is meaningless

Potential::Potential(Global& parameters__) : parameters_(parameters__), pseudo_density_order(9)
{
    Timer t("sirius::Potential::Potential");
    
    fft_ = parameters_.reciprocal_lattice()->fft();

    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            lmax_ = std::max(parameters_.lmax_rho(), parameters_.lmax_pot());
            sht_ = new SHT(lmax_);
            break;
        }
        case ultrasoft_pseudopotential:
        {
            lmax_ = parameters_.lmax_beta() * 2;
            break;
        }
        default:
        {
            stop_here
        }
    }

    l_by_lm_ = Utils::l_by_lm(lmax_);

    /* precompute i^l */
    zil_.resize(lmax_ + 1);
    for (int l = 0; l <= lmax_; l++) zil_[l] = pow(double_complex(0, 1), l);
    
    zilm_.resize(Utils::lmmax(lmax_));
    for (int l = 0, lm = 0; l <= lmax_; l++)
    {
        for (int m = -l; m <= l; m++, lm++) zilm_[lm] = zil_[l];
    }

    int ngv = (use_second_variation) ? 0 : parameters_.reciprocal_lattice()->num_gvec();

    effective_potential_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot(), parameters_.reciprocal_lattice()->num_gvec());
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        effective_magnetic_field_[j] = new Periodic_function<double>(parameters_, parameters_.lmmax_pot(), ngv);
    
    hartree_potential_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot(), parameters_.reciprocal_lattice()->num_gvec());
    hartree_potential_->allocate(false, true);
    
    xc_potential_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot());
    xc_potential_->allocate(false, false);
    
    xc_energy_density_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot());
    xc_energy_density_->allocate(false, false);

    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        local_potential_ = new Periodic_function<double>(parameters_, 0);
        local_potential_->allocate(false, true);
        local_potential_->zero();

        generate_local_potential();
    }

    vh_el_.set_dimensions(parameters_.unit_cell()->num_atoms());
    vh_el_.allocate();

    update();
}

Potential::~Potential()
{
    delete effective_potential_; 
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete effective_magnetic_field_[j];
    if (parameters_.esm_type() == full_potential_lapwlo) delete sht_;
    delete hartree_potential_;
    delete xc_potential_;
    delete xc_energy_density_;
    if (parameters_.esm_type() == ultrasoft_pseudopotential) delete local_potential_;
}

void Potential::update()
{
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        /* compute values of spherical Bessel functions at MT boundary */
        sbessel_mt_.set_dimensions(lmax_ + pseudo_density_order + 2, parameters_.unit_cell()->num_atom_types(), 
                                   parameters_.reciprocal_lattice()->num_gvec_shells_inner());
        sbessel_mt_.allocate();

        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            for (int igs = 0; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
            {
                gsl_sf_bessel_jl_array(lmax_ + pseudo_density_order + 1, 
                                       parameters_.reciprocal_lattice()->gvec_shell_len(igs) * parameters_.unit_cell()->atom_type(iat)->mt_radius(), 
                                       &sbessel_mt_(0, iat, igs));
            }
        }

        /* compute moments of spherical Bessel functions 
         *
         * In[]:= Integrate[SphericalBesselJ[l,G*x]*x^(2+l),{x,0,R},Assumptions->{R>0,G>0,l>=0}]
         * Out[]= (Sqrt[\[Pi]/2] R^(3/2+l) BesselJ[3/2+l,G R])/G^(3/2)
         *
         * and use relation between Bessel and spherical Bessel functions: 
         * Subscript[j, n](z)=Sqrt[\[Pi]/2]/Sqrt[z]Subscript[J, n+1/2](z) 
         */
        sbessel_mom_.set_dimensions(parameters_.lmax_rho() + 1, parameters_.unit_cell()->num_atom_types(), 
                                    parameters_.reciprocal_lattice()->num_gvec_shells_inner());
        sbessel_mom_.allocate();
        sbessel_mom_.zero();

        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            sbessel_mom_(0, iat, 0) = pow(parameters_.unit_cell()->atom_type(iat)->mt_radius(), 3) / 3.0; // for |G|=0
            for (int igs = 1; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
            {
                for (int l = 0; l <= parameters_.lmax_rho(); l++)
                {
                    sbessel_mom_(l, iat, igs) = pow(parameters_.unit_cell()->atom_type(iat)->mt_radius(), l + 2) * 
                                                sbessel_mt_(l + 1, iat, igs) / parameters_.reciprocal_lattice()->gvec_shell_len(igs);
                }
            }
        }
        
        /* compute Gamma[5/2 + n + l] / Gamma[3/2 + l] / R^l
         *
         * use Gamma[1/2 + p] = (2p - 1)!!/2^p Sqrt[Pi]
         */
        gamma_factors_R_.set_dimensions(parameters_.lmax_rho() + 1, parameters_.unit_cell()->num_atom_types());
        gamma_factors_R_.allocate();
        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            for (int l = 0; l <= parameters_.lmax_rho(); l++)
            {
                long double Rl = pow(parameters_.unit_cell()->atom_type(iat)->mt_radius(), l);

                int n_min = (2 * l + 3);
                int n_max = (2 * l + 1) + (2 * pseudo_density_order + 2);
                /* split factorial product into two parts to avoid overflow */
                long double f1 = 1.0;
                long double f2 = 1.0;
                for (int n = n_min; n <= n_max; n += 2) 
                {
                    if (f1 < Rl) 
                    {
                        f1 *= (n / 2.0);
                    }
                    else
                    {
                        f2 *= (n / 2.0);
                    }
                }
                gamma_factors_R_(l, iat) = static_cast<double>((f1 / Rl) * f2);
            }
        }
    }
}

void Potential::poisson_vmt(std::vector< Spheric_function<spectral, double_complex> >& rho_ylm, 
                            std::vector< Spheric_function<spectral, double_complex> >& vh_ylm, 
                            mdarray<double_complex, 2>& qmt)
{
    Timer t("sirius::Potential::poisson_vmt");

    qmt.zero();
    
    for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);

        double R = parameters_.unit_cell()->atom(ia)->mt_radius();
        int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();
       
        #pragma omp parallel default(shared)
        {
            std::vector<double_complex> g1;
            std::vector<double_complex> g2;

            Spline<double_complex> rholm(parameters_.unit_cell()->atom(ia)->radial_grid());

            #pragma omp for
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
            {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) rholm[ir] = rho_ylm[ialoc](lm, ir);
                rholm.interpolate();

                /* save multipole moment */
                qmt(lm, ia) = rholm.integrate(g1, l + 2);
                
                if (lm < parameters_.lmmax_pot())
                {
                    rholm.integrate(g2, 1 - l);
                    
                    double d1 = 1.0 / pow(R, 2 * l + 1); 
                    double d2 = 1.0 / double(2 * l + 1); 
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double r = parameters_.unit_cell()->atom(ia)->radial_grid(ir);

                        double_complex vlm = (1.0 - pow(r / R, 2 * l + 1)) * g1[ir] / pow(r, l + 1) +
                                             (g2[nmtp - 1] - g2[ir]) * pow(r, l) - 
                                             (g1[nmtp - 1] - g1[ir]) * pow(r, l) * d1;

                        vh_ylm[ialoc](lm, ir) = fourpi * vlm * d2;
                    }
                }
            }
        }
        
        /* fixed part of nuclear potential */
        for (int ir = 0; ir < nmtp; ir++)
        {
            //double r = parameters_.unit_cell()->atom(ia)->radial_grid(ir);
            //vh_ylm[ialoc](0, ir) -= parameters_.unit_cell()->atom(ia)->zn() * (1 / r - 1 / R) / y00;
            vh_ylm[ialoc](0, ir) += parameters_.unit_cell()->atom(ia)->zn() / R / y00;
        }

        //== /* write spherical potential */
        //== std::stringstream sstr;
        //== sstr << "mt_spheric_potential_" << ia << ".dat";
        //== FILE* fout = fopen(sstr.str().c_str(), "w");

        //== for (int ir = 0; ir < nmtp; ir++)
        //== {
        //==     double r = parameters_.unit_cell()->atom(ia)->radial_grid(ir);
        //==     fprintf(fout, "%20.10f %20.10f \n", r, real(vh_ylm[ialoc](0, ir)));
        //== }
        //== fclose(fout);
        //== stop_here
        
        /* nuclear multipole moment */
        qmt(0, ia) -= parameters_.unit_cell()->atom(ia)->zn() * y00;
    }

    Platform::allreduce(&qmt(0, 0), (int)qmt.size());
}

void Potential::poisson_sum_G(int lmmax__,
                              double_complex* fpw__,
                              mdarray<double, 3>& fl__,
                              mdarray<double_complex, 2>& flm__)
{
    Timer t("sirius::Potential::poisson_sum_G");
    
    //== flm.zero();

    //== mdarray<double_complex, 2> zm1(parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), parameters_.lmmax_rho());

    //== #pragma omp parallel for default(shared)
    //== for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
    //== {
    //==     for (int igloc = 0; igloc < (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
    //==         zm1(igloc, lm) = parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc) * conj(fpw[parameters_.reciprocal_lattice()->spl_num_gvec(igloc)] * zilm_[lm]);
    //== }

    //== mdarray<double_complex, 2> zm2(parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), parameters_.unit_cell()->num_atoms());

    //== for (int l = 0; l <= parameters_.lmax_rho(); l++)
    //== {
    //==     #pragma omp parallel for default(shared)
    //==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    //==     {
    //==         int iat = parameters_.unit_cell()->atom(ia)->type_id();
    //==         for (int igloc = 0; igloc < (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
    //==         {
    //==             int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
    //==             zm2(igloc, ia) = fourpi * parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia) *  
    //==                              fl(l, iat, parameters_.reciprocal_lattice()->gvec_shell(ig));
    //==         }
    //==     }

    //==     blas<cpu>::gemm(2, 0, 2 * l + 1, parameters_.unit_cell()->num_atoms(), 
    //==                     (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), 
    //==                     &zm1(0, Utils::lm_by_l_m(l, -l)), zm1.ld(), &zm2(0, 0), zm2.ld(), 
    //==                     &flm(Utils::lm_by_l_m(l, -l), 0), parameters_.lmmax_rho());
    //== }

    int ngv_loc = (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size();
    #pragma omp parallel
    {
        mdarray<double_complex, 2> zm(lmmax__, ngv_loc);
        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int iat = parameters_.unit_cell()->atom(ia)->type_id();
            for (int igloc = 0; igloc < ngv_loc; igloc++)
            {
                int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
                for (int lm = 0; lm < lmmax__; lm++)
                {
                    int l = l_by_lm_[lm];
                    zm(lm, igloc) = fourpi * parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia) * 
                                    zilm_[lm] * conj(parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc)) * 
                                    fl__(l, iat, parameters_.reciprocal_lattice()->gvec_shell(ig));
                }
            }
            blas<cpu>::gemv(0, lmmax__, ngv_loc, complex_one, zm.ptr(), zm.ld(), 
                            &fpw__[parameters_.reciprocal_lattice()->spl_num_gvec().global_offset()], 1, complex_zero, 
                            &flm__(0, ia), 1);
        }
    }
    
    Platform::allreduce(&flm__(0, 0), (int)flm__.size());
}

void Potential::poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt, mdarray<double_complex, 2>& qit, double_complex* rho_pw)
{
    Timer t("sirius::Potential::poisson_pw");
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

    Platform::allgather(&pseudo_pw[0], (int)parameters_.reciprocal_lattice()->spl_num_gvec().global_offset(), 
                        (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size());
        
    // add pseudo_density to interstitial charge density; now rho(G) has the correct multipole moments in the muffin-tins
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++) rho_pw[ig] += pseudo_pw[ig];
}

template<> void Potential::add_mt_contribution_to_pw<cpu>()
{
    Timer t("sirius::Potential::add_mt_contribution_to_pw");

    mdarray<double_complex, 1> fpw(parameters_.reciprocal_lattice()->num_gvec());
    fpw.zero();

    mdarray<Spline<double>*, 2> svlm(parameters_.lmmax_pot(), parameters_.unit_cell()->num_atoms());
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
        {
            svlm(lm, ia) = new Spline<double>(parameters_.unit_cell()->atom(ia)->type()->radial_grid());
            
            for (int ir = 0; ir < parameters_.unit_cell()->atom(ia)->num_mt_points(); ir++)
                (*svlm(lm, ia))[ir] = effective_potential_->f_mt<global>(lm, ir, ia);
            
            svlm(lm, ia)->interpolate();
        }
    }
   
    #pragma omp parallel default(shared)
    {
        mdarray<double, 1> vjlm(parameters_.lmmax_pot());

        sbessel_pw<double> jl(parameters_.unit_cell(), parameters_.lmax_pot());
        
        #pragma omp for
        for (int igloc = 0; igloc < (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
        {
            int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);

            jl.interpolate(parameters_.reciprocal_lattice()->gvec_len(ig));

            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                int iat = parameters_.unit_cell()->atom(ia)->type_id();

                for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
                {
                    int l = l_by_lm_[lm];
                    vjlm(lm) = Spline<double>::integrate(jl(l, iat), svlm(lm, ia), 2);
                }

                double_complex zt(0, 0);
                for (int l = 0; l <= parameters_.lmax_pot(); l++)
                {
                    for (int m = -l; m <= l; m++)
                    {
                        if (m == 0)
                        {
                            zt += conj(zil_[l]) * parameters_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
                                  vjlm(Utils::lm_by_l_m(l, m));

                        }
                        else
                        {
                            zt += conj(zil_[l]) * parameters_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
                                  (SHT::ylm_dot_rlm(l, m, m) * vjlm(Utils::lm_by_l_m(l, m)) + 
                                   SHT::ylm_dot_rlm(l, m, -m) * vjlm(Utils::lm_by_l_m(l, -m)));
                        }
                    }
                }
                fpw(ig) += zt * fourpi * conj(parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia)) / parameters_.unit_cell()->omega();
            }
        }
    }
    Platform::allreduce(fpw.ptr(), (int)fpw.size());
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++) delete svlm(lm, ia);
    }
}

//== #ifdef _GPU_
//== template <> void Potential::add_mt_contribution_to_pw<gpu>()
//== {
//==     // TODO: couple of things to consider: 1) global array jvlm with G-vector shells may be large; 
//==     //                                     2) MPI reduction over thousands of shell may be slow
//==     Timer t("sirius::Potential::add_mt_contribution_to_pw");
//== 
//==     mdarray<double_complex, 1> fpw(parameters_.num_gvec());
//==     fpw.zero();
//==     
//==     mdarray<int, 1> kargs(4);
//==     kargs(0) = parameters_.num_atom_types();
//==     kargs(1) = parameters_.max_num_mt_points();
//==     kargs(2) = parameters_.lmax_pot();
//==     kargs(3) = parameters_.lmmax_pot();
//==     kargs.allocate_on_device();
//==     kargs.copy_to_device();
//== 
//==     mdarray<double, 3> vlm_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmmax_pot(), 
//==                                  parameters_.num_atoms());
//==     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     {
//==         for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//==         {
//==             Spline<double> s(parameters_.atom(ia)->num_mt_points(), 
//==                              parameters_.atom(ia)->type()->radial_grid());
//==             
//==             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==                 s[ir] = effective_potential_->f_rlm(lm, ir, ia);
//==             
//==             s.interpolate();
//==             s.get_coefs(&vlm_coefs(0, lm, ia), parameters_.max_num_mt_points());
//==         }
//==     }
//==     vlm_coefs.allocate_on_device();
//==     vlm_coefs.copy_to_device();
//== 
//==     mdarray<int, 1> iat_by_ia(parameters_.num_atoms());
//==     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==         iat_by_ia(ia) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//==     iat_by_ia.allocate_on_device();
//==     iat_by_ia.copy_to_device();
//== 
//==     l_by_lm_.allocate_on_device();
//==     l_by_lm_.copy_to_device();
//==     
//==     //=============
//==     // radial grids
//==     //=============
//==     mdarray<double, 2> r_dr(parameters_.max_num_mt_points() * 2, parameters_.num_atom_types());
//==     mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
//==     for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//==     {
//==         nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
//==         parameters_.atom_type(iat)->radial_grid().get_r_dr(&r_dr(0, iat), parameters_.max_num_mt_points());
//==     }
//==     r_dr.allocate_on_device();
//==     r_dr.async_copy_to_device(-1);
//==     nmtp_by_iat.allocate_on_device();
//==     nmtp_by_iat.async_copy_to_device(-1);
//== 
//==     splindex<block> spl_num_gvec_shells(parameters_.num_gvec_shells(), Platform::num_mpi_ranks(), Platform::mpi_rank());
//==     mdarray<double, 3> jvlm(parameters_.lmmax_pot(), parameters_.num_atoms(), parameters_.num_gvec_shells());
//==     jvlm.zero();
//== 
//==     cuda_create_streams(Platform::num_threads());
//==     #pragma omp parallel
//==     {
//==         int thread_id = Platform::thread_id();
//== 
//==         mdarray<double, 3> jl_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmax_pot() + 1, 
//==                                     parameters_.num_atom_types());
//==         
//==         mdarray<double, 2> jvlm_loc(parameters_.lmmax_pot(), parameters_.num_atoms());
//== 
//==         jvlm_loc.pin_memory();
//==         jvlm_loc.allocate_on_device();
//==             
//==         jl_coefs.pin_memory();
//==         jl_coefs.allocate_on_device();
//== 
//==         sbessel_pw<double> jl(parameters_, parameters_.lmax_pot());
//==         
//==         #pragma omp for
//==         for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
//==         {
//==             int igs = spl_num_gvec_shells[igsloc];
//== 
//==             jl.interpolate(parameters_.gvec_shell_len(igs));
//== 
//==             for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//==             {
//==                 for (int l = 0; l <= parameters_.lmax_pot(); l++)
//==                     jl(l, iat)->get_coefs(&jl_coefs(0, l, iat), parameters_.max_num_mt_points());
//==             }
//==             jl_coefs.async_copy_to_device(thread_id);
//== 
//==             sbessel_vlm_inner_product_gpu(kargs.ptr_device(), parameters_.lmmax_pot(), parameters_.num_atoms(), 
//==                                           iat_by_ia.ptr_device(), l_by_lm_.ptr_device(), 
//==                                           nmtp_by_iat.ptr_device(), r_dr.ptr_device(), 
//==                                           jl_coefs.ptr_device(), vlm_coefs.ptr_device(), jvlm_loc.ptr_device(), 
//==                                           thread_id);
//== 
//==             jvlm_loc.async_copy_to_host(thread_id);
//==             
//==             cuda_stream_synchronize(thread_id);
//== 
//==             memcpy(&jvlm(0, 0, igs), &jvlm_loc(0, 0), parameters_.lmmax_pot() * parameters_.num_atoms() * sizeof(double));
//==         }
//==     }
//==     cuda_destroy_streams(Platform::num_threads());
//==     
//==     for (int igs = 0; igs < parameters_.num_gvec_shells(); igs++)
//==         Platform::allreduce(&jvlm(0, 0, igs), parameters_.lmmax_pot() * parameters_.num_atoms());
//== 
//==     #pragma omp parallel for default(shared)
//==     for (int igloc = 0; igloc < parameters_.spl_num_gvec().local_size(); igloc++)
//==     {
//==         int ig = parameters_.spl_num_gvec(igloc);
//==         int igs = parameters_.gvec_shell<local>(igloc);
//== 
//==         for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==         {
//==             double_complex zt(0, 0);
//==             for (int l = 0; l <= parameters_.lmax_pot(); l++)
//==             {
//==                 for (int m = -l; m <= l; m++)
//==                 {
//==                     if (m == 0)
//==                     {
//==                         zt += conj(zil_[l]) * parameters_.gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//==                               jvlm(Utils::lm_by_l_m(l, m), ia, igs);
//== 
//==                     }
//==                     else
//==                     {
//==                         zt += conj(zil_[l]) * parameters_.gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//==                               (SHT::ylm_dot_rlm(l, m, m) * jvlm(Utils::lm_by_l_m(l, m), ia, igs) + 
//==                                SHT::ylm_dot_rlm(l, m, -m) * jvlm(Utils::lm_by_l_m(l, -m), ia, igs));
//==                     }
//==                 }
//==             }
//==             fpw(ig) += zt * fourpi * conj(parameters_.gvec_phase_factor<local>(igloc, ia)) / parameters_.omega();
//==         }
//==     }
//== 
//==     Platform::allreduce(fpw.ptr(), (int)fpw.size());
//==     for (int ig = 0; ig < parameters_.num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
//== 
//==     l_by_lm_.deallocate_on_device();
//== }
//== #endif

void Potential::generate_pw_coefs()
{
    for (int ir = 0; ir < fft_->size(); ir++)
        fft_->buffer(ir) = effective_potential()->f_it<global>(ir) * parameters_.step_function(ir);
    
    fft_->transform(-1);
    fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), 
                 &effective_potential()->f_pw(0));

    if (!use_second_variation) // for full diagonalization we also need Beff(G)
    {
        for (int i = 0; i < parameters_.num_mag_dims(); i++)
        {
            for (int ir = 0; ir < fft_->size(); ir++)
                fft_->buffer(ir) = effective_magnetic_field(i)->f_it<global>(ir) * parameters_.step_function(ir);
    
            fft_->transform(-1);
            fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), 
                         &effective_magnetic_field(i)->f_pw(0));
        }
    }

    if (parameters_.esm_type() == full_potential_pwlo) 
    {
        switch (parameters_.processing_unit())
        {
            case cpu:
            {
                add_mt_contribution_to_pw<cpu>();
                break;
            }
            #ifdef _GPU_
            //== case gpu:
            //== {
            //==     add_mt_contribution_to_pw<gpu>();
            //==     break;
            //== }
            #endif
            default:
            {
                error_local(__FILE__, __LINE__, "wrong processing unit");
            }
        }
    }
}

//void Potential::check_potential_continuity_at_mt()
//{
//    // generate plane-wave coefficients of the potential in the interstitial region
//    parameters_.fft().input(&effective_potential_->f_it<global>(0));
//    parameters_.fft().transform(-1);
//    parameters_.fft().output(parameters_.num_gvec(), parameters_.fft_index(), &effective_potential_->f_pw(0));
//    
//    SHT sht(parameters_.lmax_pot());
//
//    double diff = 0.0;
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int itp = 0; itp < sht.num_points(); itp++)
//        {
//            double vc[3];
//            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * parameters_.atom(ia)->mt_radius();
//
//            double val_it = 0.0;
//            for (int ig = 0; ig < parameters_.num_gvec(); ig++) 
//            {
//                double vgc[3];
//                parameters_.get_coordinates<cartesian, reciprocal>(parameters_.gvec(ig), vgc);
//                val_it += real(effective_potential_->f_pw(ig) * exp(double_complex(0.0, Utils::scalar_product(vc, vgc))));
//            }
//
//            double val_mt = 0.0;
//            for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//                val_mt += effective_potential_->f_rlm(lm, parameters_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
//
//            diff += fabs(val_it - val_mt);
//        }
//    }
//    printf("Total and average potential difference at MT boundary : %.12f %.12f\n", diff, diff / parameters_.num_atoms() / sht.num_points());
//}

void Potential::poisson(Periodic_function<double>* rho, Periodic_function<double>* vh)
{
    Timer t("sirius::Potential::poisson");

    /* get plane-wave coefficients of the charge density */
    fft_->input(&rho->f_it<global>(0));
    fft_->transform(-1);
    fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho->f_pw(0));

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
        Platform::allgather(vh_el_.ptr(), (int)parameters_.unit_cell()->spl_num_atoms().global_offset(),
                            (int)parameters_.unit_cell()->spl_num_atoms().local_size());

    }
    
    /* transform Hartree potential to real space */
    fft_->input(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &vh->f_pw(0));
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
        Platform::allreduce(&evha_nuc_, 1);
        energy_vha_ += evha_nuc_;
    }
}

void Potential::xc_mt_nonmagnetic(Radial_grid& rgrid,
                                  std::vector<XC_functional*>& xc_func,
                                  Spheric_function<spectral, double>& rho_lm, 
                                  Spheric_function<spatial, double>& rho_tp, 
                                  Spheric_function<spatial, double>& vxc_tp, 
                                  Spheric_function<spatial, double>& exc_tp)
{
    Timer t("sirius::Potential::xc_mt_nonmagnetic");

    bool is_gga = false;
    for (auto& ixc: xc_func) if (ixc->gga()) is_gga = true;

    Spheric_function_gradient<spatial, double> grad_rho_tp;
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
        Spheric_function_gradient<spatial, double> grad_vsigma_tp;
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

void Potential::xc_mt_magnetic(Radial_grid& rgrid,
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

    Spheric_function_gradient<spatial, double> grad_rho_up_tp;
    Spheric_function_gradient<spatial, double> grad_rho_dn_tp;

    Spheric_function<spatial, double> lapl_rho_up_tp;
    Spheric_function<spatial, double> lapl_rho_dn_tp;

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
        Spheric_function_gradient<spatial, double> grad_vsigma_uu_tp;
        Spheric_function_gradient<spatial, double> grad_vsigma_ud_tp;
        Spheric_function_gradient<spatial, double> grad_vsigma_dd_tp;
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

    auto uc = parameters_.unit_cell();

    for (int ialoc = 0; ialoc < (int)uc->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = uc->spl_num_atoms(ialoc);
        auto& rgrid = uc->atom(ia)->radial_grid();
        int nmtp = uc->atom(ia)->num_mt_points();

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
                    mag = sqrt(mag);

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

    int num_loc_points = fft_->local_size();
    
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
    
    auto rl = parameters_.reciprocal_lattice();

    if (is_gga) 
    {
        Smooth_periodic_function<spatial, double> rho_it(&rho->f_it<global>(0), rl);

        /* get plane-wave coefficients of the density */
        Smooth_periodic_function<spectral> rho_pw = transform(rho_it);

        /* generate pw coeffs of the gradient and laplacian */
        auto grad_rho_pw = gradient(rho_pw);
        auto lapl_rho_pw = laplacian(rho_pw);

        /* gradient in real space */
        for (int x = 0; x < 3; x++) grad_rho_it[x] = transform<double, local>(grad_rho_pw[x]);

        /* product of gradients */
        grad_rho_grad_rho_it = grad_rho_it * grad_rho_it;
        
        /* Laplacian in real space */
        lapl_rho_it = transform<double, local>(lapl_rho_pw);
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
        Smooth_periodic_function<spatial, double> vsigma_it(rl);
        Platform::allgather(&vsigma_tmp(0), &vsigma_it(0), fft_->global_offset(), fft_->local_size());

        /* forward transform vsigma to plane-wave domain */
        Smooth_periodic_function<spectral> vsigma_pw = transform(vsigma_it);
        
        /* gradient of vsigma in plane-wave domain */
        auto grad_vsigma_pw = gradient(vsigma_pw);

        /* backward transform gradient from pw to real space */
        Smooth_periodic_function_gradient<spatial, double> grad_vsigma_it;
        for (int x = 0; x < 3; x++) grad_vsigma_it[x] = transform<double, local>(grad_vsigma_pw[x]);

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
    
    auto rl = parameters_.reciprocal_lattice();

    int num_loc_points = fft_->local_size();
    
    Smooth_periodic_function<spatial, double> rho_up_it(rl);
    Smooth_periodic_function<spatial, double> rho_dn_it(rl);

    /* compute "up" and "dn" components and also check for negative values of density */
    double rhomin = 0.0;

    for (int ir = 0; ir < fft_->size(); ir++)
    {
        double mag = 0.0;
        for (int j = 0; j < parameters_.num_mag_dims(); j++) mag += pow(magnetization[j]->f_it<global>(ir), 2);
        mag = sqrt(mag);

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
            grad_rho_up_it[x] = transform<double, local>(grad_rho_up_pw[x]);
            grad_rho_dn_it[x] = transform<double, local>(grad_rho_dn_pw[x]);
        }

        /* product of gradients */
        grad_rho_up_grad_rho_up_it = grad_rho_up_it * grad_rho_up_it;
        grad_rho_up_grad_rho_dn_it = grad_rho_up_it * grad_rho_dn_it;
        grad_rho_dn_grad_rho_dn_it = grad_rho_dn_it * grad_rho_dn_it;
        
        /* Laplacian in real space */
        lapl_rho_up_it = transform<double, local>(lapl_rho_up_pw);
        lapl_rho_dn_it = transform<double, local>(lapl_rho_dn_pw);
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
                             &rho_up_it(fft_->global_offset() + spl_t.global_offset()), 
                             &rho_dn_it(fft_->global_offset() + spl_t.global_offset()), 
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
                             &rho_up_it(fft_->global_offset() + spl_t.global_offset()), 
                             &rho_dn_it(fft_->global_offset() + spl_t.global_offset()), 
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
        Smooth_periodic_function<spatial, double> vsigma_uu_it(rl);
        Smooth_periodic_function<spatial, double> vsigma_ud_it(rl);
        Smooth_periodic_function<spatial, double> vsigma_dd_it(rl);
        Platform::allgather(&vsigma_uu_tmp(0), &vsigma_uu_it(0), fft_->global_offset(), fft_->local_size());
        Platform::allgather(&vsigma_ud_tmp(0), &vsigma_ud_it(0), fft_->global_offset(), fft_->local_size());
        Platform::allgather(&vsigma_dd_tmp(0), &vsigma_dd_it(0), fft_->global_offset(), fft_->local_size());

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
            grad_vsigma_uu_it[x] = transform<double, local>(grad_vsigma_uu_pw[x]);
            grad_vsigma_ud_it[x] = transform<double, local>(grad_vsigma_ud_pw[x]);
            grad_vsigma_dd_it[x] = transform<double, local>(grad_vsigma_dd_pw[x]);
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
        double m = rho_up_it(fft_->global_offset() + irloc) - rho_dn_it(fft_->global_offset() + irloc);

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
    Timer t("sirius::Potential::xc");

    if (parameters_.xc_functionals_input_section_.xc_functional_names_.size() == 0)
    {
        vxc->zero();
        exc->zero();
        for (int i = 0; i < parameters_.num_mag_dims(); i++) bxc[i]->zero();
        return;
    }

    /* create list of XC functionals */
    std::vector<XC_functional*> xc_func;
    for (int i = 0; i < (int)parameters_.xc_functionals_input_section_.xc_functional_names_.size(); i++)
    {
        std::string xc_label = parameters_.xc_functionals_input_section_.xc_functional_names_[i];
        xc_func.push_back(new XC_functional(xc_label, parameters_.num_spins()));
    }
   
    if (parameters_.unit_cell()->full_potential()) xc_mt(rho, magnetization, xc_func, vxc, bxc, exc);
    
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

void Potential::generate_effective_potential(Periodic_function<double>* rho, 
                                             Periodic_function<double>* magnetization[3])
{
    Timer t("sirius::Potential::generate_effective_potential");
    
    /* zero effective potential and magnetic field */
    zero();

    /* solve Poisson equation */
    poisson(rho, hartree_potential_);

    /* add Hartree potential to the total potential */
    effective_potential_->add(hartree_potential_);

    xc(rho, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
   
    effective_potential_->add(xc_potential_);

    effective_potential_->sync(true, true);
    for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->sync(true, true);

    //if (debug_level > 1) check_potential_continuity_at_mt();
}

void Potential::generate_effective_potential(Periodic_function<double>* rho, 
                                             Periodic_function<double>* rho_core, 
                                             Periodic_function<double>* magnetization[3])
{
    Timer t("sirius::Potential::generate_effective_potential");
    
    /* zero effective potential and magnetic field */
    zero();

    /* solve Poisson equation with valence density */
    poisson(rho, hartree_potential_);

    /* add Hartree potential to the effective potential */
    effective_potential_->add(hartree_potential_);

    /* create temporary function for rho + rho_core */
    Periodic_function<double>* rhovc = new Periodic_function<double>(parameters_, 0);
    rhovc->allocate(false, true);
    rhovc->zero();
    rhovc->add(rho);
    rhovc->add(rho_core);
    rhovc->sync(false, true);

    /* construct XC potentials from rho + rho_core */
    xc(rhovc, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
    
    /* destroy temporary function */
    delete rhovc;
    
    // add XC potential to the effective potential
    effective_potential_->add(xc_potential_);
    
    // add local ionic potential to the effective potential
    effective_potential_->add(local_potential_);

    // synchronize effective potential
    effective_potential_->sync(false, true);
    for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->sync(false, true);
}

void Potential::generate_d_mtrx()
{   
    Timer t("sirius::Potential::generate_d_mtrx");

    auto rl = parameters_.reciprocal_lattice();

    // get plane-wave coefficients of effective potential
    fft_->input(&effective_potential_->f_it<global>(0));
    fft_->transform(-1);
    fft_->output(rl->num_gvec(), rl->fft_index(), &effective_potential_->f_pw(0));

    #pragma omp parallel
    {
        mdarray<double_complex, 1> veff_tmp(rl->spl_num_gvec().local_size());
        mdarray<double_complex, 1> dm_packed(parameters_.unit_cell()->max_mt_basis_size() * 
                                             (parameters_.unit_cell()->max_mt_basis_size() + 1) / 2);
        
        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            auto atom_type = parameters_.unit_cell()->atom(ia)->type();
            int nbf = atom_type->mt_basis_size();
            
            for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
            {
                int ig = rl->spl_num_gvec(igloc);
                veff_tmp(igloc) = effective_potential_->f_pw(ig) * rl->gvec_phase_factor<local>(igloc, ia);
            }

            blas<cpu>::gemv(2, (int)rl->spl_num_gvec().local_size(), nbf * (nbf + 1) / 2, complex_one, 
                            &atom_type->uspp().q_pw(0, 0), (int)rl->spl_num_gvec().local_size(),  
                            &veff_tmp(0), 1, complex_zero, &dm_packed(0), 1);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 <= xi2; xi1++)
                {
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;

                    parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2) = dm_packed(idx12) * parameters_.unit_cell()->omega();
                    parameters_.unit_cell()->atom(ia)->d_mtrx(xi2, xi1) = conj(dm_packed(idx12)) * parameters_.unit_cell()->omega();
                }
            }
        }
    }

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        Platform::allreduce(parameters_.unit_cell()->atom(ia)->d_mtrx().ptr(),
                            (int)parameters_.unit_cell()->atom(ia)->d_mtrx().size());

        auto atom_type = parameters_.unit_cell()->atom(ia)->type();
        int nbf = atom_type->mt_basis_size();

        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = atom_type->indexb(xi2).lm;
            int idxrf2 = atom_type->indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                int lm1 = atom_type->indexb(xi1).lm;
                int idxrf1 = atom_type->indexb(xi1).idxrf;
                if (lm1 == lm2) parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2) += atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
            }
        }
    }
}

#ifdef _GPU_

extern "C" void compute_d_mtrx_valence_gpu(int num_gvec_loc,
                                           int num_elements,
                                           void* veff, 
                                           int* gvec, 
                                           double ax,
                                           double ay,
                                           double az,
                                           void* vtmp,
                                           void* q_pw_t,
                                           void* d_mtrx,
                                           int stream_id);
void Potential::generate_d_mtrx_gpu()
{   
    Timer t("sirius::Potential::generate_d_mtrx_gpu");

    // get plane-wave coefficients of effective potential
    fft_->input(&effective_potential_->f_it<global>(0));
    fft_->transform(-1);
    fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), 
                 &effective_potential_->f_pw(0));

    auto rl = parameters_.reciprocal_lattice();

    mdarray<double_complex, 1> veff_gpu(&effective_potential_->f_pw(static_cast<int>(rl->spl_num_gvec().global_offset())), 
                                        static_cast<int>(rl->spl_num_gvec().local_size()));
    veff_gpu.allocate_on_device();
    veff_gpu.copy_to_device();

    for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
    {
         auto type = parameters_.unit_cell()->atom_type(iat);
         type->uspp().q_pw.allocate_on_device();
         type->uspp().q_pw.copy_to_device();
    }
    
    mdarray<int, 2> gvec(3, rl->spl_num_gvec().local_size());
    for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
    {
        for (int x = 0; x < 3; x++) gvec(x, igloc) = rl->gvec(rl->spl_num_gvec(igloc))[x];
    }
    gvec.allocate_on_device();
    gvec.copy_to_device();

    #pragma omp parallel
    {
        mdarray<double_complex, 1> vtmp_gpu(NULL, rl->spl_num_gvec().local_size());
        vtmp_gpu.allocate_on_device();

        mdarray<double_complex, 1> d_mtrx_gpu(parameters_.unit_cell()->max_mt_basis_size() * 
                                         (parameters_.unit_cell()->max_mt_basis_size() + 1) / 2);
        d_mtrx_gpu.allocate_on_device();
        d_mtrx_gpu.pin_memory();

        int thread_id = Platform::thread_id();
        
        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            auto atom_type = parameters_.unit_cell()->atom(ia)->type();
            int nbf = atom_type->mt_basis_size();

            vector3d<double> apos = parameters_.unit_cell()->atom(ia)->position();

            compute_d_mtrx_valence_gpu((int)rl->spl_num_gvec().local_size(), 
                                       nbf * (nbf + 1) / 2, 
                                       veff_gpu.ptr_device(), 
                                       gvec.ptr_device(), 
                                       apos[0], 
                                       apos[1], 
                                       apos[2], 
                                       vtmp_gpu.ptr_device(),
                                       atom_type->uspp().q_pw.ptr_device(),
                                       d_mtrx_gpu.ptr_device(), 
                                       thread_id);
                                       
            d_mtrx_gpu.async_copy_to_host(thread_id);

            cuda_stream_synchronize(thread_id);

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 <= xi2; xi1++)
                {
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;

                    parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2) = d_mtrx_gpu(idx12) * parameters_.unit_cell()->omega();
                    parameters_.unit_cell()->atom(ia)->d_mtrx(xi2, xi1) = conj(d_mtrx_gpu(idx12)) * parameters_.unit_cell()->omega();
                }
            }
        }
    }

    for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
    {
         auto type = parameters_.unit_cell()->atom_type(iat);
         type->uspp().q_pw.deallocate_on_device();
    }

    // TODO: this is common with cpu code
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        Platform::allreduce(parameters_.unit_cell()->atom(ia)->d_mtrx().ptr(),
                            (int)parameters_.unit_cell()->atom(ia)->d_mtrx().size());

        auto atom_type = parameters_.unit_cell()->atom(ia)->type();
        int nbf = atom_type->mt_basis_size();

        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = atom_type->indexb(xi2).lm;
            int idxrf2 = atom_type->indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                int lm1 = atom_type->indexb(xi1).lm;
                int idxrf1 = atom_type->indexb(xi1).idxrf;
                if (lm1 == lm2) parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2) += atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
            }
        }
    }
}
#endif

void Potential::set_effective_potential_ptr(double* veffmt, double* veffit)
{
    effective_potential_->set_mt_ptr(veffmt);
    effective_potential_->set_it_ptr(veffit);
}

//void Potential::copy_to_global_ptr(double* fmt__, double* fit__, Periodic_function<double>* src)
//{
//    stop_here // fix thsi
//    //Periodic_function<double>* dest = new Periodic_function<double>(parameters_, parameters_.lmmax_pot());
//    //dest->set_mt_ptr(fmt);
//    //dest->set_it_ptr(fit);
//    //dest->copy(src);
//    //dest->sync(true, true);
//    //delete dest;
//}


//** void Potential::copy_xc_potential(double* vxcmt, double* vxcit)
//** {
//**     // create temporary function
//**     Periodic_function<double>* vxc = 
//**         new Periodic_function<double>(parameters_, Argument(arg_lm, parameters_.lmmax_pot()),
//**                                                    Argument(arg_radial, parameters_.max_num_mt_points()));
//**     // set global pointers
//**     vxc->set_mt_ptr(vxcmt);
//**     vxc->set_it_ptr(vxcit);
//**     
//**     // xc_potential is local, vxc is global so we can sync vxc
//**     vxc->copy(xc_potential_);
//**     vxc->sync();
//** 
//**     delete vxc;
//** }
//** 
//** void Potential::copy_effective_magnetic_field(double* beffmt, double* beffit)
//** {
//**     if (parameters_.num_mag_dims() == 0) return;
//**     assert(parameters_.num_spins() == 2);
//**     
//**     // set temporary array wrapper
//**     mdarray<double,4> beffmt_tmp(beffmt, parameters_.lmmax_pot(), parameters_.max_num_mt_points(), 
//**                                  parameters_.num_atoms(), parameters_.num_mag_dims());
//**     mdarray<double,2> beffit_tmp(beffit, parameters_.fft().size(), parameters_.num_mag_dims());
//**     
//**     Periodic_function<double>* bxc[3];
//**     for (int i = 0; i < parameters_.num_mag_dims(); i++)
//**     {
//**         bxc[i] = new Periodic_function<double>(parameters_, Argument(arg_lm, parameters_.lmmax_pot()),
//**                                                             Argument(arg_radial, parameters_.max_num_mt_points()));
//**     }
//** 
//**     if (parameters_.num_mag_dims() == 1)
//**     {
//**         // z
//**         bxc[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
//**         bxc[0]->set_it_ptr(&beffit_tmp(0, 0));
//**     }
//**     
//**     if (parameters_.num_mag_dims() == 3)
//**     {
//**         // z
//**         bxc[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 2));
//**         bxc[0]->set_it_ptr(&beffit_tmp(0, 2));
//**         // x
//**         bxc[1]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
//**         bxc[1]->set_it_ptr(&beffit_tmp(0, 0));
//**         // y
//**         bxc[2]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 1));
//**         bxc[2]->set_it_ptr(&beffit_tmp(0, 1));
//**     }
//** 
//**     for (int i = 0; i < parameters_.num_mag_dims(); i++)
//**     {
//**         bxc[i]->copy(effective_magnetic_field_[i]);
//**         bxc[i]->sync();
//**         delete bxc[i];
//**     }
//** }

void Potential::set_effective_magnetic_field_ptr(double* beffmt, double* beffit)
{
    if (parameters_.num_mag_dims() == 0) return;
    assert(parameters_.num_spins() == 2);
    
    // set temporary array wrapper
    mdarray<double,4> beffmt_tmp(beffmt, parameters_.lmmax_pot(), parameters_.unit_cell()->max_num_mt_points(), 
                                 parameters_.unit_cell()->num_atoms(), parameters_.num_mag_dims());
    mdarray<double,2> beffit_tmp(beffit, fft_->size(), parameters_.num_mag_dims());
    
    if (parameters_.num_mag_dims() == 1)
    {
        // z
        effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
        effective_magnetic_field_[0]->set_it_ptr(&beffit_tmp(0, 0));
    }
    
    if (parameters_.num_mag_dims() == 3)
    {
        // z
        effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 2));
        effective_magnetic_field_[0]->set_it_ptr(&beffit_tmp(0, 2));
        // x
        effective_magnetic_field_[1]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
        effective_magnetic_field_[1]->set_it_ptr(&beffit_tmp(0, 0));
        // y
        effective_magnetic_field_[2]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 1));
        effective_magnetic_field_[2]->set_it_ptr(&beffit_tmp(0, 1));
    }
}
         
void Potential::zero()
{
    effective_potential_->zero();
    for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->zero();
}

void Potential::update_atomic_potential()
{
    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
       int ia = parameters_.unit_cell()->atom_symmetry_class(ic)->atom_id(0);
       int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();
       
       std::vector<double> veff(nmtp);
       
       for (int ir = 0; ir < nmtp; ir++) veff[ir] = y00 * effective_potential_->f_mt<global>(0, ir, ia);

       parameters_.unit_cell()->atom_symmetry_class(ic)->set_spherical_potential(veff);
    }
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        double* veff = &effective_potential_->f_mt<global>(0, 0, ia);
        
        double* beff[] = {nullptr, nullptr, nullptr};
        for (int i = 0; i < parameters_.num_mag_dims(); i++) beff[i] = &effective_magnetic_field_[i]->f_mt<global>(0, 0, ia);
        
        parameters_.unit_cell()->atom(ia)->set_nonspherical_potential(veff, beff);
    }
}

void Potential::save()
{
    if (Platform::mpi_rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);

        effective_potential_->hdf5_write(fout["effective_potential"]);

        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            effective_magnetic_field_[j]->hdf5_write(fout["effective_magnetic_field"].create_node(j));

        //== fout["effective_potential"].create_node("free_atom_potential");
        //== for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        //== {
        //==     fout["effective_potential"]["free_atom_potential"].write(iat, parameters_.unit_cell()->atom_type(iat)->free_atom_potential());
        //== }
    }
    Platform::barrier();
}

void Potential::load()
{
    HDF5_tree fout(storage_file_name, false);
    
    effective_potential_->hdf5_read(fout["effective_potential"]);

    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        effective_magnetic_field_[j]->hdf5_read(fout["effective_magnetic_field"][j]);
    
    //== for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
    //==     fout["effective_potential"]["free_atom_potential"].read(iat, parameters_.unit_cell()->atom_type(iat)->free_atom_potential());

    if (parameters_.unit_cell()->full_potential()) update_atomic_potential();
}

void Potential::generate_local_potential()
{
    Timer t("sirius::Potential::generate_local_potential");

    auto rl = parameters_.reciprocal_lattice();
    auto uc = parameters_.unit_cell();

    mdarray<double, 2> vloc_radial_integrals(uc->num_atom_types(), rl->num_gvec_shells_inner());
    
    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
        auto atom_type = uc->atom_type(iat);
        #pragma omp parallel
        {
            Spline<double> s(atom_type->radial_grid());
            #pragma omp for
            for (int igs = 0; igs < rl->num_gvec_shells_inner(); igs++)
            {
                if (igs == 0)
                {
                    for (int ir = 0; ir < s.num_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        s[ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn()) * x;
                    }
                    vloc_radial_integrals(iat, igs) = s.interpolate().integrate(0);
                }
                else
                {
                    double g = rl->gvec_shell_len(igs);
                    double g2 = pow(g, 2);
                    for (int ir = 0; ir < s.num_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        s[ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn() * gsl_sf_erf(x)) * sin(g * x);
                    }
                    vloc_radial_integrals(iat, igs) = (s.interpolate().integrate(0) / g - atom_type->zn() * exp(-g2 / 4) / g2);
                }
            }
         }
    }

    std::vector<double_complex> v = rl->make_periodic_function(vloc_radial_integrals, rl->num_gvec());
    
    fft_->input(rl->num_gvec(), rl->fft_index(), &v[0]); 
    fft_->transform(1);
    fft_->output(&local_potential_->f_it<global>(0));
}

}

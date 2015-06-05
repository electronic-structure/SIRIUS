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
//#include "smooth_periodic_function.h"

namespace sirius {

// TODO: everything here must be documented
// TODO: better naming convention: q is meaningless

Potential::Potential(Simulation_context& ctx__)
    : ctx_(ctx__),
      parameters_(ctx__.parameters()),
      unit_cell_(ctx__.unit_cell()),
      comm_(ctx__.comm()),
      fft_(ctx__.fft()),
      pseudo_density_order(9)
{
    Timer t("sirius::Potential::Potential");
    
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            lmax_ = std::max(parameters_.lmax_rho(), parameters_.lmax_pot());
            sht_ = new SHT(lmax_);
            break;
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            lmax_ = parameters_.lmax_beta() * 2;
            break;
        }
        default:
        {
            STOP();
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

    //int ngv = (use_second_variation) ? 0 : parameters_.reciprocal_lattice()->num_gvec();

    effective_potential_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot());

    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        effective_magnetic_field_[j] = new Periodic_function<double>(ctx_, parameters_.lmmax_pot(), false);
    
    hartree_potential_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot());
    hartree_potential_->allocate(false, true);
    
    xc_potential_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot(), false);
    xc_potential_->allocate(false, false);
    
    xc_energy_density_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot(), false);
    xc_energy_density_->allocate(false, false);

    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        local_potential_ = new Periodic_function<double>(ctx_, 0);
        local_potential_->allocate(false, true);
        local_potential_->zero();

        generate_local_potential();
    }

    vh_el_ = mdarray<double, 1>(unit_cell_.num_atoms());

    init();

    //mixer_ = new Linear_mixer<double>(size(), parameters_.iip_.mixer_input_section_.beta_, parameters_.comm());
    std::vector<double> weights;
    mixer_ = new Broyden_modified_mixer<double>(size(),
                                       parameters_.mixer_input_section().max_history_,
                                       parameters_.mixer_input_section().beta_,
                                       weights,
                                       comm_);
}

Potential::~Potential()
{
    delete effective_potential_; 
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete effective_magnetic_field_[j];
    if (parameters_.esm_type() == full_potential_lapwlo) delete sht_;
    delete hartree_potential_;
    delete xc_potential_;
    delete xc_energy_density_;
    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential) delete local_potential_;
    delete mixer_;
}

void Potential::init()
{
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        auto rl = ctx_.reciprocal_lattice();
        /* compute values of spherical Bessel functions at MT boundary */
        sbessel_mt_ = mdarray<double, 3>(lmax_ + pseudo_density_order + 2, unit_cell_.num_atom_types(), 
                                         rl->num_gvec_shells_inner());
        sbessel_mt_.allocate();

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            for (int igs = 0; igs < rl->num_gvec_shells_inner(); igs++)
            {
                gsl_sf_bessel_jl_array(lmax_ + pseudo_density_order + 1, 
                                       rl->gvec_shell_len(igs) * unit_cell_.atom_type(iat)->mt_radius(), 
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
        sbessel_mom_ = mdarray<double, 3>(parameters_.lmax_rho() + 1, unit_cell_.num_atom_types(), 
                                          rl->num_gvec_shells_inner());
        sbessel_mom_.allocate();
        sbessel_mom_.zero();

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            sbessel_mom_(0, iat, 0) = std::pow(unit_cell_.atom_type(iat)->mt_radius(), 3) / 3.0; // for |G|=0
            for (int igs = 1; igs < rl->num_gvec_shells_inner(); igs++)
            {
                for (int l = 0; l <= parameters_.lmax_rho(); l++)
                {
                    sbessel_mom_(l, iat, igs) = std::pow(unit_cell_.atom_type(iat)->mt_radius(), l + 2) * 
                                                sbessel_mt_(l + 1, iat, igs) / rl->gvec_shell_len(igs);
                }
            }
        }
        #ifdef __PRINT_OBJECT_HASH
        DUMP("hash(sbessel_mom): %16llX", sbessel_mom_.hash());
        #endif
        
        /* compute Gamma[5/2 + n + l] / Gamma[3/2 + l] / R^l
         *
         * use Gamma[1/2 + p] = (2p - 1)!!/2^p Sqrt[Pi]
         */
        gamma_factors_R_ = mdarray<double, 2>(parameters_.lmax_rho() + 1, unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            for (int l = 0; l <= parameters_.lmax_rho(); l++)
            {
                long double Rl = pow(unit_cell_.atom_type(iat)->mt_radius(), l);

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

//template<> void Potential::add_mt_contribution_to_pw<CPU>()
//{
//    Timer t("sirius::Potential::add_mt_contribution_to_pw");
//
//    mdarray<double_complex, 1> fpw(parameters_.reciprocal_lattice()->num_gvec());
//    fpw.zero();
//
//    mdarray<Spline<double>*, 2> svlm(parameters_.lmmax_pot(), unit_cell_.num_atoms());
//    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//    {
//        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//        {
//            svlm(lm, ia) = new Spline<double>(unit_cell_.atom(ia)->type()->radial_grid());
//            
//            for (int ir = 0; ir < unit_cell_.atom(ia)->num_mt_points(); ir++)
//                (*svlm(lm, ia))[ir] = effective_potential_->f_mt<global>(lm, ir, ia);
//            
//            svlm(lm, ia)->interpolate();
//        }
//    }
//   
//    #pragma omp parallel default(shared)
//    {
//        mdarray<double, 1> vjlm(parameters_.lmmax_pot());
//
//        sbessel_pw<double> jl(parameters_.unit_cell(), parameters_.lmax_pot());
//        
//        #pragma omp for
//        for (int igloc = 0; igloc < (int)parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
//        {
//            int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
//
//            jl.interpolate(parameters_.reciprocal_lattice()->gvec_len(ig));
//
//            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//            {
//                int iat = unit_cell_.atom(ia)->type_id();
//
//                for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//                {
//                    int l = l_by_lm_[lm];
//                    vjlm(lm) = Spline<double>::integrate(jl(l, iat), svlm(lm, ia), 2);
//                }
//
//                double_complex zt(0, 0);
//                for (int l = 0; l <= parameters_.lmax_pot(); l++)
//                {
//                    for (int m = -l; m <= l; m++)
//                    {
//                        if (m == 0)
//                        {
//                            zt += conj(zil_[l]) * parameters_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//                                  vjlm(Utils::lm_by_l_m(l, m));
//
//                        }
//                        else
//                        {
//                            zt += conj(zil_[l]) * parameters_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//                                  (SHT::ylm_dot_rlm(l, m, m) * vjlm(Utils::lm_by_l_m(l, m)) + 
//                                   SHT::ylm_dot_rlm(l, m, -m) * vjlm(Utils::lm_by_l_m(l, -m)));
//                        }
//                    }
//                }
//                fpw(ig) += zt * fourpi * conj(parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia)) / unit_cell_.omega();
//            }
//        }
//    }
//    parameters_.comm().allreduce(fpw.at<CPU>(), (int)fpw.size());
//    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
//    
//    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//    {
//        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++) delete svlm(lm, ia);
//    }
//}

//== #ifdef _GPU_
//== template <> void Potential::add_mt_contribution_to_pw<GPU>()
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
//==     Platform::allreduce(fpw.at<CPU>(), (int)fpw.size());
//==     for (int ig = 0; ig < parameters_.num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
//== 
//==     l_by_lm_.deallocate_on_device();
//== }
//== #endif

void Potential::generate_pw_coefs()
{
    for (int ir = 0; ir < fft_->size(); ir++)
        fft_->buffer(ir) = effective_potential()->f_it<global>(ir) * ctx_.step_function()->theta_r(ir);

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z2 = mdarray<double_complex, 1>(&fft_->buffer(0), fft_->size()).checksum();
    DUMP("checksum(veff_it): %18.10f", mdarray<double, 1>(&effective_potential()->f_it<global>(0) , fft_->size()).checksum());
    //DUMP("checksum(step_function): %18.10f", mdarray<double, 1>(&parameters_.step_function(0), fft_->size()).checksum();
    DUMP("checksum(fft_buffer): %18.10f %18.10f", std::real(z2), std::imag(z2));
    #endif

    fft_->transform(-1);
    fft_->output(fft_->num_gvec(), fft_->index_map(), &effective_potential()->f_pw(0));

    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(veff_it): %18.10f", effective_potential()->f_it().checksum());
    double_complex z1 = mdarray<double_complex, 1>(&effective_potential()->f_pw(0), fft_->num_gvec()).checksum();
    DUMP("checksum(veff_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
    #endif

    if (!use_second_variation) // for full diagonalization we also need Beff(G)
    {
        for (int i = 0; i < parameters_.num_mag_dims(); i++)
        {
            for (int ir = 0; ir < fft_->size(); ir++)
                fft_->buffer(ir) = effective_magnetic_field(i)->f_it<global>(ir) * ctx_.step_function()->theta_r(ir);
    
            fft_->transform(-1);
            fft_->output(fft_->num_gvec(), fft_->index_map(), &effective_magnetic_field(i)->f_pw(0));
        }
    }

    if (parameters_.esm_type() == full_potential_pwlo) 
    {
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                STOP();
                //add_mt_contribution_to_pw<CPU>();
                break;
            }
            #ifdef _GPU_
            //== case GPU:
            //== {
            //==     add_mt_contribution_to_pw<GPU>();
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
    Timer t("sirius::Potential::generate_effective_potential", comm_);

    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        /* zero effective potential and magnetic field */
        zero();

        /* solve Poisson equation with valence density */
        poisson(rho, hartree_potential_);

        /* add Hartree potential to the effective potential */
        effective_potential_->add(hartree_potential_);

        /* create temporary function for rho + rho_core */
        Periodic_function<double>* rhovc = new Periodic_function<double>(ctx_, 0, false);
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

    if (parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        zero();

        auto rl = ctx_.reciprocal_lattice();

        /* create temporary function for rho + rho_core */
        Periodic_function<double>* rhovc = new Periodic_function<double>(ctx_, 0, false);
        rhovc->allocate(false, true);
        rhovc->zero();
        rhovc->add(rho);
        rhovc->add(rho_core);
        //rhovc->sync(false, true);

        /* construct XC potentials from rho + rho_core */
        xc(rhovc, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
        
        /* destroy temporary function */
        delete rhovc;

        /* add XC potential to the effective potential */
        effective_potential_->add(xc_potential_);
        
        /* add local ionic potential to the effective potential */
        effective_potential_->add(local_potential_);
        effective_potential_->sync(false, true);
        
        Timer t1("sirius::Potential::generate_effective_potential|fft");
        fft_->input(&effective_potential_->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(fft_->num_gvec(), fft_->index_map(), &effective_potential_->f_pw(0));

        /* get plane-wave coefficients of the charge density */
        fft_->input(&rho->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(fft_->num_gvec(), fft_->index_map(), &rho->f_pw(0));
        t1.stop();

        std::vector<double_complex> vtmp(rl->spl_num_gvec().local_size());
        
        energy_vha_ = 0.0;
        for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
        {
            int ig = rl->spl_num_gvec(igloc);
            vtmp[igloc] = (ig == 0) ? 0.0 : rho->f_pw(ig) * fourpi / std::pow(ctx_.reciprocal_lattice()->gvec_len(ig), 2);
            energy_vha_ += real(conj(rho->f_pw(ig)) * vtmp[igloc]);
            vtmp[igloc] += effective_potential_->f_pw(ig);
        }
        comm_.allreduce(&energy_vha_, 1);
        energy_vha_ *= unit_cell_.omega();

        auto offsets = rl->spl_num_gvec().offsets();
        auto counts = rl->spl_num_gvec().counts();
        //parameters_.comm().allgather(&vtmp[0], rl->spl_num_gvec().local_size(), &hartree_potential_->f_pw(0), &counts[0], &offsets[0]);
        comm_.allgather(&vtmp[0], (int)rl->spl_num_gvec().local_size(), &effective_potential_->f_pw(0), &counts[0], &offsets[0]);

        //for (int ig = 0; ig < rl->num_gvec(); ig++)
        //    effective_potential_->f_pw(ig) += hartree_potential_->f_pw(ig);


    }
}

#ifdef _GPU_
extern "C" void mul_veff_with_phase_factors_gpu(int num_atoms__,
                                                int num_gvec_loc__,
                                                double_complex const* veff__,
                                                int const* gvec__,
                                                double const* atom_pos__,
                                                double_complex* veff_a__);
#endif

void Potential::generate_d_mtrx()
{   
    Timer t("sirius::Potential::generate_d_mtrx");

    if (parameters_.esm_type() == ultrasoft_pseudopotential)
    {
        auto rl = ctx_.reciprocal_lattice();

        /* get plane-wave coefficients of effective potential */
        fft_->input(&effective_potential_->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(fft_->num_gvec(), fft_->index_map(), &effective_potential_->f_pw(0));

        #ifdef _GPU_
        mdarray<double_complex, 1> veff;
        mdarray<int, 2> gvec;

        if (parameters_.processing_unit() == GPU)
        {
            veff = mdarray<double_complex, 1> (&effective_potential_->f_pw((int)rl->spl_num_gvec().global_offset()), 
                                               rl->spl_num_gvec().local_size());
            veff.allocate_on_device();
            veff.copy_to_device();

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                 auto type = unit_cell_.atom_type(iat);
                 type->uspp().q_pw.allocate_on_device();
                 type->uspp().q_pw.copy_to_device();
            }
        
            gvec = mdarray<int, 2>(3, rl->spl_num_gvec().local_size());
            for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
            {
                for (int x = 0; x < 3; x++) gvec(x, igloc) = rl->gvec(rl->spl_num_gvec(igloc))[x];
            }
            gvec.allocate_on_device();
            gvec.copy_to_device();
        }
        #endif

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            auto atom_type = unit_cell_.atom_type(iat);
            int nbf = atom_type->mt_basis_size();
            matrix<double_complex> d_tmp(nbf * (nbf + 1) / 2, atom_type->num_atoms()); 

            if (parameters_.processing_unit() == CPU)
            {
                matrix<double_complex> veff_a(rl->spl_num_gvec().local_size(), atom_type->num_atoms());

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < atom_type->num_atoms(); i++)
                {
                    int ia = atom_type->atom_id(i);

                    for (int igloc = 0; igloc < (int)rl->spl_num_gvec().local_size(); igloc++)
                    {
                        int ig = rl->spl_num_gvec(igloc);
                        veff_a(igloc, i) = effective_potential_->f_pw(ig) * std::conj(rl->gvec_phase_factor<local>(igloc, ia));
                    }
                }

                linalg<CPU>::gemm(1, 0, nbf * (nbf + 1) / 2, atom_type->num_atoms(), (int)rl->spl_num_gvec().local_size(),
                                  &atom_type->uspp().q_pw(0, 0), (int)rl->spl_num_gvec().local_size(),
                                  &veff_a(0, 0), (int)rl->spl_num_gvec().local_size(),
                                  &d_tmp(0, 0), d_tmp.ld());
            }
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef _GPU_
                matrix<double_complex> veff_a(nullptr, rl->spl_num_gvec().local_size(), atom_type->num_atoms());
                veff_a.allocate_on_device();
                
                d_tmp.allocate_on_device();

                mdarray<double, 2> atom_pos(3, atom_type->num_atoms());
                for (int i = 0; i < atom_type->num_atoms(); i++)
                {
                    int ia = atom_type->atom_id(i);
                    for (int x = 0; x < 3; x++) atom_pos(x, i) = unit_cell_.atom(ia)->position(x);
                }
                atom_pos.allocate_on_device();
                atom_pos.copy_to_device();

                mul_veff_with_phase_factors_gpu(atom_type->num_atoms(),
                                                (int)rl->spl_num_gvec().local_size(),
                                                veff.at<GPU>(),
                                                gvec.at<GPU>(),
                                                atom_pos.at<GPU>(),
                                                veff_a.at<GPU>());

                linalg<GPU>::gemm(1, 0, nbf * (nbf + 1) / 2, atom_type->num_atoms(), (int)rl->spl_num_gvec().local_size(),
                                  atom_type->uspp().q_pw.at<GPU>(), (int)rl->spl_num_gvec().local_size(),
                                  veff_a.at<GPU>(), (int)rl->spl_num_gvec().local_size(), d_tmp.at<GPU>(), d_tmp.ld());

                d_tmp.copy_to_host();
                #else
                TERMINATE_NO_GPU
                #endif
            }

            comm_.allreduce(d_tmp.at<CPU>(), (int)d_tmp.size());

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < atom_type->num_atoms(); i++)
            {
                int ia = atom_type->atom_id(i);

                for (int xi2 = 0; xi2 < nbf; xi2++)
                {
                    int lm2 = atom_type->indexb(xi2).lm;
                    int idxrf2 = atom_type->indexb(xi2).idxrf;
                    for (int xi1 = 0; xi1 <= xi2; xi1++)
                    {
                        int lm1 = atom_type->indexb(xi1).lm;
                        int idxrf1 = atom_type->indexb(xi1).idxrf;
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;

                        if (xi1 == xi2)
                        {
                            unit_cell_.atom(ia)->d_mtrx(xi1, xi2) = real(d_tmp(idx12, i)) * unit_cell_.omega() +
                                                             atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
                        }
                        else
                        {
                            unit_cell_.atom(ia)->d_mtrx(xi1, xi2) = d_tmp(idx12, i) * unit_cell_.omega();
                            unit_cell_.atom(ia)->d_mtrx(xi2, xi1) = conj(d_tmp(idx12, i)) * unit_cell_.omega();
                            if (lm1 == lm2)
                            {
                                unit_cell_.atom(ia)->d_mtrx(xi1, xi2) += atom_type->uspp().d_mtrx_ion(idxrf1, idxrf2);
                                unit_cell_.atom(ia)->d_mtrx(xi2, xi1) += atom_type->uspp().d_mtrx_ion(idxrf2, idxrf1);
                            }
                        }
                    }
                }
            }
        }

        #ifdef _GPU_
        if (parameters_.processing_unit() == GPU)
        {
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                 auto type = unit_cell_.atom_type(iat);
                 type->uspp().q_pw.deallocate_on_device();
            }
        }
        #endif
    }
}

void Potential::set_effective_potential_ptr(double* veffmt, double* veffit)
{
    if (parameters_.esm_type() == full_potential_lapwlo || parameters_.esm_type() == full_potential_pwlo)
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
    mdarray<double,4> beffmt_tmp(beffmt, parameters_.lmmax_pot(), unit_cell_.max_num_mt_points(), 
                                 unit_cell_.num_atoms(), parameters_.num_mag_dims());
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
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++)
    {
       int ia = unit_cell_.atom_symmetry_class(ic)->atom_id(0);
       int nmtp = unit_cell_.atom(ia)->num_mt_points();
       
       std::vector<double> veff(nmtp);
       
       for (int ir = 0; ir < nmtp; ir++) veff[ir] = y00 * effective_potential_->f_mt<global>(0, ir, ia);

       const_cast<Atom_symmetry_class*>(unit_cell_.atom_symmetry_class(ic))->set_spherical_potential(veff);
    }
    
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        double* veff = &effective_potential_->f_mt<global>(0, 0, ia);
        
        double* beff[] = {nullptr, nullptr, nullptr};
        for (int i = 0; i < parameters_.num_mag_dims(); i++) beff[i] = &effective_magnetic_field_[i]->f_mt<global>(0, 0, ia);
        
        unit_cell_.atom(ia)->set_nonspherical_potential(veff, beff);
    }
}

void Potential::save()
{
    if (comm_.rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);

        effective_potential_->hdf5_write(fout["effective_potential"]);

        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            effective_magnetic_field_[j]->hdf5_write(fout["effective_magnetic_field"].create_node(j));

        //== fout["effective_potential"].create_node("free_atom_potential");
        //== for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        //== {
        //==     fout["effective_potential"]["free_atom_potential"].write(iat, unit_cell_.atom_type(iat)->free_atom_potential());
        //== }
    }
    comm_.barrier();
}

void Potential::load()
{
    HDF5_tree fout(storage_file_name, false);
    
    effective_potential_->hdf5_read(fout["effective_potential"]);

    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        effective_magnetic_field_[j]->hdf5_read(fout["effective_magnetic_field"][j]);
    
    //== for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    //==     fout["effective_potential"]["free_atom_potential"].read(iat, unit_cell_.atom_type(iat)->free_atom_potential());

    if (parameters_.full_potential()) update_atomic_potential();
}

void Potential::generate_local_potential()
{
    Timer t("sirius::Potential::generate_local_potential");

    auto rl = ctx_.reciprocal_lattice();

    mdarray<double, 2> vloc_radial_integrals(unit_cell_.num_atom_types(), rl->num_gvec_shells_inner());

    /* split G-shells between MPI ranks */
    splindex<block> spl_gshells(rl->num_gvec_shells_inner(), comm_.size(), comm_.rank());

    #pragma omp parallel
    {
        /* splines for all atom types */
        std::vector< Spline<double> > sa(unit_cell_.num_atom_types());
        
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) sa[iat] = Spline<double>(unit_cell_.atom_type(iat)->radial_grid());
    
        #pragma omp for
        for (int igsloc = 0; igsloc < (int)spl_gshells.local_size(); igsloc++)
        {
            int igs = (int)spl_gshells[igsloc];

            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            {
                auto atom_type = unit_cell_.atom_type(iat);

                if (igs == 0)
                {
                    for (int ir = 0; ir < atom_type->num_mt_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        sa[iat][ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn()) * x;
                    }
                    vloc_radial_integrals(iat, igs) = sa[iat].interpolate().integrate(0);
                }
                else
                {
                    double g = rl->gvec_shell_len(igs);
                    double g2 = std::pow(g, 2);
                    for (int ir = 0; ir < atom_type->num_mt_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        sa[iat][ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn() * gsl_sf_erf(x)) * sin(g * x);
                    }
                    vloc_radial_integrals(iat, igs) = (sa[iat].interpolate().integrate(0) / g - atom_type->zn() * exp(-g2 / 4) / g2);
                }
            }
        }
    }

    int ld = unit_cell_.num_atom_types();
    comm_.allgather(vloc_radial_integrals.at<CPU>(), static_cast<int>(ld * spl_gshells.global_offset()), 
                                 static_cast<int>(ld * spl_gshells.local_size()));

    std::vector<double_complex> v = rl->make_periodic_function(vloc_radial_integrals, rl->num_gvec());
    fft_->input(fft_->num_gvec(), fft_->index_map(), &v[0]); 
    fft_->transform(1);
    fft_->output(&local_potential_->f_it<global>(0));
}

}

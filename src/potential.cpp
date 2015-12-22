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

namespace sirius {

// TODO: everything here must be documented
// TODO: better naming convention: q is meaningless

Potential::Potential(Simulation_context& ctx__)
    : ctx_(ctx__),
      parameters_(ctx__.parameters()),
      unit_cell_(ctx__.unit_cell()),
      comm_(ctx__.comm()),
      fft_(ctx__.fft(0)),
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

    effective_potential_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot());
    
    if (parameters_.full_potential())
    {
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            effective_magnetic_field_[j] = new Periodic_function<double>(ctx_, parameters_.lmmax_pot(), false);
    }
    else
    {
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            effective_magnetic_field_[j] = new Periodic_function<double>(ctx_, parameters_.lmmax_pot());
    }
    
    hartree_potential_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot());
    hartree_potential_->allocate(false);
    
    xc_potential_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot(), false);
    xc_potential_->allocate(false);
    
    xc_energy_density_ = new Periodic_function<double>(ctx_, parameters_.lmmax_pot(), false);
    xc_energy_density_->allocate(false);

    if (!parameters_.full_potential())
    {
        local_potential_ = new Periodic_function<double>(ctx_, 0);
        local_potential_->allocate(false);
        local_potential_->zero();

        generate_local_potential();
    }

    vh_el_ = mdarray<double, 1>(unit_cell_.num_atoms());

    init();

    std::vector<double> weights;
    mixer_ = new Broyden_modified_mixer<double>(size(),
                                       parameters_.mixer_input_section().max_history_,
                                       parameters_.mixer_input_section().beta_,
                                       weights,
                                       comm_);

    spl_num_gvec_ = splindex<block>(ctx_.gvec().num_gvec(), comm_.size(), comm_.rank());
    
    if (parameters_.full_potential())
    {
        gvec_ylm_ = mdarray<double_complex, 2>(parameters_.lmmax_pot(), spl_num_gvec_.local_size());
        for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
        {
            int ig = (int)spl_num_gvec_[igloc];
            auto rtp = SHT::spherical_coordinates(ctx_.gvec().cart(ig));
            SHT::spherical_harmonics(parameters_.lmax_pot(), rtp[1], rtp[2], &gvec_ylm_(0, igloc));
        }
    }
}

Potential::~Potential()
{
    delete effective_potential_; 
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete effective_magnetic_field_[j];
    if (parameters_.esm_type() == full_potential_lapwlo) delete sht_;
    delete hartree_potential_;
    delete xc_potential_;
    delete xc_energy_density_;
    if (!parameters_.full_potential()) delete local_potential_;
    delete mixer_;
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

//== #ifdef __GPU
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
    for (int ir = 0; ir < fft_->local_size(); ir++)
        fft_->buffer(ir) = effective_potential()->f_it(ir) * ctx_.step_function()->theta_r(ir);

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z2 = mdarray<double_complex, 1>(&fft_->buffer(0), fft_->size()).checksum();
    DUMP("checksum(veff_it): %18.10f", mdarray<double, 1>(&effective_potential()->f_it(0) , fft_->size()).checksum());
    DUMP("checksum(fft_buffer): %18.10f %18.10f", std::real(z2), std::imag(z2));
    #endif
    
    fft_->transform<-1>(ctx_.gvec(), &effective_potential()->f_pw(ctx_.gvec().offset_gvec_fft()));
    fft_->comm().allgather(&effective_potential()->f_pw(0), ctx_.gvec().offset_gvec_fft(), ctx_.gvec().num_gvec_fft());

    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(veff_it): %18.10f", effective_potential()->f_it().checksum());
    double_complex z1 = mdarray<double_complex, 1>(&effective_potential()->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    DUMP("checksum(veff_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
    #endif

    if (!use_second_variation) // for full diagonalization we also need Beff(G)
    {
        for (int i = 0; i < parameters_.num_mag_dims(); i++)
        {
            for (int ir = 0; ir < fft_->size(); ir++)
                fft_->buffer(ir) = effective_magnetic_field(i)->f_it(ir) * ctx_.step_function()->theta_r(ir);
            
            STOP();
            //fft_->transform(-1, ctx_.gvec().z_sticks_coord());
            //fft_->output(ctx_.gvec().num_gvec(), ctx_.gvec().index_map(), &effective_magnetic_field(i)->f_pw(0));
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
            #ifdef __GPU
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
    PROFILE_WITH_TIMER("sirius::Potential::generate_effective_potential");

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
    PROFILE_WITH_TIMER("sirius::Potential::generate_effective_potential");

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
        rhovc->allocate(false);
        rhovc->zero();
        rhovc->add(rho);
        rhovc->add(rho_core);
        rhovc->sync(false, true);

        /* construct XC potentials from rho + rho_core */
        xc(rhovc, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
        
        /* destroy temporary function */
        delete rhovc;
        
        /* add XC potential to the effective potential */
        effective_potential_->add(xc_potential_);
        
        /* add local ionic potential to the effective potential */
        effective_potential_->add(local_potential_);

        /* synchronize effective potential */
        effective_potential_->sync(false, true);
        for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->sync(false, true);
    }

    if (parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        STOP();

        //zero();

        ///* create temporary function for rho + rho_core */
        //Periodic_function<double>* rhovc = new Periodic_function<double>(ctx_, 0, false);
        //rhovc->allocate(false, true);
        //rhovc->zero();
        //rhovc->add(rho);
        //rhovc->add(rho_core);
        ////rhovc->sync(false, true);

        ///* construct XC potentials from rho + rho_core */
        //xc(rhovc, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
        //
        ///* destroy temporary function */
        //delete rhovc;

        ///* add XC potential to the effective potential */
        //effective_potential_->add(xc_potential_);
        //
        ///* add local ionic potential to the effective potential */
        //effective_potential_->add(local_potential_);
        //effective_potential_->sync(false, true);
        //
        //Timer t1("sirius::Potential::generate_effective_potential|fft");
        //effective_potential_->fft_transform(-1);

        ///* get plane-wave coefficients of the charge density */
        //rho->fft_transform(-1);
        //t1.stop();

        //std::vector<double_complex> vtmp(spl_num_gvec_.local_size());
        //
        //energy_vha_ = 0.0;
        //for (int igloc = 0; igloc < (int)spl_num_gvec_.local_size(); igloc++)
        //{
        //    int ig = (int)spl_num_gvec_[igloc];
        //    vtmp[igloc] = (ig == 0) ? 0.0 : rho->f_pw(ig) * fourpi / std::pow(fft_->gvec_len(ig), 2);
        //    energy_vha_ += std::real(std::conj(rho->f_pw(ig)) * vtmp[igloc]);
        //    vtmp[igloc] += effective_potential_->f_pw(ig);
        //}
        //comm_.allreduce(&energy_vha_, 1);
        //energy_vha_ *= unit_cell_.omega();

        //auto offsets = spl_num_gvec_.offsets();
        //auto counts = spl_num_gvec_.counts();
        ////parameters_.comm().allgather(&vtmp[0], spl_num_gvec_.local_size(), &hartree_potential_->f_pw(0), &counts[0], &offsets[0]);
        //comm_.allgather(&vtmp[0], (int)spl_num_gvec_.local_size(), &effective_potential_->f_pw(0), &counts[0], &offsets[0]);

        ////for (int ig = 0; ig < rl->num_gvec(); ig++)
        ////    effective_potential_->f_pw(ig) += hartree_potential_->f_pw(ig);


    }

    generate_D_operator_matrix();
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


}

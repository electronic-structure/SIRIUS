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

//template<> void Potential::add_mt_contribution_to_pw<CPU>()
//{
//    Timer t("sirius::Potential::add_mt_contribution_to_pw");
//
//    mdarray<double_complex, 1> fpw(ctx_.reciprocal_lattice()->num_gvec());
//    fpw.zero();
//
//    mdarray<Spline<double>*, 2> svlm(ctx_.lmmax_pot(), unit_cell_.num_atoms());
//    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//    {
//        for (int lm = 0; lm < ctx_.lmmax_pot(); lm++)
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
//        mdarray<double, 1> vjlm(ctx_.lmmax_pot());
//
//        sbessel_pw<double> jl(ctx_.unit_cell(), ctx_.lmax_pot());
//        
//        #pragma omp for
//        for (int igloc = 0; igloc < (int)ctx_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
//        {
//            int ig = ctx_.reciprocal_lattice()->spl_num_gvec(igloc);
//
//            jl.interpolate(ctx_.reciprocal_lattice()->gvec_len(ig));
//
//            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//            {
//                int iat = unit_cell_.atom(ia)->type_id();
//
//                for (int lm = 0; lm < ctx_.lmmax_pot(); lm++)
//                {
//                    int l = l_by_lm_[lm];
//                    vjlm(lm) = Spline<double>::integrate(jl(l, iat), svlm(lm, ia), 2);
//                }
//
//                double_complex zt(0, 0);
//                for (int l = 0; l <= ctx_.lmax_pot(); l++)
//                {
//                    for (int m = -l; m <= l; m++)
//                    {
//                        if (m == 0)
//                        {
//                            zt += conj(zil_[l]) * ctx_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//                                  vjlm(Utils::lm_by_l_m(l, m));
//
//                        }
//                        else
//                        {
//                            zt += conj(zil_[l]) * ctx_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//                                  (SHT::ylm_dot_rlm(l, m, m) * vjlm(Utils::lm_by_l_m(l, m)) + 
//                                   SHT::ylm_dot_rlm(l, m, -m) * vjlm(Utils::lm_by_l_m(l, -m)));
//                        }
//                    }
//                }
//                fpw(ig) += zt * fourpi * conj(ctx_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia)) / unit_cell_.omega();
//            }
//        }
//    }
//    ctx_.comm().allreduce(fpw.at<CPU>(), (int)fpw.size());
//    for (int ig = 0; ig < ctx_.reciprocal_lattice()->num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
//    
//    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//    {
//        for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) delete svlm(lm, ia);
//    }
//}

//== #ifdef __GPU
//== template <> void Potential::add_mt_contribution_to_pw<GPU>()
//== {
//==     // TODO: couple of things to consider: 1) global array jvlm with G-vector shells may be large; 
//==     //                                     2) MPI reduction over thousands of shell may be slow
//==     Timer t("sirius::Potential::add_mt_contribution_to_pw");
//== 
//==     mdarray<double_complex, 1> fpw(ctx_.num_gvec());
//==     fpw.zero();
//==     
//==     mdarray<int, 1> kargs(4);
//==     kargs(0) = ctx_.num_atom_types();
//==     kargs(1) = ctx_.max_num_mt_points();
//==     kargs(2) = ctx_.lmax_pot();
//==     kargs(3) = ctx_.lmmax_pot();
//==     kargs.allocate_on_device();
//==     kargs.copy_to_device();
//== 
//==     mdarray<double, 3> vlm_coefs(ctx_.max_num_mt_points() * 4, ctx_.lmmax_pot(), 
//==                                  ctx_.num_atoms());
//==     for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     {
//==         for (int lm = 0; lm < ctx_.lmmax_pot(); lm++)
//==         {
//==             Spline<double> s(ctx_.atom(ia)->num_mt_points(), 
//==                              ctx_.atom(ia)->type()->radial_grid());
//==             
//==             for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==                 s[ir] = effective_potential_->f_rlm(lm, ir, ia);
//==             
//==             s.interpolate();
//==             s.get_coefs(&vlm_coefs(0, lm, ia), ctx_.max_num_mt_points());
//==         }
//==     }
//==     vlm_coefs.allocate_on_device();
//==     vlm_coefs.copy_to_device();
//== 
//==     mdarray<int, 1> iat_by_ia(ctx_.num_atoms());
//==     for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==         iat_by_ia(ia) = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     iat_by_ia.allocate_on_device();
//==     iat_by_ia.copy_to_device();
//== 
//==     l_by_lm_.allocate_on_device();
//==     l_by_lm_.copy_to_device();
//==     
//==     //=============
//==     // radial grids
//==     //=============
//==     mdarray<double, 2> r_dr(ctx_.max_num_mt_points() * 2, ctx_.num_atom_types());
//==     mdarray<int, 1> nmtp_by_iat(ctx_.num_atom_types());
//==     for (int iat = 0; iat < ctx_.num_atom_types(); iat++)
//==     {
//==         nmtp_by_iat(iat) = ctx_.atom_type(iat)->num_mt_points();
//==         ctx_.atom_type(iat)->radial_grid().get_r_dr(&r_dr(0, iat), ctx_.max_num_mt_points());
//==     }
//==     r_dr.allocate_on_device();
//==     r_dr.async_copy_to_device(-1);
//==     nmtp_by_iat.allocate_on_device();
//==     nmtp_by_iat.async_copy_to_device(-1);
//== 
//==     splindex<block> spl_num_gvec_shells(ctx_.num_gvec_shells(), Platform::num_mpi_ranks(), Platform::mpi_rank());
//==     mdarray<double, 3> jvlm(ctx_.lmmax_pot(), ctx_.num_atoms(), ctx_.num_gvec_shells());
//==     jvlm.zero();
//== 
//==     cuda_create_streams(Platform::num_threads());
//==     #pragma omp parallel
//==     {
//==         int thread_id = Platform::thread_id();
//== 
//==         mdarray<double, 3> jl_coefs(ctx_.max_num_mt_points() * 4, ctx_.lmax_pot() + 1, 
//==                                     ctx_.num_atom_types());
//==         
//==         mdarray<double, 2> jvlm_loc(ctx_.lmmax_pot(), ctx_.num_atoms());
//== 
//==         jvlm_loc.pin_memory();
//==         jvlm_loc.allocate_on_device();
//==             
//==         jl_coefs.pin_memory();
//==         jl_coefs.allocate_on_device();
//== 
//==         sbessel_pw<double> jl(ctx_, ctx_.lmax_pot());
//==         
//==         #pragma omp for
//==         for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
//==         {
//==             int igs = spl_num_gvec_shells[igsloc];
//== 
//==             jl.interpolate(ctx_.gvec_shell_len(igs));
//== 
//==             for (int iat = 0; iat < ctx_.num_atom_types(); iat++)
//==             {
//==                 for (int l = 0; l <= ctx_.lmax_pot(); l++)
//==                     jl(l, iat)->get_coefs(&jl_coefs(0, l, iat), ctx_.max_num_mt_points());
//==             }
//==             jl_coefs.async_copy_to_device(thread_id);
//== 
//==             sbessel_vlm_inner_product_gpu(kargs.ptr_device(), ctx_.lmmax_pot(), ctx_.num_atoms(), 
//==                                           iat_by_ia.ptr_device(), l_by_lm_.ptr_device(), 
//==                                           nmtp_by_iat.ptr_device(), r_dr.ptr_device(), 
//==                                           jl_coefs.ptr_device(), vlm_coefs.ptr_device(), jvlm_loc.ptr_device(), 
//==                                           thread_id);
//== 
//==             jvlm_loc.async_copy_to_host(thread_id);
//==             
//==             cuda_stream_synchronize(thread_id);
//== 
//==             memcpy(&jvlm(0, 0, igs), &jvlm_loc(0, 0), ctx_.lmmax_pot() * ctx_.num_atoms() * sizeof(double));
//==         }
//==     }
//==     cuda_destroy_streams(Platform::num_threads());
//==     
//==     for (int igs = 0; igs < ctx_.num_gvec_shells(); igs++)
//==         Platform::allreduce(&jvlm(0, 0, igs), ctx_.lmmax_pot() * ctx_.num_atoms());
//== 
//==     #pragma omp parallel for default(shared)
//==     for (int igloc = 0; igloc < ctx_.spl_num_gvec().local_size(); igloc++)
//==     {
//==         int ig = ctx_.spl_num_gvec(igloc);
//==         int igs = ctx_.gvec_shell<local>(igloc);
//== 
//==         for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==         {
//==             double_complex zt(0, 0);
//==             for (int l = 0; l <= ctx_.lmax_pot(); l++)
//==             {
//==                 for (int m = -l; m <= l; m++)
//==                 {
//==                     if (m == 0)
//==                     {
//==                         zt += conj(zil_[l]) * ctx_.gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//==                               jvlm(Utils::lm_by_l_m(l, m), ia, igs);
//== 
//==                     }
//==                     else
//==                     {
//==                         zt += conj(zil_[l]) * ctx_.gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//==                               (SHT::ylm_dot_rlm(l, m, m) * jvlm(Utils::lm_by_l_m(l, m), ia, igs) + 
//==                                SHT::ylm_dot_rlm(l, m, -m) * jvlm(Utils::lm_by_l_m(l, -m), ia, igs));
//==                     }
//==                 }
//==             }
//==             fpw(ig) += zt * fourpi * conj(ctx_.gvec_phase_factor<local>(igloc, ia)) / ctx_.omega();
//==         }
//==     }
//== 
//==     Platform::allreduce(fpw.at<CPU>(), (int)fpw.size());
//==     for (int ig = 0; ig < ctx_.num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
//== 
//==     l_by_lm_.deallocate_on_device();
//== }
//== #endif

//void Potential::check_potential_continuity_at_mt()
//{
//    // generate plane-wave coefficients of the potential in the interstitial region
//    ctx_.fft().input(&effective_potential_->f_it<global>(0));
//    ctx_.fft().transform(-1);
//    ctx_.fft().output(ctx_.num_gvec(), ctx_.fft_index(), &effective_potential_->f_pw(0));
//    
//    SHT sht(ctx_.lmax_pot());
//
//    double diff = 0.0;
//    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//    {
//        for (int itp = 0; itp < sht.num_points(); itp++)
//        {
//            double vc[3];
//            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * ctx_.atom(ia)->mt_radius();
//
//            double val_it = 0.0;
//            for (int ig = 0; ig < ctx_.num_gvec(); ig++) 
//            {
//                double vgc[3];
//                ctx_.get_coordinates<cartesian, reciprocal>(ctx_.gvec(ig), vgc);
//                val_it += real(effective_potential_->f_pw(ig) * exp(double_complex(0.0, Utils::scalar_product(vc, vgc))));
//            }
//
//            double val_mt = 0.0;
//            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++)
//                val_mt += effective_potential_->f_rlm(lm, ctx_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
//
//            diff += fabs(val_it - val_mt);
//        }
//    }
//    printf("Total and average potential difference at MT boundary : %.12f %.12f\n", diff, diff / ctx_.num_atoms() / sht.num_points());
//}

void Potential::set_effective_potential_ptr(double* veffmt, double* veffit)
{
    if (ctx_.full_potential()) effective_potential_->set_mt_ptr(veffmt);
    effective_potential_->set_rg_ptr(veffit);
}

//void Potential::copy_to_global_ptr(double* fmt__, double* fit__, Periodic_function<double>* src)
//{
//    stop_here // fix thsi
//    //Periodic_function<double>* dest = new Periodic_function<double>(ctx_, ctx_.lmmax_pot());
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
//**         new Periodic_function<double>(ctx_, Argument(arg_lm, ctx_.lmmax_pot()),
//**                                                    Argument(arg_radial, ctx_.max_num_mt_points()));
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
//**     if (ctx_.num_mag_dims() == 0) return;
//**     assert(ctx_.num_spins() == 2);
//**     
//**     // set temporary array wrapper
//**     mdarray<double,4> beffmt_tmp(beffmt, ctx_.lmmax_pot(), ctx_.max_num_mt_points(), 
//**                                  ctx_.num_atoms(), ctx_.num_mag_dims());
//**     mdarray<double,2> beffit_tmp(beffit, ctx_.fft().size(), ctx_.num_mag_dims());
//**     
//**     Periodic_function<double>* bxc[3];
//**     for (int i = 0; i < ctx_.num_mag_dims(); i++)
//**     {
//**         bxc[i] = new Periodic_function<double>(ctx_, Argument(arg_lm, ctx_.lmmax_pot()),
//**                                                             Argument(arg_radial, ctx_.max_num_mt_points()));
//**     }
//** 
//**     if (ctx_.num_mag_dims() == 1)
//**     {
//**         // z
//**         bxc[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
//**         bxc[0]->set_it_ptr(&beffit_tmp(0, 0));
//**     }
//**     
//**     if (ctx_.num_mag_dims() == 3)
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
//**     for (int i = 0; i < ctx_.num_mag_dims(); i++)
//**     {
//**         bxc[i]->copy(effective_magnetic_field_[i]);
//**         bxc[i]->sync();
//**         delete bxc[i];
//**     }
//** }

void Potential::set_effective_magnetic_field_ptr(double* beffmt, double* beffit)
{
    if (ctx_.num_mag_dims() == 0) return;
    assert(ctx_.num_spins() == 2);
    
    // set temporary array wrapper
    mdarray<double,4> beffmt_tmp(beffmt, ctx_.lmmax_pot(), unit_cell_.max_num_mt_points(), 
                                 unit_cell_.num_atoms(), ctx_.num_mag_dims());
    mdarray<double,2> beffit_tmp(beffit, fft_.size(), ctx_.num_mag_dims());
    
    if (ctx_.num_mag_dims() == 1)
    {
        // z
        effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
        effective_magnetic_field_[0]->set_rg_ptr(&beffit_tmp(0, 0));
    }
    
    if (ctx_.num_mag_dims() == 3)
    {
        // z
        effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 2));
        effective_magnetic_field_[0]->set_rg_ptr(&beffit_tmp(0, 2));
        // x
        effective_magnetic_field_[1]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
        effective_magnetic_field_[1]->set_rg_ptr(&beffit_tmp(0, 0));
        // y
        effective_magnetic_field_[2]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 1));
        effective_magnetic_field_[2]->set_rg_ptr(&beffit_tmp(0, 1));
    }
}
         
void Potential::zero()
{
    effective_potential_->zero();
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        effective_magnetic_field_[j]->zero();
    }
}

void Potential::update_atomic_potential()
{
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
        int ia = unit_cell_.atom_symmetry_class(ic).atom_id(0);
        int nmtp = unit_cell_.atom(ia).num_mt_points();
       
        std::vector<double> veff(nmtp);
       
        for (int ir = 0; ir < nmtp; ir++) {
            veff[ir] = y00 * effective_potential_->f_mt<index_domain_t::global>(0, ir, ia);
        }

       unit_cell_.atom_symmetry_class(ic).set_spherical_potential(veff);
    }
    
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        double* veff = &effective_potential_->f_mt<index_domain_t::global>(0, 0, ia);
        
        double* beff[] = {nullptr, nullptr, nullptr};
        for (int i = 0; i < ctx_.num_mag_dims(); i++) {
            beff[i] = &effective_magnetic_field_[i]->f_mt<index_domain_t::global>(0, 0, ia);
        }
        
        unit_cell_.atom(ia).set_nonspherical_potential(veff, beff);
    }
}

void Potential::save()
{
    if (comm_.rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);

        effective_potential_->hdf5_write(fout["effective_potential"]);

        for (int j = 0; j < ctx_.num_mag_dims(); j++)
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

    for (int j = 0; j < ctx_.num_mag_dims(); j++)
        effective_magnetic_field_[j]->hdf5_read(fout["effective_magnetic_field"][j]);
    
    //== for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    //==     fout["effective_potential"]["free_atom_potential"].read(iat, unit_cell_.atom_type(iat)->free_atom_potential());

    if (ctx_.full_potential()) update_atomic_potential();
}


}

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

/** \file band.cpp
 *   
 *   \brief Contains remaining implementation of sirius::Band class.
 */

#include "band.h"

namespace sirius {

//== void Band::apply_so_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi)
//== {
//==     Timer t("sirius::Band::apply_so_correction");
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         Atom_type* type = unit_cell_.atom(ia)->type();
//== 
//==         int offset = unit_cell_.atom(ia)->offset_wf();
//== 
//==         for (int l = 0; l <= parameters_.lmax_apw(); l++)
//==         {
//==             int nrf = type->indexr().num_rf(l);
//== 
//==             for (int order1 = 0; order1 < nrf; order1++)
//==             {
//==                 for (int order2 = 0; order2 < nrf; order2++)
//==                 {
//==                     double sori = unit_cell_.atom(ia)->symmetry_class()->so_radial_integral(l, order1, order2);
//==                     
//==                     for (int m = -l; m <= l; m++)
//==                     {
//==                         int idx1 = type->indexb_by_l_m_order(l, m, order1);
//==                         int idx2 = type->indexb_by_l_m_order(l, m, order2);
//==                         int idx3 = (m + l != 0) ? type->indexb_by_l_m_order(l, m - 1, order2) : 0;
//==                         int idx4 = (m - l != 0) ? type->indexb_by_l_m_order(l, m + 1, order2) : 0;
//== 
//==                         for (int ist = 0; ist < (int)parameters_.spl_fv_states().local_size(); ist++)
//==                         {
//==                             double_complex z1 = fv_states(offset + idx2, ist) * double(m) * sori;
//==                             hpsi(offset + idx1, ist, 0) += z1;
//==                             hpsi(offset + idx1, ist, 1) -= z1;
//==                             // apply L_{-} operator
//==                             if (m + l) hpsi(offset + idx1, ist, 2) += fv_states(offset + idx3, ist) * sori * 
//==                                                                       sqrt(double(l * (l + 1) - m * (m - 1)));
//==                             // apply L_{+} operator
//==                             if (m - l) hpsi(offset + idx1, ist, 3) += fv_states(offset + idx4, ist) * sori * 
//==                                                                       sqrt(double(l * (l + 1) - m * (m + 1)));
//==                         }
//==                     }
//==                 }
//==             }
//==         }
//==     }
//== }

//=====================================================================================================================
// CPU code, plane-wave basis
//=====================================================================================================================
//** template<> void Band::set_fv_h_o<cpu, pwlo>(K_point* kp, Periodic_function<double>* effective_potential, 
//**                                             mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
//** {
//**     Timer t("sirius::Band::set_fv_h_o");
//**     
//**     h.zero();
//**     o.zero();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc2 = 0; igkloc2 < k->num_gkvec_col(); igkloc2++) // loop over columns
//**     {
//**         for (int igkloc1 = 0; igkloc1 < kp->num_gkvec_row(); igkloc1++) // for each column loop over rows
//**         {
//**             if (kp->gklo_basis_descriptor_row(igkloc1).idxglob == gklo_basis_descriptor_col_[igkloc2].idxglob) 
//**             {
//**                 h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
//**                 o(igkloc1, igkloc2) = double_complex(1, 0);
//**             }
//**                                
//**             int ig12 = parameters_.index_g12(gklo_basis_descriptor_row_[igkloc1].ig,
//**                                              gklo_basis_descriptor_col_[igkloc2].ig);
//**             h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
//**         }
//**     }
//**     
//**     set_fv_h_o_pw_lo<CPU>(effective_potential, num_ranks, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** }
//**
//** template<> void K_point::set_fv_h_o_pw_lo<CPU>(Periodic_function<double>* effective_potential, int num_ranks, 
//**                                               mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o_pw_lo");
//**     
//**     int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
//**     
//**     mdarray<Spline<double_complex>*, 2> svlo(parameters_.lmmax_pw(), std::max(num_lo_col(), num_lo_row()));
//** 
//**     // first part: compute <G+k|H|lo> and <G+k|lo>
//** 
//**     Timer t1("sirius::K_point::set_fv_h_o_pw_lo:vlo", false);
//**     Timer t2("sirius::K_point::set_fv_h_o_pw_lo:ohk", false);
//**     Timer t3("sirius::K_point::set_fv_h_o_pw_lo:hvlo", false);
//** 
//**     // compute V|lo>
//**     t1.start();
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         int ia = gklo_basis_descriptor_col_[num_gkvec_col() + icol].ia;
//**         int lm = gklo_basis_descriptor_col_[num_gkvec_col() + icol].lm;
//**         int idxrf = gklo_basis_descriptor_col_[num_gkvec_col() + icol].idxrf;
//**         
//**         for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**         {
//**             svlo(lm1, icol) = new Spline<double_complex>(parameters_.atom(ia)->num_mt_points(), 
//**                                                     parameters_.atom(ia)->radial_grid());
//** 
//**             for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**             {
//**                 int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**                 double_complex cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**                 for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**                 {
//**                     (*svlo(lm1, icol))[ir] += (cg * effective_potential->f_mt<global>(lm3, ir, ia) * 
//**                                                parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**                 }
//**             }
//** 
//**             svlo(lm1, icol)->interpolate();
//**         }
//**     }
//**     t1.stop();
//**     
//**     t2.start();
//**     // compute overlap and kinetic energy
//**     for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**     {
//**         int ia = gklo_basis_descriptor_col_[icol].ia;
//**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**         int l = gklo_basis_descriptor_col_[icol].l;
//**         int lm = gklo_basis_descriptor_col_[icol].lm;
//**         int idxrf = gklo_basis_descriptor_col_[icol].idxrf;
//** 
//**         Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**         for (int ir = 0; ir < slo.num_points(); ir++)
//**             slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**         slo.interpolate();
//**         
//**         #pragma omp parallel for default(shared)
//**         for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**         {
//**             o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
//**                               Spline<double>::integrate(&slo, (*sbessel_[igkloc])(l, iat)) * 
//**                               conj(gkvec_phase_factors_(igkloc, ia));
//** 
//**             // kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
//**             h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
//**         }
//**     }
//**     t2.stop();
//** 
//**     t3.start();
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**     {
//**         for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**         {
//**             int ia = gklo_basis_descriptor_col_[icol].ia;
//**             int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**             //int l = gklo_basis_descriptor_col_[icol].l;
//**             //int lm = gklo_basis_descriptor_col_[icol].lm;
//**             //int idxrf = gklo_basis_descriptor_col_[icol].idxrf;
//** 
//**             //*// compue overlap <G+k|lo>
//**             //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**             //*for (int ir = 0; ir < s.num_points(); ir++)
//**             //*{
//**             //*    s[ir] = (*sbessel_[igkloc])(ir, l, iat) * 
//**             //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**             //*}
//**             //*s.interpolate();
//**             //*    
//**             //*o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
//**             //*                  s.integrate(2) * conj(gkvec_phase_factors_(igkloc, ia));
//** 
//**             //*// kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
//**             //*h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
//** 
//**             // add <G+k|V|lo>
//**             double_complex zt1(0, 0);
//**             for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
//**             {
//**                 for (int m1 = -l1; m1 <= l1; m1++)
//**                 {
//**                     int lm1 = Utils::lm_by_l_m(l1, m1);
//** 
//**                     zt1 += Spline<double_complex>::integrate(svlo(lm1, icol - num_gkvec_col()), 
//**                                                         (*sbessel_[igkloc])(l1, iat)) * 
//**                            conj(zil_[l1]) * gkvec_ylm_(lm1, igkloc);
//**                 }
//**             }
//**             zt1 *= ((fourpi / sqrt(parameters_.omega())) * conj(gkvec_phase_factors_(igkloc, ia)));
//**             h(igkloc, icol) += zt1;
//**         }
//**     }
//**     t3.stop();
//**    
//**     // deallocate V|lo>
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, icol);
//**     }
//** 
//**     // restore the <lo|H|G+k> and <lo|G+k> bocks and exit
//**     if (num_ranks == 1)
//**     {
//**         for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**         {
//**             for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**             {
//**                 h(irow, igkloc) = conj(h(igkloc, irow));
//**                 o(irow, igkloc) = conj(o(igkloc, irow));
//**             }
//**         }
//**         return;
//**     }
//** 
//**     // second part: compute <lo|H|G+k> and <lo|G+k>
//** 
//**     // compute V|lo>
//**     t1.start();
//**     for (int irow = 0; irow < num_lo_row(); irow++)
//**     {
//**         int ia = gklo_basis_descriptor_row_[num_gkvec_row() + irow].ia;
//**         int lm = gklo_basis_descriptor_row_[num_gkvec_row() + irow].lm;
//**         int idxrf = gklo_basis_descriptor_row_[num_gkvec_row() + irow].idxrf;
//**         
//**         for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**         {
//**             svlo(lm1, irow) = new Spline<double_complex>(parameters_.atom(ia)->num_mt_points(), 
//**                                                     parameters_.atom(ia)->radial_grid());
//** 
//**             for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**             {
//**                 int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**                 double_complex cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**                 for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**                 {
//**                     (*svlo(lm1, irow))[ir] += (cg * effective_potential->f_mt<global>(lm3, ir, ia) * 
//**                                                parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**                 }
//**             }
//** 
//**             svlo(lm1, irow)->interpolate();
//**         }
//**     }
//**     t1.stop();
//**    
//**     t2.start();
//**     for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**     {
//**         int ia = gklo_basis_descriptor_row_[irow].ia;
//**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**         int l = gklo_basis_descriptor_row_[irow].l;
//**         int lm = gklo_basis_descriptor_row_[irow].lm;
//**         int idxrf = gklo_basis_descriptor_row_[irow].idxrf;
//** 
//**         // compue overlap <lo|G+k>
//**         Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**         for (int ir = 0; ir < slo.num_points(); ir++)
//**             slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**         slo.interpolate();
//**         
//**         #pragma omp parallel for default(shared)
//**         for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**         {
//**             o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**                               conj(gkvec_ylm_(lm, offset_col + igkloc)) * 
//**                               Spline<double>::integrate(&slo, (*sbessel_[offset_col + igkloc])(l, iat)) * 
//**                               gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**             // kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**             h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//**         }
//**     }
//**     t2.stop();
//** 
//**     t3.start();
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**     {
//**         for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**         {
//**             int ia = gklo_basis_descriptor_row_[irow].ia;
//**             int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**             //int l = gklo_basis_descriptor_row_[irow].l;
//**             //int lm = gklo_basis_descriptor_row_[irow].lm;
//**             //int idxrf = gklo_basis_descriptor_row_[irow].idxrf;
//** 
//**             //*// compue overlap <lo|G+k>
//**             //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**             //*for (int ir = 0; ir < s.num_points(); ir++)
//**             //*    s[ir] = (*sbessel_[offset_col + igkloc])(ir, l, iat) * 
//**             //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**             //*s.interpolate();
//**             //*    
//**             //*o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**             //*                  conj(gkvec_ylm_(lm, offset_col + igkloc)) * s.integrate(2) * 
//**             //*                  gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**             //*// kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**             //*h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//** 
//**             // add <lo|V|G+k>
//**             double_complex zt1(0, 0);
//**             for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
//**             {
//**                 for (int m1 = -l1; m1 <= l1; m1++)
//**                 {
//**                     int lm1 = Utils::lm_by_l_m(l1, m1);
//** 
//**                     zt1 += conj(Spline<double_complex>::integrate(svlo(lm1, irow - num_gkvec_row()), 
//**                                                              (*sbessel_[offset_col + igkloc])(l1, iat))) * 
//**                            zil_[l1] * conj(gkvec_ylm_(lm1, offset_col + igkloc));
//**                 }
//**             }
//**             zt1 *= ((fourpi / sqrt(parameters_.omega())) * gkvec_phase_factors_(offset_col + igkloc, ia));
//**             h(irow, igkloc) += zt1;
//**         }
//**     }
//**     t3.stop();
//**     
//**     for (int irow = 0; irow < num_lo_row(); irow++)
//**     {
//**         for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, irow);
//**     }
//** }
//** 
//** template<> void K_point::set_fv_h_o<cpu, pwlo>(Periodic_function<double>* effective_potential, int num_ranks,
//**                                               mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o");
//**     
//**     h.zero();
//**     o.zero();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
//**     {
//**         for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
//**         {
//**             if (gklo_basis_descriptor_row_[igkloc1].idxglob == gklo_basis_descriptor_col_[igkloc2].idxglob) 
//**             {
//**                 h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
//**                 o(igkloc1, igkloc2) = double_complex(1, 0);
//**             }
//**                                
//**             int ig12 = parameters_.index_g12(gklo_basis_descriptor_row_[igkloc1].ig,
//**                                              gklo_basis_descriptor_col_[igkloc2].ig);
//**             h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
//**         }
//**     }
//**     
//**     set_fv_h_o_pw_lo<CPU>(effective_potential, num_ranks, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** }
//** 
//** 


//=====================================================================================================================
// GPU code, plane-wave basis
//=====================================================================================================================
//** template<> void K_point::set_fv_h_o<gpu, pwlo>(Periodic_function<double>* effective_potential, int num_ranks,
//**                                               mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o");
//**     
//**     h.zero();
//**     o.zero();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
//**     {
//**         for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
//**         {
//**             if (gklo_basis_descriptor_row_[igkloc1].idxglob == gklo_basis_descriptor_col_[igkloc2].idxglob) 
//**             {
//**                 h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
//**                 o(igkloc1, igkloc2) = double_complex(1, 0);
//**             }
//**                                
//**             int ig12 = parameters_.index_g12(gklo_basis_descriptor_row_[igkloc1].ig,
//**                                              gklo_basis_descriptor_col_[igkloc2].ig);
//**             h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
//**         }
//**     }
//**     
//**     set_fv_h_o_pw_lo<GPU>(effective_potential, num_ranks, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** }
//** #endif
//** 
//** template<> void K_point::set_fv_h_o_pw_lo<GPU>(Periodic_function<double>* effective_potential, int num_ranks, 
//**                                               mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o_pw_lo");
//**     
//**     // ===========================================
//**     // first part: compute <G+k|H|lo> and <G+k|lo>
//**     // ===========================================
//** 
//**     Timer t1("sirius::K_point::set_fv_h_o_pw_lo:vlo", false);
//**     Timer t2("sirius::K_point::set_fv_h_o_pw_lo:ohk", false);
//**     Timer t3("sirius::K_point::set_fv_h_o_pw_lo:hvlo", false);
//** 
//**     mdarray<int, 1> kargs(4);
//**     kargs(0) = parameters_.num_atom_types();
//**     kargs(1) = parameters_.max_num_mt_points();
//**     kargs(2) = parameters_.lmax_pw();
//**     kargs(3) = parameters_.lmmax_pw();
//**     kargs.allocate_on_device();
//**     kargs.copy_to_device();
//** 
//**     // =========================
//**     // compute V|lo> for columns
//**     // =========================
//**     t1.start();
//**     mdarray<double_complex, 3> vlo_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmmax_pw(), num_lo_col());
//**     #pragma omp parallel for default(shared)
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         int ia = gklo_basis_descriptor_col_[num_gkvec_col() + icol].ia;
//**         int lm = gklo_basis_descriptor_col_[num_gkvec_col() + icol].lm;
//**         int idxrf = gklo_basis_descriptor_col_[num_gkvec_col() + icol].idxrf;
//**         
//**         for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**         {
//**             Spline<double_complex> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//** 
//**             for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**             {
//**                 int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**                 double_complex cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**                 for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**                 {
//**                     s[ir] += (cg * effective_potential->f_rlm(lm3, ir, ia) * 
//**                               parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**                 }
//**             }
//**             s.interpolate();
//**             s.get_coefs(&vlo_coefs(0, lm1, icol), parameters_.max_num_mt_points());
//**         }
//**     }
//**     vlo_coefs.pin_memory();
//**     vlo_coefs.allocate_on_device();
//**     vlo_coefs.async_copy_to_device(-1);
//**     t1.stop();
//**     
//**     // ===========================================
//**     // pack Bessel function splines into one array
//**     // ===========================================
//**     mdarray<double, 4> sbessel_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmax_pw() + 1, 
//**                                      parameters_.num_atom_types(), num_gkvec_row());
//**     for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**     {
//**         for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//**         {
//**             for (int l = 0; l <= parameters_.lmax_pw(); l++)
//**             {
//**                 (*sbessel_[igkloc])(l, iat)->get_coefs(&sbessel_coefs(0, l, iat, igkloc), 
//**                                                        parameters_.max_num_mt_points());
//**             }
//**         }
//**     }
//**     sbessel_coefs.pin_memory();
//**     sbessel_coefs.allocate_on_device();
//**     sbessel_coefs.async_copy_to_device(-1);
//** 
//**     // ==============================
//**     // pack lo splines into one array
//**     // ==============================
//**     mdarray<double, 2> lo_coefs(parameters_.max_num_mt_points() * 4, num_lo_col());
//**     mdarray<int, 1> l_by_ilo(num_lo_col());
//**     mdarray<int, 1> iat_by_ilo(num_lo_col());
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         int ia = gklo_basis_descriptor_col_[num_gkvec_col() + icol].ia;
//**         l_by_ilo(icol) = gklo_basis_descriptor_col_[num_gkvec_col() + icol].l;
//**         iat_by_ilo(icol) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**         int idxrf = gklo_basis_descriptor_col_[num_gkvec_col() + icol].idxrf;
//** 
//**         Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**         for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**             s[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**         s.interpolate();
//**         s.get_coefs(&lo_coefs(0, icol), parameters_.max_num_mt_points());
//**     }
//**     lo_coefs.pin_memory();
//**     lo_coefs.allocate_on_device();
//**     lo_coefs.async_copy_to_device(-1);
//**     l_by_ilo.allocate_on_device();
//**     l_by_ilo.async_copy_to_device(-1);
//**     iat_by_ilo.allocate_on_device();
//**     iat_by_ilo.async_copy_to_device(-1);
//** 
//**     // ============
//**     // radial grids
//**     // ============
//**     mdarray<double, 2> r_dr(parameters_.max_num_mt_points() * 2, parameters_.num_atom_types());
//**     mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
//**     for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//**     {
//**         nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
//**         parameters_.atom_type(iat)->radial_grid().get_r_dr(&r_dr(0, iat), parameters_.max_num_mt_points());
//**     }
//**     r_dr.allocate_on_device();
//**     r_dr.async_copy_to_device(-1);
//**     nmtp_by_iat.allocate_on_device();
//**     nmtp_by_iat.async_copy_to_device(-1);
//** 
//**     mdarray<double, 2> jlo(num_gkvec_row(), num_lo_col());
//**     jlo.allocate_on_device();
//** 
//**     t2.start();
//**     sbessel_lo_inner_product_gpu(kargs.get_ptr_device(), num_gkvec_row(), num_lo_col(), l_by_ilo.get_ptr_device(), 
//**                                  iat_by_ilo.get_ptr_device(), nmtp_by_iat.get_ptr_device(), r_dr.get_ptr_device(), 
//**                                  sbessel_coefs.get_ptr_device(), lo_coefs.get_ptr_device(), jlo.get_ptr_device());
//**     jlo.copy_to_host();
//**     // compute overlap and kinetic energy
//**     for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**     {
//**         int ia = gklo_basis_descriptor_col_[icol].ia;
//**         int l = gklo_basis_descriptor_col_[icol].l;
//**         int lm = gklo_basis_descriptor_col_[icol].lm;
//** 
//**         for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**         {
//**             o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
//**                               jlo(igkloc, icol - num_gkvec_col()) * conj(gkvec_phase_factors_(igkloc, ia));
//** 
//**             // kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
//**             h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
//**         }
//**     }
//**     t2.stop();
//** 
//**     l_by_lm_.allocate_on_device();
//**     l_by_lm_.copy_to_device();
//** 
//**     mdarray<double_complex, 3> jvlo(parameters_.lmmax_pw(), num_gkvec_row(), num_lo_col());
//**     jvlo.allocate_on_device();
//**     
//**     t3.start();
//**     sbessel_vlo_inner_product_gpu(kargs.get_ptr_device(), num_gkvec_row(), num_lo_col(), parameters_.lmmax_pw(), 
//**                                   l_by_lm_.get_ptr_device(), iat_by_ilo.get_ptr_device(), nmtp_by_iat.get_ptr_device(), 
//**                                   r_dr.get_ptr_device(), sbessel_coefs.get_ptr_device(), vlo_coefs.get_ptr_device(), 
//**                                   jvlo.get_ptr_device());
//**     jvlo.copy_to_host();
//** 
//**     l_by_lm_.deallocate_on_device();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**     {
//**         for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**         {
//**             int ia = gklo_basis_descriptor_col_[icol].ia;
//** 
//**             // add <G+k|V|lo>
//**             double_complex zt1(0, 0);
//**             for (int l = 0; l <= parameters_.lmax_pw(); l++)
//**             {
//**                 for (int m = -l; m <= l; m++)
//**                 {
//**                     int lm = Utils::lm_by_l_m(l, m);
//**                     zt1 += jvlo(lm, igkloc, icol - num_gkvec_col()) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc);
//**                 }
//**             }
//**             zt1 *= ((fourpi / sqrt(parameters_.omega())) * conj(gkvec_phase_factors_(igkloc, ia)));
//**             h(igkloc, icol) += zt1;
//**         }
//**     }
//**     t3.stop();
//**    
//**     // restore the <lo|H|G+k> and <lo|G+k> bocks and exit
//**     if (num_ranks == 1)
//**     {
//**         for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**         {
//**             for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**             {
//**                 h(irow, igkloc) = conj(h(igkloc, irow));
//**                 o(irow, igkloc) = conj(o(igkloc, irow));
//**             }
//**         }
//**         return;
//**     }
//** 
//**     //** // second part: compute <lo|H|G+k> and <lo|G+k>
//** 
//**     //** int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
//**     //** // compute V|lo>
//**     //** t1.start();
//**     //** for (int irow = 0; irow < num_lo_row(); irow++)
//**     //** {
//**     //**     int ia = gklo_basis_descriptor_row_[num_gkvec_row() + irow].ia;
//**     //**     int lm = gklo_basis_descriptor_row_[num_gkvec_row() + irow].lm;
//**     //**     int idxrf = gklo_basis_descriptor_row_[num_gkvec_row() + irow].idxrf;
//**     //**     
//**     //**     for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**     //**     {
//**     //**         svlo(lm1, irow) = new Spline<double_complex>(parameters_.atom(ia)->num_mt_points(), 
//**     //**                                                 parameters_.atom(ia)->radial_grid());
//** 
//**     //**         for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**     //**         {
//**     //**             int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**     //**             double_complex cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**     //**             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**     //**             {
//**     //**                 (*svlo(lm1, irow))[ir] += (cg * effective_potential->f_rlm(lm3, ir, ia) * 
//**     //**                                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**     //**             }
//**     //**         }
//** 
//**     //**         svlo(lm1, irow)->interpolate();
//**     //**     }
//**     //** }
//**     //** t1.stop();
//**    
//**     //** t2.start();
//**     //** for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**     //** {
//**     //**     int ia = gklo_basis_descriptor_row_[irow].ia;
//**     //**     int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**     //**     int l = gklo_basis_descriptor_row_[irow].l;
//**     //**     int lm = gklo_basis_descriptor_row_[irow].lm;
//**     //**     int idxrf = gklo_basis_descriptor_row_[irow].idxrf;
//** 
//**     //**     // compue overlap <lo|G+k>
//**     //**     Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**     //**     for (int ir = 0; ir < slo.num_points(); ir++)
//**     //**         slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**     //**     slo.interpolate();
//**     //**     
//**     //**     #pragma omp parallel for default(shared)
//**     //**     for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**     //**     {
//**     //**         o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**     //**                           conj(gkvec_ylm_(lm, offset_col + igkloc)) * 
//**     //**                           Spline<double>::integrate(&slo, (*sbessel_[offset_col + igkloc])(l, iat)) * 
//**     //**                           gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**     //**         // kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**     //**         h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//**     //**     }
//**     //** }
//**     //** t2.stop();
//** 
//**     //** t3.start();
//**     //** #pragma omp parallel for default(shared)
//**     //** for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**     //** {
//**     //**     for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**     //**     {
//**     //**         int ia = gklo_basis_descriptor_row_[irow].ia;
//**     //**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**     //**         //int l = gklo_basis_descriptor_row_[irow].l;
//**     //**         //int lm = gklo_basis_descriptor_row_[irow].lm;
//**     //**         //int idxrf = gklo_basis_descriptor_row_[irow].idxrf;
//** 
//**     //**         //*// compue overlap <lo|G+k>
//**     //**         //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**     //**         //*for (int ir = 0; ir < s.num_points(); ir++)
//**     //**         //*    s[ir] = (*sbessel_[offset_col + igkloc])(ir, l, iat) * 
//**     //**         //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**     //**         //*s.interpolate();
//**     //**         //*    
//**     //**         //*o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**     //**         //*                  conj(gkvec_ylm_(lm, offset_col + igkloc)) * s.integrate(2) * 
//**     //**         //*                  gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**     //**         //*// kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**     //**         //*h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//** 
//**     //**         // add <lo|V|G+k>
//**     //**         double_complex zt1(0, 0);
//**     //**         for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
//**     //**         {
//**     //**             for (int m1 = -l1; m1 <= l1; m1++)
//**     //**             {
//**     //**                 int lm1 = Utils::lm_by_l_m(l1, m1);
//** 
//**     //**                 zt1 += conj(Spline<double_complex>::integrate(svlo(lm1, irow - num_gkvec_row()), 
//**     //**                                                          (*sbessel_[offset_col + igkloc])(l1, iat))) * 
//**     //**                        zil_[l1] * conj(gkvec_ylm_(lm1, offset_col + igkloc));
//**     //**             }
//**     //**         }
//**     //**         zt1 *= ((fourpi / sqrt(parameters_.omega())) * gkvec_phase_factors_(offset_col + igkloc, ia));
//**     //**         h(irow, igkloc) += zt1;
//**     //**     }
//**     //** }
//**     //** t3.stop();
//**     //** 
//**     //** for (int irow = 0; irow < num_lo_row(); irow++)
//**     //** {
//**     //**     for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, irow);
//**     //** }
//** }
//** 
//**


//=====================================================================================================================
// CPU code, (L)APW+lo basis
//=====================================================================================================================
//== template<> 
//== void Band::set_fv_h_o<cpu, full_potential_lapwlo>(K_point* kp, 
//==                                                   Periodic_function<double>* effective_potential,
//==                                                   dmatrix<double_complex>& h, 
//==                                                   dmatrix<double_complex>& o)
//== {
//==     log_function_enter(__func__);
//==     Timer t("sirius::Band::set_fv_h_o", _global_timer_);
//== 
//==     h.zero();
//==     o.zero();
//== 
//==     int naw = unit_cell_.mt_aw_basis_size();
//== 
//==     /* generate and conjugate panel of matching coefficients; this would be the <bra| states */
//==     dmatrix<double_complex> alm_panel_n;
//==     alm_panel_n.set_dimensions(kp->num_gkvec(), naw, parameters_.blacs_context());
//==     alm_panel_n.allocate(alloc_mode);
//== 
//==     kp->alm_coeffs_row()->generate<true>(alm_panel_n);
//==     for (int j = 0; j < alm_panel_n.num_cols_local(); j++)
//==     {
//==         for (int i = 0; i < alm_panel_n.num_rows_local(); i++) alm_panel_n(i, j) = conj(alm_panel_n(i, j));
//==     }
//== 
//==     /* generate another panel of matching coefficients; this would be the |ket> states */
//==     dmatrix<double_complex> alm_panel_t;
//==     alm_panel_t.set_dimensions(naw, kp->num_gkvec(), parameters_.blacs_context());
//==     alm_panel_t.allocate(alloc_mode);
//==     kp->alm_coeffs_col()->generate<false>(alm_panel_t);
//== 
//==     /* generate slice of matching coefficients */
//==     splindex<block> sspl_gkvec(kp->num_gkvec_col(), kp->num_ranks_row(), kp->rank_row());
//==     mdarray<double_complex, 2> alm_v(naw, sspl_gkvec.local_size());
//==     kp->alm_coeffs_col()->generate(sspl_gkvec, alm_v);
//== 
//==     mdarray<double_complex, 2> halm_v(naw, sspl_gkvec.local_size());
//==     /* apply muffin-tin Hamiltonian to alm slice */
//==     apply_hmt_to_apw<nm>(alm_v, halm_v);
//== 
//==     /* scatter halm slice to panels */
//==     dmatrix<double_complex> halm_panel_t;
//==     halm_panel_t.set_dimensions(naw, kp->num_gkvec(), parameters_.blacs_context());
//==     halm_panel_t.allocate(alloc_mode);
//==     halm_panel_t.scatter(halm_v, parameters_.mpi_grid().communicator(1 << _dim_row_));
//==     
//==     // TODO: this depends on implementation and will possibly be simplified
//==     /* all the above data preparation is done in order to 
//==      * execute "normal" pzgemm without matrix transpose 
//==      * because it gives the best performance
//==      */
//==     Timer t1("sirius::Band::set_fv_h_o|zgemm", _global_timer_);
//==     blas<CPU>::gemm(0, 0, kp->num_gkvec(), kp->num_gkvec(), naw, complex_one, alm_panel_n, halm_panel_t, 
//==                     complex_zero, h); 
//==     
//==     blas<CPU>::gemm(0, 0, kp->num_gkvec(), kp->num_gkvec(), naw, complex_one, alm_panel_n, alm_panel_t, 
//==                     complex_zero, o); 
//==     t1.stop();
//== 
//==     mdarray<double_complex, 2> alm_row(kp->num_gkvec_row(), unit_cell_.max_mt_aw_basis_size());
//==     mdarray<double_complex, 2> alm_col(kp->num_gkvec_col(), unit_cell_.max_mt_aw_basis_size());
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         Atom* atom = unit_cell_.atom(ia);
//==         Atom_type* type = atom->type();
//==        
//==         /* generate matching coefficients for current atom */
//==         kp->alm_coeffs_row()->generate(ia, alm_row);
//==         for (int xi = 0; xi < type->mt_aw_basis_size(); xi++)
//==         {
//==             for (int igk = 0; igk < kp->num_gkvec_row(); igk++) alm_row(igk, xi) = conj(alm_row(igk, xi));
//==         }
//==         kp->alm_coeffs_col()->generate(ia, alm_col);
//==         
//==         /* setup apw-lo and lo-apw blocks */
//==         set_fv_h_o_apw_lo(kp, type, atom, ia, alm_row, alm_col, h.data(), o.data());
//==     }
//==     
//==     /* add interstitial contributon */
//==     set_fv_h_o_it(kp, effective_potential, h.data(), o.data());
//== 
//==     /* setup lo-lo block */
//==     set_fv_h_o_lo_lo(kp, h.data(), o.data());
//== 
//==     log_function_exit(__func__);
//==     kp->comm().barrier();
//== }

//== void K_point::solve_fv_evp_2stage(mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
//== {
//==     if (parameters_.eigen_value_solver() != lapack) error_local(__FILE__, __LINE__, "implemented for LAPACK only");
//==     
//==     standard_evp_lapack s;
//== 
//==     std::vector<double> o_eval(apwlo_basis_size());
//==     
//==     mdarray<double_complex, 2> o_tmp(apwlo_basis_size(), apwlo_basis_size());
//==     memcpy(o_tmp.get_ptr(), o.get_ptr(), o.size() * sizeof(double_complex));
//==     mdarray<double_complex, 2> o_evec(apwlo_basis_size(), apwlo_basis_size());
//==  
//==     s.solve(apwlo_basis_size(), o_tmp.get_ptr(), o_tmp.ld(), &o_eval[0], o_evec.get_ptr(), o_evec.ld());
//== 
//==     int num_dependent_apwlo = 0;
//==     for (int i = 0; i < apwlo_basis_size(); i++) 
//==     {
//==         if (fabs(o_eval[i]) < 1e-4) 
//==         {
//==             num_dependent_apwlo++;
//==         }
//==         else
//==         {
//==             o_eval[i] = 1.0 / sqrt(o_eval[i]);
//==         }
//==     }
//== 
//==     //std::cout << "num_dependent_apwlo = " << num_dependent_apwlo << std::endl;
//== 
//==     mdarray<double_complex, 2> h_tmp(apwlo_basis_size(), apwlo_basis_size());
//==     // compute h_tmp = Z^{h.c.} * H
//==     blas<CPU>::gemm(2, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), o_evec.get_ptr(), 
//==                     o_evec.ld(), h.get_ptr(), h.ld(), h_tmp.get_ptr(), h_tmp.ld());
//==     // compute \tilda H = Z^{h.c.} * H * Z = h_tmp * Z
//==     blas<CPU>::gemm(0, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), h_tmp.get_ptr(), 
//==                     h_tmp.ld(), o_evec.get_ptr(), o_evec.ld(), h.get_ptr(), h.ld());
//== 
//==     int reduced_apwlo_basis_size = apwlo_basis_size() - num_dependent_apwlo;
//==     
//==     for (int i = 0; i < reduced_apwlo_basis_size; i++)
//==     {
//==         for (int j = 0; j < reduced_apwlo_basis_size; j++)
//==         {
//==             double d = o_eval[num_dependent_apwlo + j] * o_eval[num_dependent_apwlo + i];
//==             h(num_dependent_apwlo + j, num_dependent_apwlo + i) *= d;
//==         }
//==     }
//== 
//==     std::vector<double> h_eval(reduced_apwlo_basis_size);
//==     s.solve(reduced_apwlo_basis_size, &h(num_dependent_apwlo, num_dependent_apwlo), h.ld(), &h_eval[0], 
//==             h_tmp.get_ptr(), h_tmp.ld());
//== 
//==     for (int i = 0; i < reduced_apwlo_basis_size; i++)
//==     {
//==         for (int j = 0; j < reduced_apwlo_basis_size; j++) h_tmp(j, i) *= o_eval[num_dependent_apwlo + j];
//==     }
//== 
//==     for (int i = 0; i < parameters_.num_fv_states(); i++) fv_eigen_values_[i] = h_eval[i];
//== 
//==     blas<CPU>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), reduced_apwlo_basis_size, 
//==                     &o_evec(0, num_dependent_apwlo), o_evec.ld(), h_tmp.get_ptr(), h_tmp.ld(), 
//==                     fv_eigen_vectors_.get_ptr(), fv_eigen_vectors_.ld());
//== }

//void Band::set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
//                        mdarray<double_complex, 2>& o)
//{
//    Timer t("sirius::Band::set_o_apw_lo");
//    
//    int apw_offset_col = kp->apw_offset_col();
//    
//    // apw-lo block
//    for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
//    {
//        int icol = kp->lo_col(ia, i);
//
//        int l = kp->gklo_basis_descriptor_col(icol).l;
//        int lm = kp->gklo_basis_descriptor_col(icol).lm;
//        int order = kp->gklo_basis_descriptor_col(icol).order;
//        
//        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
//        {
//            for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
//            {
//                o(igkloc, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order) * 
//                                   alm(igkloc, type->indexb_by_lm_order(lm, order1));
//            }
//        }
//    }
//
//    std::vector<double_complex> ztmp(kp->num_gkvec_col());
//    // lo-apw block
//    for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
//    {
//        int irow = kp->lo_row(ia, i);
//
//        int l = kp->gklo_basis_descriptor_row(irow).l;
//        int lm = kp->gklo_basis_descriptor_row(irow).lm;
//        int order = kp->gklo_basis_descriptor_row(irow).order;
//
//        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
//        {
//            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
//            {
//                o(irow, igkloc) += atom->symmetry_class()->o_radial_integral(l, order, order1) * 
//                                   conj(alm(apw_offset_col + igkloc, type->indexb_by_lm_order(lm, order1)));
//            }
//        }
//    }
//}

void Band::set_o_it(K_point* kp, mdarray<double_complex, 2>& o)
{
    runtime::Timer t("sirius::Band::set_o_it");

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            int ig12 = ctx_.gvec().index_g12(kp->gklo_basis_descriptor_row(igk_row).gvec,
                                             kp->gklo_basis_descriptor_col(igk_col).gvec);
            
            o(igk_row, igk_col) += ctx_.step_function().theta_pw(ig12);
        }
    }
}

void Band::set_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& o)
{
    runtime::Timer t("sirius::Band::set_o_lo_lo");

    // lo-lo block
    #pragma omp parallel for default(shared)
    for (int icol = kp->num_gkvec_col(); icol < kp->gklo_basis_size_col(); icol++)
    {
        int ia = kp->gklo_basis_descriptor_col(icol).ia;
        int lm2 = kp->gklo_basis_descriptor_col(icol).lm; 

        for (int irow = kp->num_gkvec_row(); irow < kp->gklo_basis_size_row(); irow++)
        {
            if (ia == kp->gklo_basis_descriptor_row(irow).ia)
            {
                auto& atom = unit_cell_.atom(ia);
                int lm1 = kp->gklo_basis_descriptor_row(irow).lm; 

                if (lm1 == lm2)
                {
                    int l = kp->gklo_basis_descriptor_row(irow).l;
                    int order1 = kp->gklo_basis_descriptor_row(irow).order; 
                    int order2 = kp->gklo_basis_descriptor_col(icol).order; 
                    o(irow, icol) += atom.symmetry_class().o_radial_integral(l, order1, order2);
                }
            }
        }
    }
}

//== void Band::set_o(K_point* kp, mdarray<double_complex, 2>& o)
//== {
//==     Timer t("sirius::Band::set_o");
//==    
//==     // index of column apw coefficients in apw array
//==     int apw_offset_col = kp->apw_offset_col();
//==     
//==     mdarray<double_complex, 2> alm(kp->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
//==     o.zero();
//== 
//==     double_complex zone(1, 0);
//==     
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         Atom* atom = unit_cell_.atom(ia);
//==         Atom_type* type = atom->type();
//==        
//==         // generate conjugated coefficients
//==         kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
//==         
//==         // generate <apw|apw> block; |ket> is conjugated, so it is "unconjugated" back
//==         blas<CPU>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), zone, 
//==                         &alm(0, 0), alm.ld(), &alm(apw_offset_col, 0), alm.ld(), zone, &o(0, 0), o.ld()); 
//==             
//==         // setup apw-lo blocks
//==         set_o_apw_lo(kp, type, atom, ia, alm, o);
//==     } //ia
//== 
//==     set_o_it(kp, o);
//== 
//==     set_o_lo_lo(kp, o);
//== 
//==     alm.deallocate();
//== }

void Band::solve_fv(K_point* kp__, Periodic_function<double>* effective_potential__)
{
    if (kp__->gklo_basis_size() < ctx_.num_fv_states()) TERMINATE("basis size is too small");

    switch (ctx_.esm_type())
    {
        case full_potential_pwlo:
        case full_potential_lapwlo:
        {
            diag_fv_full_potential(kp__, effective_potential__);
            break;
        }
        default:
        {
            TERMINATE_NOT_IMPLEMENTED
        }
    }
}

void Band::solve_fd(K_point* kp__,
                    Periodic_function<double>* effective_potential__, 
                    Periodic_function<double>* effective_magnetic_field__[3])
{
    switch (ctx_.esm_type())
    {
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            if (ctx_.gamma_point())
            {
                diag_pseudo_potential<double>(kp__, effective_potential__, effective_magnetic_field__);
            }
            else
            {
                diag_pseudo_potential<double_complex>(kp__, effective_potential__, effective_magnetic_field__);
            }
            break;
        }
        default:
        {
            TERMINATE_NOT_IMPLEMENTED
        }
    }
    //== if (kp->num_ranks() > 1 && !parameters_.gen_evp_solver()->parallel())
    //==     error_local(__FILE__, __LINE__, "eigen-value solver is not parallel");

    //== mdarray<double_complex, 2> h(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
    //== mdarray<double_complex, 2> o(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
    //== 
    //== set_o(kp, o);

    //== std::vector<double> eval(parameters_.num_bands());
    //== mdarray<double_complex, 2>& fd_evec = kp->fd_eigen_vectors();

    //== if (parameters_.num_mag_dims() == 0)
    //== {
    //==     assert(kp->gklo_basis_size() >= parameters_.num_fv_states());
    //==     set_h<nm>(kp, effective_potential, effective_magnetic_field, h);
    //==    
    //==     Timer t2("sirius::Band::solve_fd|diag");
    //==     parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(), 
    //==                                         parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(), 
    //==                                         &eval[0], fd_evec.ptr(), fd_evec.ld());
    //== }
    //== 
    //== if (parameters_.num_mag_dims() == 1)
    //== {
    //==     assert(kp->gklo_basis_size() >= parameters_.num_fv_states());

    //==     mdarray<double_complex, 2> o1(kp->gklo_basis_size_row(), kp->gklo_basis_size_col());
    //==     memcpy(&o1(0, 0), &o(0, 0), o.size() * sizeof(double_complex));

    //==     set_h<uu>(kp, effective_potential, effective_magnetic_field, h);
    //==    
    //==     Timer t2("sirius::Band::solve_fd|diag");
    //==     parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(), 
    //==                                         parameters_.num_fv_states(), h.ptr(), h.ld(), o.ptr(), o.ld(), 
    //==                                         &eval[0], &fd_evec(0, 0), fd_evec.ld());
    //==     t2.stop();

    //==     set_h<dd>(kp, effective_potential, effective_magnetic_field, h);
    //==     
    //==     t2.start();
    //==     parameters_.gen_evp_solver()->solve(kp->gklo_basis_size(), kp->gklo_basis_size_row(), kp->gklo_basis_size_col(), 
    //==                                         parameters_.num_fv_states(), h.ptr(), h.ld(), o1.ptr(), o1.ld(), 
    //==                                         &eval[parameters_.num_fv_states()], 
    //==                                         &fd_evec(0, parameters_.spl_fv_states().local_size()), fd_evec.ld());
    //==     t2.stop();
    //== }

    //== kp->set_band_energies(&eval[0]);
}

}


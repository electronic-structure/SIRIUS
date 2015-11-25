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

/** \file k_point.cpp
 *   
 *  \brief Contains remaining implementation of sirius::K_point class.
 */

#include "k_point.h"
#include "error_handling.h"

namespace sirius {

K_point::K_point(Simulation_context& ctx__,
                 double* vk__,
                 double weight__,
                 BLACS_grid const& blacs_grid__,
                 BLACS_grid const& blacs_grid_slab__,
                 BLACS_grid const& blacs_grid_slice__)
    : ctx_(ctx__),
      parameters_(ctx__.parameters()),
      unit_cell_(ctx_.unit_cell()),
      blacs_grid_(blacs_grid__),
      blacs_grid_slice_(blacs_grid_slice__),
      weight_(weight__),
      fv_states_(nullptr),
      alm_coeffs_row_(nullptr),
      alm_coeffs_col_(nullptr),
      alm_coeffs_(nullptr),
      beta_projectors_(nullptr),
      comm_(blacs_grid_.comm()),
      comm_row_(blacs_grid_.comm_row()),
      comm_col_(blacs_grid_.comm_col())
{
    PROFILE();

    for (int x = 0; x < 3; x++) vk_[x] = vk__[x];
    
    band_occupancies_ = std::vector<double>(parameters_.num_bands(), 1);
    band_energies_    = std::vector<double>(parameters_.num_bands(), 0);
    
    num_ranks_row_ = comm_row_.size();
    num_ranks_col_ = comm_col_.size();
    
    rank_row_ = comm_row_.rank();
    rank_col_ = comm_col_.rank();

    //if (comm_.rank() != blacs_grid_slab_.comm().rank()) TERMINATE("ranks don't match");
    if (comm_.rank() != blacs_grid_slice_.comm().rank()) TERMINATE("ranks don't match");
    
    iterative_solver_input_section_ = parameters_.iterative_solver_input_section();

    #ifndef __GPU
    if (parameters_.processing_unit() == GPU) TERMINATE_NO_GPU
    #endif

    spinor_wave_functions_[0] = nullptr;
    spinor_wave_functions_[1] = nullptr;
}

//== void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
//== {
//==     static SHT* sht = NULL;
//==     if (!sht) sht = new SHT(parameters_.lmax_apw());
//== 
//==     Atom* atom = unit_cell_.atom(ia);
//==     Atom_type* type = atom->type();
//== 
//==     mdarray<double_complex, 2> z1(sht->num_points(), type->mt_aw_basis_size());
//==     for (int i = 0; i < type->mt_aw_basis_size(); i++)
//==     {
//==         int lm = type->indexb(i).lm;
//==         int idxrf = type->indexb(i).idxrf;
//==         double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
//==         }
//==     }
//== 
//==     mdarray<double_complex, 2> z2(sht->num_points(), num_gkvec_loc);
//==     blas<CPU>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.ptr(), z1.ld(),
//==                     alm.ptr(), alm.ld(), z2.ptr(), z2.ld());
//== 
//==     vector3d<double> vc = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia)->position());
//==     
//==     double tdiff = 0;
//==     for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
//==     {
//==         vector3d<double> gkc = gkvec_cart(igkglob(igloc));
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             double_complex aw_value = z2(itp, igloc);
//==             vector3d<double> r;
//==             for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
//==             double_complex pw_value = exp(double_complex(0, Utils::scalar_product(r, gkc))) / sqrt(unit_cell_.omega());
//==             tdiff += abs(pw_value - aw_value);
//==         }
//==     }
//== 
//==     printf("atom : %i  absolute alm error : %e  average alm error : %e\n", 
//==            ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
//== }


//Periodic_function<double_complex>* K_point::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::K_point::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    Periodic_function<double_complex, index_order>* func = 
//        new Periodic_function<double_complex, index_order>(parameters_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//    
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(parameters_.omega());
//        
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            double_complex z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//            {
//                int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//                
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//                                parameters_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//                            &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, jloc));
//    parameters_.fft().transform(1);
//    parameters_.fft().output(func->f_it());
//
//    for (int i = 0; i < parameters_.fft().size(); i++) func->f_it(i) /= sqrt(parameters_.omega());
//    
//    return func;
//}

//== void K_point::spinor_wave_function_component_mt(int lmax, int ispn, int jloc, mt_functions<double_complex>& psilm)
//== {
//==     Timer t("sirius::K_point::spinor_wave_function_component_mt");
//== 
//==     //int lmmax = Utils::lmmax_by_lmax(lmax);
//== 
//==     psilm.zero();
//==     
//==     //if (basis_type == pwlo)
//==     //{
//==     //    if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//== 
//==     //    double fourpi_omega = fourpi / sqrt(parameters_.omega());
//== 
//==     //    mdarray<double_complex, 2> zm(parameters_.max_num_mt_points(),  num_gkvec_row());
//== 
//==     //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     //    {
//==     //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//==     //        for (int l = 0; l <= lmax; l++)
//==     //        {
//==     //            #pragma omp parallel for default(shared)
//==     //            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //            {
//==     //                int igk = igkglob(igkloc);
//==     //                double_complex z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//==     //                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
//==     //                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==     //                    zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
//==     //            }
//==     //            blas<CPU>::gemm(0, 2, parameters_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
//==     //                            &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(), 
//==     //                            &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
//==     //        }
//==     //    }
//==     //    //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //    //{
//==     //    //    int igk = igkglob(igkloc);
//==     //    //    double_complex z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//== 
//==     //    //    // TODO: possilbe optimization with zgemm
//==     //    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     //    //    {
//==     //    //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//==     //    //        double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//==     //    //        
//==     //    //        #pragma omp parallel for default(shared)
//==     //    //        for (int lm = 0; lm < lmmax; lm++)
//==     //    //        {
//==     //    //            int l = l_by_lm_(lm);
//==     //    //            double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//==     //    //            for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==     //    //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//==     //    //        }
//==     //    //    }
//==     //    //}
//== 
//==     //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     //    {
//==     //        Platform::allreduce(&fylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//==     //                            parameters_.mpi_grid().communicator(1 << band->dim_row()));
//==     //    }
//==     //}
//== 
//==     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     {
//==         for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//==         {
//==             int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//==             int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//==             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==             {
//==                 psilm(lm, ir, ia) += 
//==                     spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//==                     parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//==             }
//==         }
//==     }
//== }

void K_point::test_spinor_wave_functions(int use_fft)
{
    STOP();

//==     if (num_ranks() > 1) error_local(__FILE__, __LINE__, "test of spinor wave functions on multiple ranks is not implemented");
//== 
//==     std::vector<double_complex> v1[2];
//==     std::vector<double_complex> v2;
//== 
//==     if (use_fft == 0 || use_fft == 1) v2.resize(fft_->size());
//==     
//==     if (use_fft == 0) 
//==     {
//==         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++) v1[ispn].resize(num_gkvec());
//==     }
//==     
//==     if (use_fft == 1) 
//==     {
//==         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++) v1[ispn].resize(fft_->size());
//==     }
//==     
//==     double maxerr = 0;
//== 
//==     for (int j1 = 0; j1 < parameters_.num_bands(); j1++)
//==     {
//==         if (use_fft == 0)
//==         {
//==             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//==             {
//==                 fft_->input(num_gkvec(), gkvec_.index_map(),
//==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
//==                 fft_->transform(1);
//==                 fft_->output(&v2[0]);
//== 
//==                 for (int ir = 0; ir < fft_->size(); ir++) v2[ir] *= ctx_.step_function()->theta_r(ir);
//==                 
//==                 fft_->input(&v2[0]);
//==                 fft_->transform(-1);
//==                 fft_->output(num_gkvec(), gkvec_.index_map(), &v1[ispn][0]); 
//==             }
//==         }
//==         
//==         if (use_fft == 1)
//==         {
//==             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//==             {
//==                 fft_->input(num_gkvec(), gkvec_.index_map(),
//==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
//==                 fft_->transform(1);
//==                 fft_->output(&v1[ispn][0]);
//==             }
//==         }
//==        
//==         for (int j2 = 0; j2 < parameters_.num_bands(); j2++)
//==         {
//==             double_complex zsum(0, 0);
//==             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//==             {
//==                 for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==                 {
//==                     int offset_wf = unit_cell_.atom(ia)->offset_wf();
//==                     Atom_type* type = unit_cell_.atom(ia)->type();
//==                     Atom_symmetry_class* symmetry_class = unit_cell_.atom(ia)->symmetry_class();
//== 
//==                     for (int l = 0; l <= parameters_.lmax_apw(); l++)
//==                     {
//==                         int ordmax = type->indexr().num_rf(l);
//==                         for (int io1 = 0; io1 < ordmax; io1++)
//==                         {
//==                             for (int io2 = 0; io2 < ordmax; io2++)
//==                             {
//==                                 for (int m = -l; m <= l; m++)
//==                                 {
//==                                     zsum += conj(spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io1), ispn, j1)) *
//==                                             spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io2), ispn, j2) * 
//==                                             symmetry_class->o_radial_integral(l, io1, io2);
//==                                 }
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//==             
//==             if (use_fft == 0)
//==             {
//==                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//==                {
//==                    for (int ig = 0; ig < num_gkvec(); ig++)
//==                        zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(unit_cell_.mt_basis_size() + ig, ispn, j2);
//==                }
//==             }
//==            
//==             if (use_fft == 1)
//==             {
//==                 for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//==                 {
//==                     fft_->input(num_gkvec(), gkvec_.index_map(), &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j2));
//==                     fft_->transform(1);
//==                     fft_->output(&v2[0]);
//== 
//==                     for (int ir = 0; ir < fft_->size(); ir++)
//==                         zsum += std::conj(v1[ispn][ir]) * v2[ir] * ctx_.step_function()->theta_r(ir) / double(fft_->size());
//==                 }
//==             }
//==             
//==             if (use_fft == 2) 
//==             {
//==                 STOP();
//==                 //for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
//==                 //{
//==                 //    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
//==                 //    {
//==                 //        int ig3 = ctx_.gvec().index_g12(ig1, ig2);
//==                 //        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//==                 //        {
//==                 //            zsum += std::conj(spinor_wave_functions_(unit_cell_.mt_basis_size() + ig1, ispn, j1)) * 
//==                 //                    spinor_wave_functions_(unit_cell_.mt_basis_size() + ig2, ispn, j2) * 
//==                 //                    ctx_.step_function()->theta_pw(ig3);
//==                 //        }
//==                 //    }
//==                 //}
//==             }
//== 
//==             zsum = (j1 == j2) ? zsum - double_complex(1.0, 0.0) : zsum;
//==             maxerr = std::max(maxerr, std::abs(zsum));
//==         }
//==     }
//==     std :: cout << "maximum error = " << maxerr << std::endl;
}

void K_point::save(int id)
{
    if (num_ranks() > 1) error_local(__FILE__, __LINE__, "writing of distributed eigen-vectors is not implemented");

    STOP();

    //if (parameters_.mpi_grid().root(1 << _dim_col_))
    //{
    //    HDF5_tree fout(storage_file_name, false);

    //    fout["K_set"].create_node(id);
    //    fout["K_set"][id].create_node("spinor_wave_functions");
    //    fout["K_set"][id].write("coordinates", &vk_[0], 3);
    //    fout["K_set"][id].write("band_energies", band_energies_);
    //    fout["K_set"][id].write("band_occupancies", band_occupancies_);
    //    if (num_ranks() == 1)
    //    {
    //        fout["K_set"][id].write("fv_eigen_vectors", fv_eigen_vectors_panel_.data());
    //        fout["K_set"][id].write("sv_eigen_vectors", sv_eigen_vectors_[0].data());
    //    }
    //}
    //
    //comm_col_.barrier();
    //
    //mdarray<double_complex, 2> wfj(NULL, wf_size(), parameters_.num_spins()); 
    //for (int j = 0; j < parameters_.num_bands(); j++)
    //{
    //    int rank = parameters_.spl_spinor_wf().local_rank(j);
    //    int offs = (int)parameters_.spl_spinor_wf().local_index(j);
    //    if (parameters_.mpi_grid().coordinate(_dim_col_) == rank)
    //    {
    //        HDF5_tree fout(storage_file_name, false);
    //        wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
    //        fout["K_set"][id]["spinor_wave_functions"].write(j, wfj);
    //    }
    //    comm_col_.barrier();
    //}
}

void K_point::load(HDF5_tree h5in, int id)
{
    STOP();
    //== band_energies_.resize(parameters_.num_bands());
    //== h5in[id].read("band_energies", band_energies_);

    //== band_occupancies_.resize(parameters_.num_bands());
    //== h5in[id].read("band_occupancies", band_occupancies_);
    //== 
    //== h5in[id].read_mdarray("fv_eigen_vectors", fv_eigen_vectors_panel_);
    //== h5in[id].read_mdarray("sv_eigen_vectors", sv_eigen_vectors_);
}

//== void K_point::save_wave_functions(int id)
//== {
//==     if (parameters_.mpi_grid().root(1 << _dim_col_))
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//== 
//==         fout["K_points"].create_node(id);
//==         fout["K_points"][id].write("coordinates", &vk_[0], 3);
//==         fout["K_points"][id].write("mtgk_size", mtgk_size());
//==         fout["K_points"][id].create_node("spinor_wave_functions");
//==         fout["K_points"][id].write("band_energies", &band_energies_[0], parameters_.num_bands());
//==         fout["K_points"][id].write("band_occupancies", &band_occupancies_[0], parameters_.num_bands());
//==     }
//==     
//==     Platform::barrier(parameters_.mpi_grid().communicator(1 << _dim_col_));
//==     
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
//==     for (int j = 0; j < parameters_.num_bands(); j++)
//==     {
//==         int rank = parameters_.spl_spinor_wf_col().location(_splindex_rank_, j);
//==         int offs = parameters_.spl_spinor_wf_col().location(_splindex_offs_, j);
//==         if (parameters_.mpi_grid().coordinate(_dim_col_) == rank)
//==         {
//==             HDF5_tree fout(storage_file_name, false);
//==             wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
//==             fout["K_points"][id]["spinor_wave_functions"].write_mdarray(j, wfj);
//==         }
//==         Platform::barrier(parameters_.mpi_grid().communicator(_dim_col_));
//==     }
//== }
//== 
//== void K_point::load_wave_functions(int id)
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==     
//==     int mtgk_size_in;
//==     fin["K_points"][id].read("mtgk_size", &mtgk_size_in);
//==     if (mtgk_size_in != mtgk_size()) error_local(__FILE__, __LINE__, "wrong wave-function size");
//== 
//==     band_energies_.resize(parameters_.num_bands());
//==     fin["K_points"][id].read("band_energies", &band_energies_[0], parameters_.num_bands());
//== 
//==     band_occupancies_.resize(parameters_.num_bands());
//==     fin["K_points"][id].read("band_occupancies", &band_occupancies_[0], parameters_.num_bands());
//== 
//==     spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), 
//==                                           parameters_.spl_spinor_wf_col().local_size());
//==     spinor_wave_functions_.allocate();
//== 
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
//==     for (int jloc = 0; jloc < parameters_.spl_spinor_wf_col().local_size(); jloc++)
//==     {
//==         int j = parameters_.spl_spinor_wf_col(jloc);
//==         wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
//==         fin["K_points"][id]["spinor_wave_functions"].read_mdarray(j, wfj);
//==     }
//== }

void K_point::get_fv_eigen_vectors(mdarray<double_complex, 2>& fv_evec)
{
    assert((int)fv_evec.size(0) >= gklo_basis_size());
    assert((int)fv_evec.size(1) == parameters_.num_fv_states());
    
    fv_evec.zero();
    STOP();

    //for (int iloc = 0; iloc < (int)spl_fv_states_.local_size(); iloc++)
    //{
    //    int i = (int)spl_fv_states_[iloc];
    //    for (int jloc = 0; jloc < gklo_basis_size_row(); jloc++)
    //    {
    //        int j = gklo_basis_descriptor_row(jloc).id;
    //        fv_evec(j, i) = fv_eigen_vectors_(jloc, iloc);
    //    }
    //}
    //comm_.allreduce(fv_evec.at<CPU>(), (int)fv_evec.size());
}

void K_point::get_sv_eigen_vectors(mdarray<double_complex, 2>& sv_evec)
{
    assert((int)sv_evec.size(0) == parameters_.num_bands());
    assert((int)sv_evec.size(1) == parameters_.num_bands());

    sv_evec.zero();

    if (!parameters_.need_sv())
    {
        for (int i = 0; i < parameters_.num_fv_states(); i++) sv_evec(i, i) = complex_one;
        return;
    }

    int nsp = (parameters_.num_mag_dims() == 3) ? 1 : parameters_.num_spins();

    for (int ispn = 0; ispn < nsp; ispn++)
    {
        int offs = parameters_.num_fv_states() * ispn;
        for (int jloc = 0; jloc < sv_eigen_vectors_[ispn].num_cols_local(); jloc++)
        {
            int j = sv_eigen_vectors_[ispn].icol(jloc);
            for (int iloc = 0; iloc < sv_eigen_vectors_[ispn].num_rows_local(); iloc++)
            {
                int i = sv_eigen_vectors_[ispn].irow(iloc);
                sv_evec(i + offs, j + offs) = sv_eigen_vectors_[ispn](iloc, jloc);
            }
        }
    }

    comm_.allreduce(sv_evec.at<CPU>(), (int)sv_evec.size());
}

}


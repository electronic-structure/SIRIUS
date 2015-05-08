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

namespace sirius {

K_point::K_point(Global& parameters__,
                 double* vk__,
                 double weight__,
                 BLACS_grid const& blacs_grid__) 
    : parameters_(parameters__), 
      blacs_grid_(blacs_grid__),
      weight_(weight__),
      alm_coeffs_row_(nullptr),
      alm_coeffs_col_(nullptr)
{
    for (int x = 0; x < 3; x++) vk_[x] = vk__[x];
    
    band_occupancies_ = std::vector<double>(parameters_.num_bands(), 1);
    band_energies_ = std::vector<double>(parameters_.num_bands(), 0);
    
    comm_ = blacs_grid_.comm();
    comm_row_ = blacs_grid_.comm_row();
    comm_col_ = blacs_grid_.comm_col();
    
    num_ranks_ = comm_.size();
    num_ranks_row_ = comm_row_.size();
    num_ranks_col_ = comm_col_.size();
    
    rank_row_ = comm_row_.rank();
    rank_col_ = comm_col_.rank();
    
    fft_ = parameters_.fft();
    
    /* distribue first-variational states along columns */
    spl_fv_states_ = splindex<block_cyclic>(parameters_.num_fv_states(), num_ranks_col_, rank_col_, blacs_grid_.cyclic_block_size());
    
    /* distribue spinor wave-functions along columns */
    spl_spinor_wf_ = splindex<block_cyclic>(parameters_.num_bands(), num_ranks_col_, rank_col_, blacs_grid_.cyclic_block_size());
    
    /* additionally split along rows */
    sub_spl_fv_states_ = splindex<block>(spl_fv_states_.local_size(), num_ranks_row_, rank_row_);
    sub_spl_spinor_wf_ = splindex<block>(spl_spinor_wf_.local_size(), num_ranks_row_, rank_row_);
    
    iterative_solver_input_section_ = parameters_.iterative_solver_input_section_;
}

void K_point::initialize()
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::K_point::initialize");
    
    zil_.resize(parameters_.lmax_apw() + 1);
    for (int l = 0; l <= parameters_.lmax_apw(); l++) zil_[l] = pow(double_complex(0, 1), l);
   
    l_by_lm_ = Utils::l_by_lm(parameters_.lmax_apw());

    if (use_second_variation) fv_eigen_values_.resize(parameters_.num_fv_states());

    if (use_second_variation && parameters_.need_sv())
    {
        /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix */
        if (parameters_.num_mag_dims() == 3)
        {
            sv_eigen_vectors_[0] = dmatrix<double_complex>(parameters_.num_bands(), parameters_.num_bands(), blacs_grid_);
        }
        else
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                sv_eigen_vectors_[ispn] = dmatrix<double_complex>(parameters_.num_fv_states(), parameters_.num_fv_states(), blacs_grid_);
            }
        }
    }
    
    update();

    LOG_FUNC_END();
}

void K_point::update()
{
    LOG_FUNC_BEGIN();

    double gk_cutoff = 0;
    switch (parameters_.esm_type())
    {
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            gk_cutoff = parameters_.gk_cutoff();
            break;
        }
        case full_potential_lapwlo:
        {
            gk_cutoff = parameters_.aw_cutoff() / parameters_.unit_cell()->min_mt_radius();
            break;
        }
        default:
        {
            STOP();
        }
    }

    /* Build a full list of G+k vectors for all MPI ranks */
    generate_gkvec(gk_cutoff);
    
    /* build a list of basis functions */
    build_gklo_basis_descriptors();

    /* distribute basis functions */
    distribute_basis_index();
    
    if (parameters_.esm_type() == full_potential_lapwlo || parameters_.esm_type() == full_potential_pwlo)
    {
        atom_lo_cols_.clear();
        atom_lo_cols_.resize(parameters_.unit_cell()->num_atoms());

        atom_lo_rows_.clear();
        atom_lo_rows_.resize(parameters_.unit_cell()->num_atoms());

        for (int icol = num_gkvec_col(); icol < gklo_basis_size_col(); icol++)
        {
            int ia = gklo_basis_descriptor_col(icol).ia;
            atom_lo_cols_[ia].push_back(icol);
        }
        
        for (int irow = num_gkvec_row(); irow < gklo_basis_size_row(); irow++)
        {
            int ia = gklo_basis_descriptor_row(irow).ia;
            atom_lo_rows_[ia].push_back(irow);
        }
    }
    if (parameters_.esm_type() == full_potential_pwlo)
    {
        /** \todo Correct the memory leak */
        STOP();
        //== sbessel_.resize(num_gkvec_loc()); 
        //== for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        //== {
        //==     sbessel_[igkloc] = new sbessel_pw<double>(parameters_.unit_cell(), parameters_.lmax_pw());
        //==     sbessel_[igkloc]->interpolate(gkvec_len_[igkloc]);
        //== }
    }
    
    init_gkvec();
    
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        if (alm_coeffs_row_) delete alm_coeffs_row_;
        alm_coeffs_row_ = new Matching_coefficients(parameters_, num_gkvec_row(), gklo_basis_descriptors_row_);

        if (alm_coeffs_col_) delete alm_coeffs_col_;
        alm_coeffs_col_ = new Matching_coefficients(parameters_, num_gkvec_col(), gklo_basis_descriptors_col_);
    }

    /* compute |beta> projectors for atom types */
    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        Timer t1("sirius::K_point::update|beta_pw");
        
        std::vector<std::pair<double, std::vector<int> > > gkvec_shells_;
        
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++)
        {
            int igk = gklo_basis_descriptors_row_[igk_loc].igk;
            double gk_len = gkvec<cartesian>(igk).length();

            if (gkvec_shells_.empty() || std::abs(gkvec_shells_.back().first - gk_len) > 1e-10) 
                gkvec_shells_.push_back(std::pair<double, std::vector<int> >(gk_len, std::vector<int>()));
            gkvec_shells_.back().second.push_back(igk_loc);
        }

        auto uc = parameters_.unit_cell();

        beta_gk_t_ = matrix<double_complex>(num_gkvec_loc(), uc->num_beta_t()); 

        mdarray<Spline<double>*, 2> beta_rf(uc->max_mt_radial_basis_size(), uc->num_atom_types());
        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {
            auto atom_type = uc->atom_type(iat);
            for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            {
                int nr = atom_type->uspp().num_beta_radial_points[idxrf];
                beta_rf(idxrf, iat) = new Spline<double>(atom_type->radial_grid());
                for (int ir = 0; ir < nr; ir++) 
                    (*beta_rf(idxrf, iat))[ir] = atom_type->uspp().beta_radial_functions(ir, idxrf);
                beta_rf(idxrf, iat)->interpolate();
            }
        }

        #pragma omp parallel
        {
            std::vector<double> gkvec_rlm(Utils::lmmax(parameters_.lmax_beta()));
            std::vector<double> beta_radial_integrals_(uc->max_mt_radial_basis_size());
            sbessel_pw<double> jl(uc, parameters_.lmax_beta());
            #pragma omp for
            for (int ish = 0; ish < (int)gkvec_shells_.size(); ish++)
            {
                jl.interpolate(gkvec_shells_[ish].first);
                for (int i = 0; i < (int)gkvec_shells_[ish].second.size(); i++)
                {
                    int igk_loc = gkvec_shells_[ish].second[i];
                    int igk = gklo_basis_descriptors_row_[igk_loc].igk;
                    /* vs = {r, theta, phi} */
                    auto vs = SHT::spherical_coordinates(gkvec<cartesian>(igk));
                    SHT::spherical_harmonics(parameters_.lmax_beta(), vs[1], vs[2], &gkvec_rlm[0]);

                    for (int iat = 0; iat < uc->num_atom_types(); iat++)
                    {
                        auto atom_type = uc->atom_type(iat);
                        for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
                        {
                            int l = atom_type->indexr(idxrf).l;
                            int nr = atom_type->uspp().num_beta_radial_points[idxrf];
                            beta_radial_integrals_[idxrf] = inner(*jl(l, iat), *beta_rf(idxrf, iat), 1, nr);
                        }

                        for (int xi = 0; xi < atom_type->mt_basis_size(); xi++)
                        {
                            int l = atom_type->indexb(xi).l;
                            int lm = atom_type->indexb(xi).lm;
                            int idxrf = atom_type->indexb(xi).idxrf;

                            double_complex z = pow(double_complex(0, -1), l) * fourpi / sqrt(parameters_.unit_cell()->omega());
                            beta_gk_t_(igk_loc, atom_type->offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals_[idxrf];
                        }
                    }
                }
            }
        }

        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {
            auto atom_type = uc->atom_type(iat);
            for (int idxrf = 0; idxrf < atom_type->mt_radial_basis_size(); idxrf++)
            {
                delete beta_rf(idxrf, iat);
            }
        }

        beta_gk_ = matrix<double_complex>(num_gkvec_loc(), uc->mt_basis_size());

        for (int i = 0; i < uc->mt_basis_size(); i++)
        {
            int ia = uc->mt_lo_basis_descriptor(i).ia;
            int xi = uc->mt_lo_basis_descriptor(i).xi;

            auto atom_type = parameters_.unit_cell()->atom(ia)->type();

            for (int igk = 0; igk < num_gkvec_loc(); igk++)
            {
                beta_gk_(igk, i) = beta_gk_t_(igk, atom_type->offset_lo() + xi) * conj(gkvec_phase_factors_(igk, ia));
            }
        }

        p_mtrx_ = mdarray<double_complex, 3>(uc->max_mt_basis_size(), uc->max_mt_basis_size(), uc->num_atom_types());
        p_mtrx_.zero();

        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {
            auto atom_type = uc->atom_type(iat);
            int nbf = atom_type->mt_basis_size();
            int ofs = atom_type->offset_lo();

            matrix<double_complex> qinv(nbf, nbf);
            atom_type->uspp().q_mtrx >> qinv;
            linalg<CPU>::geinv(nbf, qinv);
            
            /* compute P^{+}*P */
            linalg<CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(), &beta_gk_t_(0, ofs), beta_gk_t_.ld(), 
                              &beta_gk_t_(0, ofs), beta_gk_t_.ld(), &p_mtrx_(0, 0, iat), p_mtrx_.ld());
            comm_row().allreduce(&p_mtrx_(0, 0, iat), uc->max_mt_basis_size() * uc->max_mt_basis_size());

            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                for (int xi2 = 0; xi2 < nbf; xi2++) qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
            }
            /* compute (Q^{-1} + P^{+}*P)^{-1} */
            linalg<CPU>::geinv(nbf, qinv);
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                for (int xi2 = 0; xi2 < nbf; xi2++) p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
            }
        }
        
        if (parameters_.processing_unit() == GPU)
        {
            //== STOP();
            //== #ifdef _GPU_
            //== gkvec_row_ = mdarray<double, 2>(3, num_gkvec_row());
            //== /* copy G+k vectors */
            //== for (int igk_row = 0; igk_row < num_gkvec_row(); igk_row++)
            //== {
            //==     for (int x = 0; x < 3; x++) gkvec_row_(x, igk_row) = gklo_basis_descriptor_row(igk_row).gkvec[x];
            //== }
            //== gkvec_row_.allocate_on_device();
            //== gkvec_row_.copy_to_device();

            //== beta_gk_t_.allocate_on_device();
            //== beta_gk_t_.copy_to_device();
            //== #endif
        }
    }
    
    splindex<block> spl_bands(parameters_.num_fv_states(), comm_.size(), comm_.rank());
    
    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        spinor_wave_functions_ = mdarray<double_complex, 3>(nullptr, wf_size(), sub_spl_spinor_wf_.local_size(), parameters_.num_spins());
    }
    else
    {
        spinor_wave_functions_ = mdarray<double_complex, 3>(nullptr, wf_size(), spl_bands.local_size(), parameters_.num_spins());
    }

    if (use_second_variation)
    {
        /* allocate memory for first-variational eigen vectors */
        if (parameters_.unit_cell()->full_potential())
        {
            fv_eigen_vectors_panel_ = dmatrix<double_complex>(nullptr, gklo_basis_size(), parameters_.num_fv_states(), blacs_grid_);
            fv_eigen_vectors_panel_.allocate(alloc_mode);
        }

        if (parameters_.unit_cell()->full_potential())
        {
            // TODO: in case of one rank fv_states_ and fv_states_panel_ arrays are identical
            fv_states_panel_ = dmatrix<double_complex>(wf_size(), parameters_.num_fv_states(), blacs_grid_);
            fv_states_ = mdarray<double_complex, 2>(wf_size(), sub_spl_fv_states_.local_size());
        }
        else
        {
            fv_states_slab_ = matrix<double_complex>(num_gkvec_loc(), parameters_.num_fv_states());
            fv_states_ = matrix<double_complex>(num_gkvec(), spl_bands.local_size());
        }

        if (parameters_.esm_type() == ultrasoft_pseudopotential ||
            parameters_.esm_type() == norm_conserving_pseudopotential)
        {
            fv_states_slab_.zero();
            
            for (int i = 0; i < parameters_.num_fv_states(); i++)
            {
                auto location = spl_gkvec_.location(i);
                if (location.second == comm_.rank()) fv_states_slab_(location.first, i) = complex_one;
            }

            fv_states_.zero();
            for (size_t i = 0; i < spl_bands.local_size(); i++)
            {
                fv_states_(spl_bands[i], i) = complex_one;
            }

            //fv_states_panel_.zero();
            //for (int i = 0; i < parameters_.num_fv_states(); i++) fv_states_panel_.set(i, i, complex_one);

            //== fv_states_panel_.zero();
            //== for (int i = 0; i < parameters_.num_fv_states(); i++)
            //== {
            //==     int n = 0;
            //==     for (int i0 = -1; i0 <= 1; i0++)
            //==     {
            //==         for (int i1 = -1; i1 <= 1; i1++)
            //==         {
            //==             for (int i2 = -1; i2 <= 1; i2++)
            //==             {
            //==                 if (i == n)
            //==                 {
            //==                     int ig = parameters_.reciprocal_lattice()->gvec_index(vector3d<int>(i0, i1, i2));
            //==                     for (int igk = 0; igk < num_gkvec(); igk++)
            //==                     {
            //==                         if (gklo_basis_descriptor_row(igk).ig == ig) fv_states_panel_.set(igk, i, complex_one);
            //==                     }
            //==                 }
            //==                 n++;
            //==             }
            //==         }
            //==     }
            //== }

            //fv_states_panel_.gather(fv_states_);
        }
        
        if (parameters_.need_sv())
        {
            spinor_wave_functions_.allocate();
        }
        else
        {
            //spinor_wave_functions_ = mdarray<double_complex, 3>(fv_states_.at<CPU>(), wf_size(), sub_spl_spinor_wf_.local_size(), parameters_.num_spins());
            spinor_wave_functions_ = mdarray<double_complex, 3>(fv_states_.at<CPU>(), wf_size(), spl_bands.local_size(), parameters_.num_spins());
        }
    }
    else  /* use full diagonalziation */
    {
        if (parameters_.unit_cell()->full_potential())
        {
            fd_eigen_vectors_ = mdarray<double_complex, 2>(gklo_basis_size_row(), spl_spinor_wf_.local_size());
            spinor_wave_functions_.allocate();
        }
    }

    LOG_FUNC_END();
}

//== void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
//== {
//==     static SHT* sht = NULL;
//==     if (!sht) sht = new SHT(parameters_.lmax_apw());
//== 
//==     Atom* atom = parameters_.unit_cell()->atom(ia);
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
//==     vector3d<double> vc = parameters_.unit_cell()->get_cartesian_coordinates(parameters_.unit_cell()->atom(ia)->position());
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
//==             double_complex pw_value = exp(double_complex(0, Utils::scalar_product(r, gkc))) / sqrt(parameters_.unit_cell()->omega());
//==             tdiff += abs(pw_value - aw_value);
//==         }
//==     }
//== 
//==     printf("atom : %i  absolute alm error : %e  average alm error : %e\n", 
//==            ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
//== }

//== #ifdef _GPU_
//== void K_point::generate_fv_states_aw_mt_gpu()
//== {
//==     int num_fv_loc = parameters_.spl_fv_states_col().local_size();
//== 
//==     mdarray<double_complex, 2> fv_eigen_vectors_gpu_(NULL, num_gkvec_row(), num_fv_loc);
//==     fv_eigen_vectors_gpu_.allocate_on_device();
//== 
//==     cublas_set_matrix(num_gkvec_row(), num_fv_loc, sizeof(double_complex),
//==                       fv_eigen_vectors_panel_.ptr(), fv_eigen_vectors_panel_.ld(), 
//==                       fv_eigen_vectors_gpu_.ptr_device(), fv_eigen_vectors_gpu_.ld());
//== 
//==     mdarray<double_complex, 2> fv_states_col_gpu_(NULL, parameters_.unit_cell()->mt_basis_size(), num_fv_loc);
//==     fv_states_col_gpu_.allocate_on_device();
//==     fv_states_col_gpu_.zero_on_device();
//== 
//==     mdarray<double_complex, 2> alm(num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());
//==     alm.allocate_on_device();
//==     
//==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
//==     {
//==         Atom* atom = parameters_.unit_cell()->atom(ia);
//==         Atom_type* type = atom->type();
//==         
//==         generate_matching_coefficients<true>(num_gkvec_row(), ia, alm);
//==         alm.copy_to_device(); // TODO: copy only necessary fraction of the data
//== 
//==         blas<GPU>::gemm(2, 0, type->mt_aw_basis_size(), num_fv_loc, num_gkvec_row(), 
//==                         alm.ptr_device(), alm.ld(), 
//==                         fv_eigen_vectors_gpu_.ptr_device(), fv_eigen_vectors_gpu_.ld(), 
//==                         fv_states_col_gpu_.ptr_device(atom->offset_wf(), 0), fv_states_col_gpu_.ld());
//==     }
//== 
//==     cublas_get_matrix(parameters_.unit_cell()->mt_basis_size(), num_fv_loc, sizeof(double_complex), 
//==                       fv_states_col_gpu_.ptr_device(), fv_states_col_gpu_.ld(),
//==                       fv_states_col_.ptr(), fv_states_col_.ld());
//== 
//==     alm.deallocate_on_device();
//==     fv_states_col_gpu_.deallocate_on_device();
//==     fv_eigen_vectors_gpu_.deallocate_on_device();
//== }
//== #endif

void K_point::generate_fv_states()
{
    log_function_enter(__func__);
    Timer t("sirius::K_point::generate_fv_states");
    
    if (parameters_.unit_cell()->full_potential())
    {
        if (parameters_.processing_unit() == GPU && num_ranks() == 1)
        {
            #ifdef _GPU_
            auto uc = parameters_.unit_cell();

            /* copy eigen-vectors to GPU */
            fv_eigen_vectors_panel_.panel().allocate_on_device();
            fv_eigen_vectors_panel_.panel().copy_to_device();

            /* allocate GPU memory for fv_states */
            fv_states_.allocate_on_device();

            double_complex alpha(1, 0);
            double_complex beta(0, 0);

            int num_atoms_in_block = 2 * Platform::max_num_threads();
            int nblk = uc->num_atoms() / num_atoms_in_block + std::min(1, uc->num_atoms() % num_atoms_in_block);
            DUMP("nblk: %i", nblk);

            int max_mt_aw = num_atoms_in_block * uc->max_mt_aw_basis_size();
            DUMP("max_mt_aw: %i", max_mt_aw);

            mdarray<double_complex, 3> alm_row(nullptr, num_gkvec_row(), max_mt_aw, 2);
            alm_row.allocate(1);
            alm_row.allocate_on_device();
            
            int mt_aw_blk_offset = 0;
            for (int iblk = 0; iblk < nblk; iblk++)
            {
                int num_mt_aw_blk = 0;
                std::vector<int> offsets(num_atoms_in_block);
                for (int ia = iblk * num_atoms_in_block; ia < std::min(uc->num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
                {
                    auto atom = uc->atom(ia);
                    auto type = atom->type();
                    offsets[ia - iblk * num_atoms_in_block] = num_mt_aw_blk;
                    num_mt_aw_blk += type->mt_aw_basis_size();
                }

                int s = iblk % 2;
                    
                #pragma omp parallel
                {
                    int tid = Platform::thread_id();
                    for (int ia = iblk * num_atoms_in_block; ia < std::min(uc->num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
                    {
                        if (ia % Platform::num_threads() == tid)
                        {
                            int ialoc = ia - iblk * num_atoms_in_block;
                            auto atom = uc->atom(ia);
                            auto type = atom->type();

                            mdarray<double_complex, 2> alm_row_tmp(alm_row.at<CPU>(0, offsets[ialoc], s),
                                                                   alm_row.at<GPU>(0, offsets[ialoc], s),
                                                                   num_gkvec_row(), type->mt_aw_basis_size());

                            alm_coeffs_row()->generate(ia, alm_row_tmp);
                            alm_row_tmp.async_copy_to_device(tid);
                        }
                    }
                    cuda_stream_synchronize(tid);
                }
                cuda_stream_synchronize(Platform::max_num_threads());
                /* gnerate aw expansion coefficients */
                linalg<GPU>::gemm(1, 0, num_mt_aw_blk, parameters_.num_fv_states(), num_gkvec_row(), &alpha,
                                  alm_row.at<GPU>(0, 0, s), alm_row.ld(),
                                  fv_eigen_vectors_panel_.panel().at<GPU>(), fv_eigen_vectors_panel_.panel().ld(),
                                  &beta, fv_states_.at<GPU>(mt_aw_blk_offset, 0), fv_states_.ld(), Platform::max_num_threads());
                mt_aw_blk_offset += num_mt_aw_blk;
            }
            cuda_stream_synchronize(Platform::max_num_threads());
            alm_row.deallocate_on_device();

            mdarray<double_complex, 2> tmp_buf(nullptr, uc->max_mt_aw_basis_size(), parameters_.num_fv_states());
            tmp_buf.allocate_on_device();

            /* copy aw coefficients starting from bottom */
            for (int ia = parameters_.unit_cell()->num_atoms() - 1; ia >= 0; ia--)
            {
                int offset_wf = uc->atom(ia)->offset_wf();
                int offset_aw = uc->atom(ia)->offset_aw();
                int mt_aw_size = uc->atom(ia)->mt_aw_basis_size();
                
                /* copy to temporary array */
                cuda_memcpy2D_device_to_device(tmp_buf.at<GPU>(), tmp_buf.ld(),
                                               fv_states_.at<GPU>(offset_aw, 0), fv_states_.ld(),
                                               mt_aw_size, parameters_.num_fv_states(), sizeof(double_complex));

                /* copy to proper place in wave-function array */
                cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(offset_wf, 0), fv_states_.ld(),
                                               tmp_buf.at<GPU>(), tmp_buf.ld(),
                                               mt_aw_size, parameters_.num_fv_states(), sizeof(double_complex));
                
                /* copy block of local orbital coefficients */
                cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(offset_wf + mt_aw_size, 0), fv_states_.ld(),
                                               fv_eigen_vectors_panel_.panel().at<GPU>(num_gkvec_row() + uc->atom(ia)->offset_lo(), 0),
                                               fv_eigen_vectors_panel_.panel().ld(),
                                               uc->atom(ia)->mt_lo_basis_size(), parameters_.num_fv_states(), sizeof(double_complex));
            }
            /* copy block of pw coefficients */
            cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(uc->mt_basis_size(), 0), fv_states_.ld(),
                                           fv_eigen_vectors_panel_.panel().at<GPU>(),  fv_eigen_vectors_panel_.panel().ld(),
                                           num_gkvec_row(), parameters_.num_fv_states(), sizeof(double_complex));

            fv_eigen_vectors_panel_.panel().deallocate_on_device();
            fv_states_.copy_to_host();
            //fv_states_.deallocate_on_device();
            #else
            TERMINATE_NO_GPU
            #endif
        }
        else
        {
            /* local number of first-variational states, assigned to each MPI rank in the column */
            int nfv_loc = (int)sub_spl_fv_states_.local_size();

            /* total number of augmented-wave basis functions over all atoms */
            int naw = parameters_.unit_cell()->mt_aw_basis_size();

            dmatrix<double_complex> alm_panel(num_gkvec(), naw, blacs_grid_);
            /* generate panel of matching coefficients, normal layout */
            alm_coeffs_row_->generate<true>(alm_panel);

            dmatrix<double_complex> aw_coefs_panel(naw, parameters_.num_fv_states(), blacs_grid_);
            /* gnerate aw expansion coefficients */
            linalg<CPU>::gemm(1, 0, naw, parameters_.num_fv_states(), num_gkvec(), complex_one, alm_panel, 
                              fv_eigen_vectors_panel_, complex_zero, aw_coefs_panel); 
            alm_panel.deallocate(); // we don't need alm any more

            /* We have a panel of aw coefficients and a panel of 
             * first-variational eigen-vectors. We need to collect
             * them as whole vectors and setup aw, lo and G+k parts
             * of the first-variational states.
             */
            mdarray<double_complex, 2> fv_eigen_vectors(gklo_basis_size(), nfv_loc);
            /* gather full first-variational eigen-vector array */
            fv_eigen_vectors_panel_.gather(fv_eigen_vectors);

            mdarray<double_complex, 2> aw_coefs(naw, nfv_loc);
            /* gather aw coefficients */
            aw_coefs_panel.gather(aw_coefs);

            for (int i = 0; i < nfv_loc; i++)
            {
                for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
                {
                    int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
                    int offset_aw = parameters_.unit_cell()->atom(ia)->offset_aw();
                    int mt_aw_size = parameters_.unit_cell()->atom(ia)->mt_aw_basis_size();

                    /* apw block */
                    memcpy(&fv_states_(offset_wf, i), &aw_coefs(offset_aw, i), mt_aw_size * sizeof(double_complex));

                    /* lo block */
                    memcpy(&fv_states_(offset_wf + mt_aw_size, i),
                           &fv_eigen_vectors(num_gkvec() + parameters_.unit_cell()->atom(ia)->offset_lo(), i),
                           parameters_.unit_cell()->atom(ia)->mt_lo_basis_size() * sizeof(double_complex));

                    /* G+k block */
                    memcpy(&fv_states_(parameters_.unit_cell()->mt_basis_size(), i), &fv_eigen_vectors(0, i), 
                           num_gkvec() * sizeof(double_complex));
                }
            }
        }

        //if (parameters_.processing_unit() == GPU)
        //{
        //    fv_states_.allocate_on_device();
        //    fv_states_.copy_to_device();
        //}

        fv_states_panel_.scatter(fv_states_);
    }
    else
    {
        //fv_states_panel_.gather(fv_states_);
    }

    log_function_exit(__func__);
}

void K_point::generate_spinor_wave_functions()
{
    log_function_enter(__func__);
    Timer t("sirius::K_point::generate_spinor_wave_functions");

    int nfv = parameters_.num_fv_states();
    double_complex alpha(1, 0);
    double_complex beta(0, 0);

    if (use_second_variation) 
    {
        if (!parameters_.need_sv()) return;

        /* serial version */
        if (num_ranks() == 1)
        {
            spinor_wave_functions_.zero();
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef _GPU_
                spinor_wave_functions_.allocate_on_device();
                spinor_wave_functions_.zero_on_device();
                #endif
            }

            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                if (parameters_.num_mag_dims() != 3)
                {
                    if (parameters_.processing_unit() == GPU)
                    {
                        #ifdef _GPU_
                        sv_eigen_vectors_[ispn].panel().allocate_on_device();
                        sv_eigen_vectors_[ispn].panel().copy_to_device();

                        linalg<GPU>::gemm(0, 0, wf_size(), nfv, nfv, &alpha, fv_states_.at<GPU>(), fv_states_.ld(), 
                                          sv_eigen_vectors_[ispn].panel().at<GPU>(), sv_eigen_vectors_[ispn].panel().ld(),
                                          &beta, spinor_wave_functions_.at<GPU>(0, ispn * nfv, ispn), spinor_wave_functions_.ld());

                        sv_eigen_vectors_[ispn].panel().deallocate_on_device();
                        #else
                        TERMINATE_NO_GPU
                        #endif
                    }
                    else
                    {
                        /* multiply up block for first half of the bands, dn block for second half of the bands */
                        linalg<CPU>::gemm(0, 0, wf_size(), nfv, nfv, fv_states_.at<CPU>(), fv_states_.ld(), 
                                          &sv_eigen_vectors_[ispn](0, 0), sv_eigen_vectors_[ispn].ld(), 
                                          &spinor_wave_functions_(0, ispn * nfv, ispn), spinor_wave_functions_.ld());
                    }
                }
                else
                {
                    /* multiply up block and then dn block for all bands */
                    linalg<CPU>::gemm(0, 0, wf_size(), parameters_.num_bands(), nfv, fv_states_.at<CPU>(), fv_states_.ld(), 
                                      &sv_eigen_vectors_[0](ispn * nfv, 0), sv_eigen_vectors_[0].ld(), 
                                      &spinor_wave_functions_(0, 0, ispn), spinor_wave_functions_.ld());
                }
            }
            if (parameters_.processing_unit() == GPU)
            {
                #ifdef _GPU_
                spinor_wave_functions_.copy_to_host();
                spinor_wave_functions_.deallocate_on_device();
                fv_states_.deallocate_on_device();
                #endif
            }
        }
        /* parallel version */
        else
        {
            /* spin component of spinor wave functions */
            dmatrix<double_complex> spin_component_panel_(wf_size(), parameters_.num_bands(), blacs_grid_);
            
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                spin_component_panel_.zero();

                if (parameters_.num_mag_dims() != 3)
                {
                    /* multiply up block for first half of the bands, dn block for second half of the bands */
                    linalg<CPU>::gemm(0, 0, wf_size(), nfv, nfv, complex_one, fv_states_panel_, 0, 0, 
                                      sv_eigen_vectors_[ispn], 0, 0, complex_zero, spin_component_panel_, 0, ispn * nfv);
                    
                }
                else
                {
                    /* multiply up block and then dn block for all bands */
                    linalg<CPU>::gemm(0, 0, wf_size(), parameters_.num_bands(), nfv, complex_one, fv_states_panel_, 0, 0, 
                                      sv_eigen_vectors_[0], ispn * nfv, 0, complex_zero, spin_component_panel_, 0, 0);

                }
                auto sm = spinor_wave_functions_.submatrix(ispn); 
                spin_component_panel_.gather(sm);
            }
        }
    }
    else
    {
        STOP();
    //==     mdarray<double_complex, 2> alm(num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());

    //==     /** \todo generalize for non-collinear case */
    //==     spinor_wave_functions_.zero();
    //==     for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
    //==     {
    //==         for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    //==         {
    //==             Atom* atom = parameters_.unit_cell()->atom(ia);
    //==             Atom_type* type = atom->type();
    //==             
    //==             /** \todo generate unconjugated coefficients for better readability */
    //==             generate_matching_coefficients<true>(num_gkvec_row(), ia, alm);

    //==             blas<CPU>::gemm(2, 0, type->mt_aw_basis_size(), ncol, num_gkvec_row(), &alm(0, 0), alm.ld(), 
    //==                             &fd_eigen_vectors_(0, ispn * ncol), fd_eigen_vectors_.ld(), 
    //==                             &spinor_wave_functions_(atom->offset_wf(), ispn, ispn * ncol), wfld); 
    //==         }

    //==         for (int j = 0; j < ncol; j++)
    //==         {
    //==             copy_lo_blocks(&fd_eigen_vectors_(0, j + ispn * ncol), &spinor_wave_functions_(0, ispn, j + ispn * ncol));

    //==             copy_pw_block(&fd_eigen_vectors_(0, j + ispn * ncol), &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j + ispn * ncol));
    //==         }
    //==     }
    //==     /** \todo how to distribute states in case of full diagonalziation. num_fv_states will probably be reused. 
    //==               maybe the 'fv' should be renamed. */
    }
    //== 
    //== for (int i = 0; i < parameters_.spl_spinor_wf_col().local_size(); i++)
    //==     Platform::allreduce(&spinor_wave_functions_(0, 0, i), wfld, parameters_.mpi_grid().communicator(1 << _dim_row_));
    //== 
    log_function_exit(__func__);
}

void K_point::generate_gkvec(double gk_cutoff)
{
    if ((gk_cutoff * parameters_.unit_cell()->max_mt_radius() > double(parameters_.lmax_apw())) && 
        parameters_.unit_cell()->full_potential())
    {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff << ") is too large for a given lmax (" 
          << parameters_.lmax_apw() << ") and a maximum MT radius (" << parameters_.unit_cell()->max_mt_radius() << ")" << std::endl
          << "suggested minimum value for lmax : " << int(gk_cutoff * parameters_.unit_cell()->max_mt_radius()) + 1;
        warning_local(__FILE__, __LINE__, s);
    }

    if (gk_cutoff * 2 > parameters_.pw_cutoff())
    {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << parameters_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff * 2;
        error_local(__FILE__, __LINE__, s);
    }

    std::vector< std::pair<double, int> > gkmap;

    /* find G-vectors for which |G+k| < cutoff */
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
    {
        vector3d<double> vgk;
        for (int x = 0; x < 3; x++) vgk[x] = parameters_.reciprocal_lattice()->gvec(ig)[x] + vk_[x];

        vector3d<double> v = parameters_.reciprocal_lattice()->get_cartesian_coordinates(vgk);
        double gklen = v.length();

        if (gklen <= gk_cutoff) gkmap.push_back(std::pair<double, int>(gklen, ig));
    }

    std::sort(gkmap.begin(), gkmap.end());

    gkvec_ = mdarray<double, 2>(3, gkmap.size());

    gvec_index_.resize(gkmap.size());

    for (int ig = 0; ig < (int)gkmap.size(); ig++)
    {
        gvec_index_[ig] = gkmap[ig].second;
        for (int x = 0; x < 3; x++)
        {
            gkvec_(x, ig) = parameters_.reciprocal_lattice()->gvec(gkmap[ig].second)[x] + vk_[x];
        }
    }
    
    fft_index_.resize(num_gkvec());
    for (int igk = 0; igk < num_gkvec(); igk++) fft_index_[igk] = parameters_.fft()->index_map(gvec_index_[igk]);

    if (parameters_.esm_type() == ultrasoft_pseudopotential ||
        parameters_.esm_type() == norm_conserving_pseudopotential)
    {
        fft_index_coarse_.resize(num_gkvec());
        for (int igk = 0; igk < num_gkvec(); igk++)
        {
            /* G-vector index in the fine mesh */
            int ig = gvec_index_[igk];
            /* G-vector fractional coordinates */
            vector3d<int> gvec = parameters_.reciprocal_lattice()->gvec(ig);

            /* linear index inside coarse FFT buffer */
            fft_index_coarse_[igk] = parameters_.fft_coarse()->index(gvec[0], gvec[1], gvec[2]);
        }
    }
}

void K_point::init_gkvec_ylm_and_len(int lmax__, int num_gkvec__, std::vector<gklo_basis_descriptor>& desc__)
{
    gkvec_ylm_ = mdarray<double_complex, 2>(Utils::lmmax(lmax__), num_gkvec__);

    //gkvec_len_.resize(num_gkvec_row());

    #pragma omp parallel for default(shared)
    for (int i = 0; i < num_gkvec__; i++)
    {
        int igk = desc__[i].igk;

        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(gkvec<cartesian>(igk));
        
        SHT::spherical_harmonics(lmax__, vs[1], vs[2], &gkvec_ylm_(0, i));
        
        //gkvec_len_[igk_row] = vs[0];
    }
}

void K_point::init_gkvec_phase_factors(int num_gkvec__, std::vector<gklo_basis_descriptor>& desc__)
{
    gkvec_phase_factors_ = mdarray<double_complex, 2>(num_gkvec__, parameters_.unit_cell()->num_atoms());

    #pragma omp parallel for default(shared)
    for (int i = 0; i < num_gkvec__; i++)
    {
        int igk = desc__[i].igk;

        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            double phase = twopi * (gkvec<fractional>(igk) * parameters_.unit_cell()->atom(ia)->position());

            gkvec_phase_factors_(i, ia) = std::exp(double_complex(0.0, phase));
        }
    }
}

void K_point::init_gkvec()
{
    int lmax = - 1;
    switch (parameters_.esm_type())
    {
        case full_potential_lapwlo:
        {
            lmax = parameters_.lmax_apw();
            break;
            //init_gkvec_ylm_and_len(parameters_.lmax_apw(), num_gkvec_row(), gklo_basis_descriptors_row_);
            //init_gkvec_phase_factors(num_gkvec_row(), gklo_basis_descriptors_row_);
            //break;
        }
        case full_potential_pwlo:
        {
            lmax = parameters_.lmax_pw();
            break;

            //init_gkvec_ylm_and_len(parameters_.lmax_pw(), num_gkvec_row(), gklo_basis_descriptors_row_);
            //init_gkvec_phase_factors(num_gkvec_row(), gklo_basis_descriptors_row_);
            //break;
        }
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            if (num_gkvec() != wf_size()) TERMINATE("wrong size of wave-functions");
            lmax = parameters_.lmax_beta();
            //init_gkvec_ylm_and_len(parameters_.lmax_beta(), num_gkvec_loc(), gklo_basis_descriptors_local_);
            //init_gkvec_phase_factors(num_gkvec_loc(), gklo_basis_descriptors_local_);
            break;
        }
    }
    
    init_gkvec_ylm_and_len(lmax, num_gkvec_row(), gklo_basis_descriptors_row_);
    init_gkvec_phase_factors(num_gkvec_row(), gklo_basis_descriptors_row_);
}

void K_point::build_gklo_basis_descriptors()
{
    gklo_basis_descriptors_.clear();

    gklo_basis_descriptor gklo;

    /* G+k basis functions */
    for (int igk = 0; igk < num_gkvec(); igk++)
    {
        gklo.id = (int)gklo_basis_descriptors_.size();
        gklo.igk = igk;
        gklo.gkvec = gkvec<fractional>(igk);
        gklo.gkvec_cart = gkvec<cartesian>(igk);
        gklo.ig = gvec_index(igk);
        gklo.ia = -1;
        gklo.l = -1;
        gklo.lm = -1;
        gklo.order = -1;
        gklo.idxrf = -1;

        gklo_basis_descriptors_.push_back(gklo);
    }

    if (parameters_.esm_type() == full_potential_lapwlo || parameters_.esm_type() == full_potential_pwlo)
    {
        /* local orbital basis functions */
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            Atom* atom = parameters_.unit_cell()->atom(ia);
            Atom_type* type = atom->type();
        
            int lo_index_offset = type->mt_aw_basis_size();
            
            for (int j = 0; j < type->mt_lo_basis_size(); j++) 
            {
                int l = type->indexb(lo_index_offset + j).l;
                int lm = type->indexb(lo_index_offset + j).lm;
                int order = type->indexb(lo_index_offset + j).order;
                int idxrf = type->indexb(lo_index_offset + j).idxrf;
                gklo.id = (int)gklo_basis_descriptors_.size();
                gklo.igk = -1;
                gklo.gkvec = vector3d<double>(0.0);
                gklo.gkvec_cart = vector3d<double>(0.0);
                gklo.ig = -1;
                gklo.ia = ia;
                gklo.l = l;
                gklo.lm = lm;
                gklo.order = order;
                gklo.idxrf = idxrf;

                gklo_basis_descriptors_.push_back(gklo);
            }
        }
    
        /* ckeck if we count basis functions correctly */
        if ((int)gklo_basis_descriptors_.size() != (num_gkvec() + parameters_.unit_cell()->mt_lo_basis_size()))
        {
            std::stringstream s;
            s << "(L)APW+lo basis descriptors array has a wrong size" << std::endl
              << "size of apwlo_basis_descriptors_ : " << gklo_basis_descriptors_.size() << std::endl
              << "num_gkvec : " << num_gkvec() << std::endl 
              << "mt_lo_basis_size : " << parameters_.unit_cell()->mt_lo_basis_size();
            error_local(__FILE__, __LINE__, s);
        }
    }
}

void K_point::distribute_basis_index()
{
    if (parameters_.wave_function_distribution() == block_cyclic_2d)
    {
        /* distribute Gk+lo basis between rows */
        splindex<block_cyclic> spl_row(gklo_basis_size(), num_ranks_row_, rank_row_, blacs_grid_.cyclic_block_size());
        gklo_basis_descriptors_row_.resize(spl_row.local_size());
        for (int i = 0; i < (int)spl_row.local_size(); i++)
            gklo_basis_descriptors_row_[i] = gklo_basis_descriptors_[spl_row[i]];

        /* distribute Gk+lo basis between columns */
        splindex<block_cyclic> spl_col(gklo_basis_size(), num_ranks_col_, rank_col_, blacs_grid_.cyclic_block_size());
        gklo_basis_descriptors_col_.resize(spl_col.local_size());
        for (int i = 0; i < (int)spl_col.local_size(); i++)
            gklo_basis_descriptors_col_[i] = gklo_basis_descriptors_[spl_col[i]];

        #ifdef _SCALAPACK_
        int bs = blacs_grid_.cyclic_block_size();
        int nr = linalg_base::numroc(gklo_basis_size(), bs, rank_row(), 0, num_ranks_row());
        
        if (nr != gklo_basis_size_row()) error_local(__FILE__, __LINE__, "numroc returned a different local row size");

        int nc = linalg_base::numroc(gklo_basis_size(), bs, rank_col(), 0, num_ranks_col());
        
        if (nc != gklo_basis_size_col()) error_local(__FILE__, __LINE__, "numroc returned a different local column size");
        #endif

        /* get number of column G+k vectors */
        num_gkvec_col_ = 0;
        for (int i = 0; i < gklo_basis_size_col(); i++)
        {
            if (gklo_basis_descriptor_col(i).igk != -1) num_gkvec_col_++;
        }
    }

    if (parameters_.wave_function_distribution() == slab)
    {
        /* split G+k vectors between all available ranks and keep the split index */
        spl_gkvec_ = splindex<block>(gklo_basis_size(), comm_.size(), comm_.rank());
        gklo_basis_descriptors_row_.resize(spl_gkvec_.local_size());
        for (int i = 0; i < (int)spl_gkvec_.local_size(); i++)
            gklo_basis_descriptors_row_[i] = gklo_basis_descriptors_[spl_gkvec_[i]];

    }

    /* get the number of row G+k-vectors */
    num_gkvec_row_ = 0;
    for (int i = 0; i < gklo_basis_size_row(); i++)
    {
        if (gklo_basis_descriptor_row(i).igk != -1) num_gkvec_row_++;
    }
    
    //== spl_gkvec_ = splindex<block>(gklo_basis_size(), comm_.size(), comm_.rank());
    //== gklo_basis_descriptors_local_.resize(spl_gkvec_.local_size());
    //== for (int i = 0; i < (int)spl_gkvec_.local_size(); i++)
    //==     gklo_basis_descriptors_local_[i] = gklo_basis_descriptors_[spl_gkvec_[i]];
}

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

void K_point::test_fv_states(int use_fft)
{
    STOP();

    //== std::vector<double_complex> v1;
    //== std::vector<double_complex> v2;
    //== 
    //== if (use_fft == 0) 
    //== {
    //==     v1.resize(num_gkvec());
    //==     v2.resize(fft_->size());
    //== }
    //== 
    //== if (use_fft == 1) 
    //== {
    //==     v1.resize(fft_->size());
    //==     v2.resize(fft_->size());
    //== }
    //== 
    //== double maxerr = 0;

    //== for (int j1 = 0; j1 < parameters_.spl_fv_states_col().local_size(); j1++)
    //== {
    //==     if (use_fft == 0)
    //==     {
    //==         fft_->input(num_gkvec(), &fft_index_[0], &fv_states_col_(parameters_.unit_cell()->mt_basis_size(), j1));
    //==         fft_->transform(1);
    //==         fft_->output(&v2[0]);

    //==         for (int ir = 0; ir < fft_->size(); ir++) 
    //==             v2[ir] *= parameters_.step_function(ir);
    //==         
    //==         fft_->input(&v2[0]);
    //==         fft_->transform(-1);
    //==         fft_->output(num_gkvec(), &fft_index_[0], &v1[0]); 
    //==     }
    //==     
    //==     if (use_fft == 1)
    //==     {
    //==         fft_->input(num_gkvec(), &fft_index_[0], &fv_states_col_(parameters_.unit_cell()->mt_basis_size(), j1));
    //==         fft_->transform(1);
    //==         fft_->output(&v1[0]);
    //==     }
    //==    
    //==     for (int j2 = 0; j2 < parameters_.spl_fv_states_row().local_size(); j2++)
    //==     {
    //==         double_complex zsum(0, 0);
    //==         for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    //==         {
    //==             int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
    //==             Atom_type* type = parameters_.unit_cell()->atom(ia)->type();
    //==             Atom_symmetry_class* symmetry_class = parameters_.unit_cell()->atom(ia)->symmetry_class();

    //==             for (int l = 0; l <= parameters_.lmax_apw(); l++)
    //==             {
    //==                 int ordmax = type->indexr().num_rf(l);
    //==                 for (int io1 = 0; io1 < ordmax; io1++)
    //==                 {
    //==                     for (int io2 = 0; io2 < ordmax; io2++)
    //==                     {
    //==                         for (int m = -l; m <= l; m++)
    //==                         {
    //==                             zsum += conj(fv_states_col_(offset_wf + type->indexb_by_l_m_order(l, m, io1), j1)) *
    //==                                          fv_states_row_(offset_wf + type->indexb_by_l_m_order(l, m, io2), j2) * 
    //==                                          symmetry_class->o_radial_integral(l, io1, io2);
    //==                         }
    //==                     }
    //==                 }
    //==             }
    //==         }
    //==         
    //==         if (use_fft == 0)
    //==         {
    //==            for (int ig = 0; ig < num_gkvec(); ig++)
    //==                zsum += conj(v1[ig]) * fv_states_row_(parameters_.unit_cell()->mt_basis_size() + ig, j2);
    //==         }
    //==        
    //==         if (use_fft == 1)
    //==         {
    //==             fft_->input(num_gkvec(), &fft_index_[0], &fv_states_row_(parameters_.unit_cell()->mt_basis_size(), j2));
    //==             fft_->transform(1);
    //==             fft_->output(&v2[0]);

    //==             for (int ir = 0; ir < fft_->size(); ir++)
    //==                 zsum += conj(v1[ir]) * v2[ir] * parameters_.step_function(ir) / double(fft_->size());
    //==         }
    //==         
    //==         if (use_fft == 2) 
    //==         {
    //==             for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
    //==             {
    //==                 for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
    //==                 {
    //==                     int ig3 = parameters_.reciprocal_lattice()->index_g12(gvec_index(ig1), gvec_index(ig2));
    //==                     zsum += conj(fv_states_col_(parameters_.unit_cell()->mt_basis_size() + ig1, j1)) * 
    //==                                  fv_states_row_(parameters_.unit_cell()->mt_basis_size() + ig2, j2) * 
    //==                             parameters_.step_function()->theta_pw(ig3);
    //==                 }
    //==            }
    //==         }

    //==         if (parameters_.spl_fv_states_col(j1) == parameters_.spl_fv_states_row(j2)) zsum = zsum - double_complex(1, 0);
    //==        
    //==         maxerr = std::max(maxerr, abs(zsum));
    //==     }
    //== }

    //== Platform::allreduce<op_max>(&maxerr, 1, parameters_.mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_));

    //== if (parameters_.mpi_grid().side(1 << _dim_k_)) 
    //== {
    //==     printf("k-point: %f %f %f, interstitial integration : %i, maximum error : %18.10e\n", 
    //==            vk_[0], vk_[1], vk_[2], use_fft, maxerr);
    //== }
}

void K_point::test_spinor_wave_functions(int use_fft)
{
    if (num_ranks() > 1) error_local(__FILE__, __LINE__, "test of spinor wave functions on multiple ranks is not implemented");

    std::vector<double_complex> v1[2];
    std::vector<double_complex> v2;

    if (use_fft == 0 || use_fft == 1) v2.resize(fft_->size());
    
    if (use_fft == 0) 
    {
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++) v1[ispn].resize(num_gkvec());
    }
    
    if (use_fft == 1) 
    {
        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++) v1[ispn].resize(fft_->size());
    }
    
    double maxerr = 0;

    for (int j1 = 0; j1 < parameters_.num_bands(); j1++)
    {
        if (use_fft == 0)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                fft_->input(num_gkvec(), &fft_index_[0], 
                                       &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j1));
                fft_->transform(1);
                fft_->output(&v2[0]);

                for (int ir = 0; ir < fft_->size(); ir++) v2[ir] *= parameters_.step_function(ir);
                
                fft_->input(&v2[0]);
                fft_->transform(-1);
                fft_->output(num_gkvec(), &fft_index_[0], &v1[ispn][0]); 
            }
        }
        
        if (use_fft == 1)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                fft_->input(num_gkvec(), &fft_index_[0], 
                                       &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j1));
                fft_->transform(1);
                fft_->output(&v1[ispn][0]);
            }
        }
       
        for (int j2 = 0; j2 < parameters_.num_bands(); j2++)
        {
            double_complex zsum(0, 0);
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
                {
                    int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
                    Atom_type* type = parameters_.unit_cell()->atom(ia)->type();
                    Atom_symmetry_class* symmetry_class = parameters_.unit_cell()->atom(ia)->symmetry_class();

                    for (int l = 0; l <= parameters_.lmax_apw(); l++)
                    {
                        int ordmax = type->indexr().num_rf(l);
                        for (int io1 = 0; io1 < ordmax; io1++)
                        {
                            for (int io2 = 0; io2 < ordmax; io2++)
                            {
                                for (int m = -l; m <= l; m++)
                                {
                                    zsum += conj(spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io1), ispn, j1)) *
                                            spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m, io2), ispn, j2) * 
                                            symmetry_class->o_radial_integral(l, io1, io2);
                                }
                            }
                        }
                    }
                }
            }
            
            if (use_fft == 0)
            {
               for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
               {
                   for (int ig = 0; ig < num_gkvec(); ig++)
                       zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size() + ig, ispn, j2);
               }
            }
           
            if (use_fft == 1)
            {
                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                {
                    fft_->input(num_gkvec(), &fft_index_[0], 
                                           &spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size(), ispn, j2));
                    fft_->transform(1);
                    fft_->output(&v2[0]);

                    for (int ir = 0; ir < fft_->size(); ir++)
                        zsum += conj(v1[ispn][ir]) * v2[ir] * parameters_.step_function(ir) / double(fft_->size());
                }
            }
            
            if (use_fft == 2) 
            {
                for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                {
                    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                    {
                        int ig3 = parameters_.reciprocal_lattice()->index_g12(gvec_index(ig1), gvec_index(ig2));
                        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                        {
                            zsum += conj(spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size() + ig1, ispn, j1)) * 
                                    spinor_wave_functions_(parameters_.unit_cell()->mt_basis_size() + ig2, ispn, j2) * 
                                    parameters_.step_function()->theta_pw(ig3);
                        }
                    }
               }
           }

           zsum = (j1 == j2) ? zsum - double_complex(1.0, 0.0) : zsum;
           maxerr = std::max(maxerr, abs(zsum));
        }
    }
    std :: cout << "maximum error = " << maxerr << std::endl;
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

    for (int iloc = 0; iloc < (int)spl_fv_states_.local_size(); iloc++)
    {
        int i = (int)spl_fv_states_[iloc];
        for (int jloc = 0; jloc < gklo_basis_size_row(); jloc++)
        {
            int j = gklo_basis_descriptor_row(jloc).id;
            fv_evec(j, i) = fv_eigen_vectors_panel_(jloc, iloc);
        }
    }
    comm_.allreduce(fv_evec.at<CPU>(), (int)fv_evec.size());
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


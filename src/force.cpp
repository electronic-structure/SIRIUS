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

/** \file force.cpp
 *   
 *  \brief Contains implementation of sirius::Force class.
 */
 
#include "force.h"

namespace sirius {

void Force::compute_dmat(Global& parameters_, K_point* kp, mdarray<double_complex, 2>& dm)
{
    Timer t("sirius::Force::compute_dmat");

    stop_here

    //=== dm.zero();

    //=== // second-variational dimensions
    //=== int nsrow = parameters_.spl_fv_states_row().local_size();
    //=== int nscol = parameters_.spl_fv_states_col().local_size();

    //=== // compute the density matrix
    //=== if (!parameters_.need_sv())
    //=== {
    //===     for (int i = 0; i < nscol; i++)
    //===     {
    //===         int ist = parameters_.spl_fv_states_col(i);
    //===         for (int j = 0; j < nsrow; j++)
    //===         {
    //===             if (parameters_.spl_fv_states_row(j) == ist) dm(j, i) = kp->band_occupancy(ist);
    //===         }
    //===     }
    //=== }
    //=== else
    //=== {
    //===     // multiply second-variational eigen-vectors with band occupancies
    //===     mdarray<double_complex, 2>& sv_evec = kp->sv_eigen_vectors();
    //===     mdarray<double_complex, 2> evq(sv_evec.size(0), sv_evec.size(1));
    //===     for (int i = 0; i < parameters_.spl_spinor_wf_col().local_size(); i++)
    //===     {
    //===         int n = parameters_.spl_spinor_wf_col(i);
    //===         for (int j = 0; j < sv_evec.size(0); j++) evq(j, i) = sv_evec(j, i) * kp->band_occupancy(n);
    //===     }
    //===     
    //===     int ns = (parameters_.num_mag_dims() == 3) ? 2 : 1;
    //===     
    //===     // Important! Obtained with the following zgemm, density matrix is conjugated. 
    //===     if (kp->num_ranks() == 1)
    //===     {
    //===         //== // TODO: this possibly can be combined
    //===         //== if (parameters_.num_mag_dims() != 3)
    //===         //== {
    //===         //==     blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //===         //==                     &sv_evec(0, 0), sv_evec.ld(), &evq(0, 0), evq.ld(), &dm(0, 0), dm.ld());
    //===         //== }
    //===         //== else
    //===         //== {
    //===         //==     for (int ispn = 0; ispn < 2; ispn++)
    //===         //==     {
    //===         //==         blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //===         //==                         double_complex(1, 0), &sv_evec(ispn * parameters_.num_fv_states(), 0), sv_evec.ld(), 
    //===         //==                         &evq(ispn * parameters_.num_fv_states(), 0), evq.ld(), double_complex(1, 0), &dm(0, 0), dm.ld());
    //===         //==     }


    //===         //== }

    //===         for (int s = 0; s < ns; s++)
    //===         {
    //===             blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //===                             double_complex(1, 0), &sv_evec(s * parameters_.num_fv_states(), 0), sv_evec.ld(), 
    //===                             &evq(s * parameters_.num_fv_states(), 0), evq.ld(), double_complex(1, 0), &dm(0, 0), dm.ld());
    //===         }
    //===     }
    //===     else
    //===     {
    //===         #ifdef _SCALAPACK_
    //===         for (int s = 0; s < ns; s++)
    //===         {
    //===             pblas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //===                              double_complex(1, 0), &sv_evec(s * nsrow, 0), sv_evec.ld(), &evq(s * nsrow, 0), evq.ld(),
    //===                              double_complex(1, 0), &dm(0, 0), dm.ld(), parameters_.cyclic_block_size(), 
    //===                              parameters_.blacs_context());
    //===         }
    //===         #else
    //===         error_local(__FILE__, __LINE__, "not compiled with ScaLAPACK");
    //===         #endif
    //===     }
    //=== }
}

void Force::ibs_force(Global& parameters_, Band* band, K_point* kp, mdarray<double, 2>& ffac, mdarray<double, 2>& forcek)
{
    Timer timer("sirius::Force::ibs_force");
    
    stop_here

    //== int apw_offset_col = kp->apw_offset_col();

    //== forcek.zero();

    //== // first-variational dimensions
    //== int nfrow = kp->gklo_basis_size_row();
    //== int nfcol = kp->gklo_basis_size_col();

    //== // second-variational dimensions
    //== int nsrow = parameters_.spl_fv_states_row().local_size();
    //== int nscol = parameters_.spl_fv_states_col().local_size();

    //== mdarray<double_complex, 2> h(nfrow, nfcol);
    //== mdarray<double_complex, 2> o(nfrow, nfcol);
    //== 
    //== mdarray<double_complex, 2> vh(nfrow, nfcol);
    //== mdarray<double_complex, 2> vo(nfrow, nfcol);
    //== 
    //== mdarray<double_complex, 2> alm(kp->num_gkvec_loc(), parameters_.unit_cell()->max_mt_aw_basis_size());
    //== mdarray<double_complex, 2> halm(kp->num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());
    //== 
    //== mdarray<double_complex, 2> dm(nsrow, nscol);

    //== compute_dmat(parameters_, kp, dm);

    //== //== dm.zero();

    //== //== // compute the density matrix
    //== //== if (!parameters_.need_sv())
    //== //== {
    //== //==     for (int i = 0; i < nscol; i++)
    //== //==     {
    //== //==         int ist = parameters_.spl_fv_states_col(i);
    //== //==         for (int j = 0; j < nsrow; j++)
    //== //==         {
    //== //==             if (parameters_.spl_fv_states_row(j) == ist) dm(j, i) = kp->band_occupancy(ist);
    //== //==         }
    //== //==     }
    //== //== }
    //== //== else
    //== //== {
    //== //==     // multiply second-variational eigen-vectors with band occupancies
    //== //==     mdarray<double_complex, 2>& sv_evec = kp->sv_eigen_vectors();
    //== //==     mdarray<double_complex, 2> evq(sv_evec.size(0), sv_evec.size(1));
    //== //==     for (int i = 0; i < parameters_.spl_spinor_wf_col().local_size(); i++)
    //== //==     {
    //== //==         int n = parameters_.spl_spinor_wf_col(i);
    //== //==         for (int j = 0; j < sv_evec.size(0); j++) evq(j, i) = sv_evec(j, i) * kp->band_occupancy(n);
    //== //==     }
    //== //==     
    //== //==     // Important! Obtained with the following zgemm, density matrix is conjugated. 
    //== //==     if (kp->num_ranks() == 1)
    //== //==     {
    //== //==         // TODO: this possibly can be combined
    //== //==         if (parameters_.num_mag_dims() != 3)
    //== //==         {
    //== //==             blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //== //==                             &sv_evec(0, 0), sv_evec.ld(), &evq(0, 0), evq.ld(), &dm(0, 0), dm.ld());
    //== //==         }
    //== //==         else
    //== //==         {
    //== //==             for (int ispn = 0; ispn < 2; ispn++)
    //== //==             {
    //== //==                 blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //== //==                                 double_complex(1, 0), &sv_evec(ispn * parameters_.num_fv_states(), 0), sv_evec.ld(), 
    //== //==                                 &evq(ispn * parameters_.num_fv_states(), 0), evq.ld(), double_complex(1, 0), &dm(0, 0), dm.ld());
    //== //==             }


    //== //==         }
    //== //==     }
    //== //==     else
    //== //==     {
    //== //==         #ifdef _SCALAPACK_
    //== //==         int ns = (parameters_.num_mag_dims() == 3) ? 2 : 1;

    //== //==         for (int s = 0; s < ns; s++)
    //== //==         {
    //== //==             pblas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
    //== //==                              double_complex(1, 0), &sv_evec(s * nsrow, 0), sv_evec.ld(), &evq(s * nsrow, 0), evq.ld(),
    //== //==                              double_complex(1, 0), &dm(0, 0), dm.ld(), parameters_.cyclic_block_size(), 
    //== //==                              parameters_.blacs_context());
    //== //==         }
    //== //==         #else
    //== //==         error_local(__FILE__, __LINE__, "not compiled with ScaLAPACK");
    //== //==         #endif
    //== //==     }

    //== //==     //// TODO: this is a zgemm or pzgemm
    //== //==     //for (int n = 0; n < parameters_.num_bands(); n++)
    //== //==     //{
    //== //==     //    for (int i = 0; i < band->spl_fv_states_row().global_size(); i++)
    //== //==     //    {
    //== //==     //        int ist = i % parameters_.num_fv_states();
    //== //==     //        int ispn = i / parameters_.num_fv_states();
    //== //==     //        for (int j = 0; j < band->spl_fv_states_row().global_size(); j++)
    //== //==     //        {
    //== //==     //            int jst = j % parameters_.num_fv_states();
    //== //==     //            int jspn = j / parameters_.num_fv_states();

    //== //==     //            if (ispn == jspn)
    //== //==     //            {
    //== //==     //                dm(ist, jst) += band_occupancy(n) * conj(sv_eigen_vectors_(i, n)) * sv_eigen_vectors_(j, n); 
    //== //==     //            }
    //== //==     //        }
    //== //==     //    }
    //== //==     //}
    //== //== }

    //== mdarray<double_complex, 2>& fv_evec = kp->fv_eigen_vectors_panel();
    //== mdarray<double_complex, 2> zm1(nfrow, nscol);
    //== mdarray<double_complex, 2> zf(nsrow, nscol);

    //== for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    //== {
    //==     h.zero();
    //==     o.zero();
    //==     
    //==     Atom* atom = parameters_.unit_cell()->atom(ia);
    //==     Atom_type* type = atom->type();

    //==     int iat = type->id();
    //==     
    //==     kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
    //==     
    //==     band->apply_hmt_to_apw<nm>(kp->num_gkvec_row(), ia, alm, halm);
    //==     
    //==     // apw-apw block of the overlap matrix
    //==     blas<cpu>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), &alm(0, 0), alm.ld(), 
    //==                     &alm(apw_offset_col, 0), alm.ld(), &o(0, 0), o.ld()); 
    //==         
    //==     // apw-apw block of the Hamiltonian matrix
    //==     blas<cpu>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), &halm(0, 0), halm.ld(), 
    //==                     &alm(apw_offset_col, 0), alm.ld(), &h(0, 0), h.ld());
    //==     
    //==     // apw-lo and lo-apw blocks of Hamiltonian and overlap
    //==     band->set_fv_h_o_apw_lo(kp, type, atom, ia, alm, h, o);

    //==     for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    //==     {
    //==         for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
    //==         {
    //==             int ig12 = parameters_.reciprocal_lattice()->index_g12(kp->gklo_basis_descriptor_row(igk_row).ig,
    //==                                                                    kp->gklo_basis_descriptor_col(igk_col).ig);
    //==             int igs = parameters_.reciprocal_lattice()->gvec_shell(ig12);

    //==             double_complex zt = conj(parameters_.reciprocal_lattice()->gvec_phase_factor<global>(ig12, ia)) * ffac(igs, iat);

    //==             double t1 = 0.5 * Utils::scalar_product(kp->gklo_basis_descriptor_row(igk_row).gkvec_cart, 
    //==                                                     kp->gklo_basis_descriptor_col(igk_col).gkvec_cart);

    //==             h(igk_row, igk_col) -= t1 * zt;
    //==             o(igk_row, igk_col) -= zt;
    //==         }
    //==     }

    //==     for (int x = 0; x < 3; x++)
    //==     {
    //==         for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    //==         {
    //==             for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
    //==             {
    //==                 int ig12 = parameters_.reciprocal_lattice()->index_g12(kp->gklo_basis_descriptor_row(igk_row).ig,
    //==                                                  kp->gklo_basis_descriptor_col(igk_col).ig);

    //==                 vector3d<double> vg = parameters_.reciprocal_lattice()->gvec_cart(ig12);
    //==                 vh(igk_row, igk_col) = double_complex(0.0, vg[x]) * h(igk_row, igk_col);
    //==                 vo(igk_row, igk_col) = double_complex(0.0, vg[x]) * o(igk_row, igk_col);
    //==             }
    //==         }

    //==         for (int icol = kp->num_gkvec_col(); icol < kp->gklo_basis_size_col(); icol++)
    //==         {
    //==             for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++)
    //==             {
    //==                 vector3d<double>& vgk = kp->gklo_basis_descriptor_row(igk_row).gkvec_cart;
    //==                 vh(igk_row, icol) = double_complex(0.0, vgk[x]) * h(igk_row, icol);
    //==                 vo(igk_row, icol) = double_complex(0.0, vgk[x]) * o(igk_row, icol);
    //==             }
    //==         }
    //==                 
    //==         for (int irow = kp->num_gkvec_row(); irow < kp->gklo_basis_size_row(); irow++)
    //==         {
    //==             for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++)
    //==             {
    //==                 vector3d<double>& vgk = kp->gklo_basis_descriptor_col(igk_col).gkvec_cart;
    //==                 vh(irow, igk_col) = double_complex(0.0, -vgk[x]) * h(irow, igk_col);
    //==                 vo(irow, igk_col) = double_complex(0.0, -vgk[x]) * o(irow, igk_col);
    //==             }
    //==         }

    //==         if (kp->num_ranks() == 1)
    //==         {
    //==             // zm1 = H * V
    //==             blas<cpu>::gemm(0, 0, kp->gklo_basis_size(), parameters_.num_fv_states(), kp->gklo_basis_size(), 
    //==                             &vh(0, 0), vh.ld(), &fv_evec(0, 0), fv_evec.ld(), &zm1(0, 0), zm1.ld());
    //==             
    //==             // F = V^{+} * zm1 = V^{+} * H * V
    //==             blas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), kp->gklo_basis_size(),
    //==                             &fv_evec(0, 0), fv_evec.ld(), &zm1(0, 0), zm1.ld(), &zf(0, 0), zf.ld());

    //==             // zm1 = O * V
    //==             blas<cpu>::gemm(0, 0, kp->gklo_basis_size(), parameters_.num_fv_states(), kp->gklo_basis_size(), 
    //==                             &vo(0, 0), vo.ld(), &fv_evec(0, 0), fv_evec.ld(), &zm1(0, 0), zm1.ld());

    //==             // multiply by energy
    //==             for (int i = 0; i < parameters_.num_fv_states(); i++)
    //==             {
    //==                 for (int j = 0; j < kp->gklo_basis_size(); j++) zm1(j, i) = zm1(j, i) * kp->fv_eigen_value(i);
    //==             }

    //==             // F = F - V^{+} * zm1 = F - V^{+} * O * (E*V)
    //==             blas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), kp->gklo_basis_size(),
    //==                             double_complex(-1, 0), &fv_evec(0, 0), fv_evec.ld(), &zm1(0, 0), zm1.ld(), double_complex(1, 0), 
    //==                             &zf(0, 0), zf.ld());

    //==             for (int i = 0; i < parameters_.num_fv_states(); i++)
    //==             {
    //==                 for (int j = 0; j < parameters_.num_fv_states(); j++) 
    //==                     forcek(x, ia) += kp->weight() * real(conj(dm(j, i)) * zf(j, i));
    //==             }
    //==         }
    //==         else
    //==         {
    //==             #ifdef _SCALAPACK_
    //==             // zm1 = H * V
    //==             pblas<cpu>::gemm(0, 0, kp->gklo_basis_size(), parameters_.num_fv_states(), kp->gklo_basis_size(), 
    //==                              double_complex(1, 0), &vh(0, 0), vh.ld(), &fv_evec(0, 0), fv_evec.ld(), 
    //==                              double_complex(0, 0), &zm1(0, 0), zm1.ld(), parameters_.cyclic_block_size(), 
    //==                              parameters_.blacs_context());

    //==             // F = V^{+} * zm1 = V^{+} * H * V
    //==             pblas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), kp->gklo_basis_size(),
    //==                              double_complex(1, 0), &fv_evec(0, 0), fv_evec.ld(), &zm1(0, 0), zm1.ld(), 
    //==                              double_complex(0, 0), &zf(0, 0), zf.ld(), parameters_.cyclic_block_size(), 
    //==                              parameters_.blacs_context());

    //==             // zm1 = O * V
    //==             pblas<cpu>::gemm(0, 0, kp->gklo_basis_size(), parameters_.num_fv_states(), kp->gklo_basis_size(), 
    //==                              double_complex(1, 0), &vo(0, 0), vo.ld(), &fv_evec(0, 0), fv_evec.ld(),
    //==                              double_complex(0, 0), &zm1(0, 0), zm1.ld(), parameters_.cyclic_block_size(), 
    //==                              parameters_.blacs_context());

    //==             // multiply by energy
    //==             for (int i = 0; i < parameters_.spl_fv_states_col().local_size(); i++)
    //==             {
    //==                 int ist = parameters_.spl_fv_states_col(i);
    //==                 for (int j = 0; j < kp->gklo_basis_size_row(); j++) zm1(j, i) = zm1(j, i) * kp->fv_eigen_value(ist);
    //==             }

    //==             // F = F - V^{+} * zm1 = F - V^{+} * O * (E*V)
    //==             pblas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), kp->gklo_basis_size(),
    //==                              double_complex(-1, 0), &fv_evec(0, 0), fv_evec.ld(), &zm1(0, 0), zm1.ld(), 
    //==                              double_complex(1, 0), &zf(0, 0), zf.ld(), parameters_.cyclic_block_size(), 
    //==                              parameters_.blacs_context());

    //==             // TODO: this can be combined with the previous code
    //==             for (int i = 0; i < parameters_.spl_fv_states_col().local_size(); i++)
    //==             {
    //==                 for (int j = 0; j < parameters_.spl_fv_states_row().local_size(); j++) 
    //==                     forcek(x, ia) += kp->weight() * real(conj(dm(j, i)) * zf(j, i));
    //==             }

    //==             #else
    //==             error_local(__FILE__, __LINE__, "not compiled with ScaLAPACK");
    //==             #endif
    //==         }
    //==     }
    //== } //ia
    //== 
    //== if (kp->num_ranks() > 1)
    //== {
    //==     Platform::allreduce(&forcek(0, 0), (int)forcek.size(), 
    //==                         parameters_.mpi_grid().communicator((1 << _dim_row_) | (1 << _dim_col_)));
    //== }
}

void Force::total_force(Global& parameters_, Potential* potential, Density* density, K_set* ks, 
                        mdarray<double, 2>& force)
{
    Timer t("sirius::Force::total_force");

    mdarray<double, 2> ffac(parameters_.reciprocal_lattice()->num_gvec_shells_inner(), parameters_.unit_cell()->num_atom_types());
    parameters_.step_function()->get_step_function_form_factors(ffac);

    force.zero();

    mdarray<double, 2> forcek(3, parameters_.unit_cell()->num_atoms());
    for (int ikloc = 0; ikloc < (int)ks->spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks->spl_num_kpoints(ikloc);
        ibs_force(parameters_, ks->band(), (*ks)[ik], ffac, forcek);
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            for (int x = 0; x < 3; x++) force(x, ia) += forcek(x, ia);
        }
    }
    Platform::allreduce(&force(0, 0), (int)force.size(), parameters_.mpi_grid().communicator(1 << _dim_k_));

    mdarray<double, 2> forcehf(3, parameters_.unit_cell()->num_atoms());

    forcehf.zero();
    for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
        auto g = gradient(potential->hartree_potential_mt(ialoc));
        for (int x = 0; x < 3; x++) forcehf(x, ia) = parameters_.unit_cell()->atom(ia)->type()->zn() * g[x](0, 0) * y00;
    }
    Platform::allreduce(&forcehf(0, 0), (int)forcehf.size());
    
    mdarray<double, 2> forcerho(3, parameters_.unit_cell()->num_atoms());
    forcerho.zero();
    for (int ialoc = 0; ialoc < (int)parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
        auto g = gradient(density->density_mt(ialoc));
        for (int x = 0; x < 3; x++) forcerho(x, ia) = inner(potential->effective_potential_mt(ialoc), g[x]);
    }
    Platform::allreduce(&forcerho(0, 0), (int)forcerho.size());
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        for (int x = 0; x < 3; x++) force(x, ia) += (forcehf(x, ia) + forcerho(x, ia));
    }
}

}


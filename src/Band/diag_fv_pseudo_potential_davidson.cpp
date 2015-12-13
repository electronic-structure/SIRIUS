// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file diag_fv_pseudo_potential.cpp
 *   
 *  \brief Implementation of Davidson iterative solver.
 */

#include "band.h"

namespace sirius {

void Band::diag_fv_pseudo_potential_davidson(K_point* kp__,
                                             double v0__,
                                             std::vector<double>& veff_it_coarse__)
{
    PROFILE();

    Timer t("sirius::Band::diag_fv_pseudo_potential_davidsonl");

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* we need to apply overlap operator in case of ultrasoft pseudopotential */
    bool with_overlap = (parameters_.esm_type() == ultrasoft_pseudopotential);

    /* get diagonal elements for preconditioning */
    std::vector<double> h_diag;
    std::vector<double> o_diag;
    if (with_overlap) 
    {
        get_h_o_diag<true>(kp__, v0__, pw_ekin, h_diag, o_diag);
    } 
    else 
    {
        get_h_o_diag<false>(kp__, v0__, pw_ekin, h_diag, o_diag);
    }

    auto pu = parameters_.processing_unit();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    /* short notation for number of G+k vectors */
    int ngk = kp__->num_gkvec();

    auto& itso = kp__->iterative_solver_input_section_;

    /* short notation for target wave-functions */
    auto& psi = kp__->fv_states<false>();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);
    
    assert(num_bands * 2 < ngk); // iterative subspace size can't be smaller than this

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, ngk);

    /* allocate wave-functions */
    Wave_functions<false> phi(num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> hphi(num_phi, num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> ophi(num_phi, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> hpsi(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    Wave_functions<false> opsi(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);
    /* residuals */
    Wave_functions<false> res(num_bands, kp__->gkvec(), ctx_.mpi_grid_fft(), pu);

    /* allocate Hamiltonian and overlap */
    matrix<double_complex> hmlt(num_phi, num_phi);
    matrix<double_complex> ovlp(num_phi, num_phi);
    matrix<double_complex> hmlt_old(num_phi, num_phi);
    matrix<double_complex> ovlp_old(num_phi, num_phi);

    matrix<double_complex> evec;
    if (converge_by_energy)
    {
        evec = matrix<double_complex>(num_phi, num_bands * 2);
    }
    else
    {
        evec = matrix<double_complex>(num_phi, num_bands);
    }

    int bs = parameters_.cyclic_block_size();

    dmatrix<double_complex> hmlt_dist;
    dmatrix<double_complex> ovlp_dist;
    dmatrix<double_complex> evec_dist;
    if (kp__->comm().size() == 1)
    {
        hmlt_dist = dmatrix<double_complex>(&hmlt(0, 0), num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<double_complex>(&ovlp(0, 0), num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        evec_dist = dmatrix<double_complex>(&evec(0, 0), num_phi, num_bands, kp__->blacs_grid(), bs, bs);
    }
    else
    {
        hmlt_dist = dmatrix<double_complex>(num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<double_complex>(num_phi, num_phi,   kp__->blacs_grid(), bs, bs);
        evec_dist = dmatrix<double_complex>(num_phi, num_bands, kp__->blacs_grid(), bs, bs);
    }

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    kp__->beta_projectors().allocate_workspace();

    D_operator d_op(kp__->beta_projectors(), pu);
    Q_operator q_op(kp__->beta_projectors(), pu);
    Hloc_operator h_op(ctx_.fft_coarse_ctx(), kp__->gkvec(), pw_ekin, veff_it_coarse__);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        psi.allocate_on_device();
        psi.copy_to_device(0, num_bands);

        phi.allocate_on_device();
        res.allocate_on_device();

        hphi.allocate_on_device();
        ophi.allocate_on_device();

        hpsi.allocate_on_device();
        opsi.allocate_on_device();

        evec.allocate_on_device();
    }
    #endif

    /* trial basis functions */
    phi.copy_from(psi, 0, num_bands);

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;
    
    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o(kp__, N, n, phi, hphi, ophi, h_op, d_op, q_op);
        
        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_fv_h_o(kp__, N, n, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);
 
        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve generalized eigen-value problem with the size N */
        diag_h_o(kp__, N, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);

        /* check if occupied bands have converged */
        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++)
        {
            if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval_old[i] - eval[i]) > ctx_.iterative_solver_tolerance()) 
                occ_band_converged = false;
        }

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1 && !occ_band_converged)
        {
            /* get new preconditionined residuals, and also hpss and opsi as a by-product */
            n = residuals(kp__, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag);
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
        {   
            Timer t1("sirius::Band::diag_fv_pseudo_potential|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            psi.transform_from(phi, N, evec, num_bands);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1) || occ_band_converged)
            {
                //if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                //{
                //    double demax = 0;
                //    for (int i = 0; i < num_bands; i++)
                //    {
                //         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                //    }
                //    DUMP("exiting after %i iterations with maximum eigen-value error %18.12f", k + 1, demax);
                //}
                break;
            }
            else /* otherwise, set Psi as a new trial basis */
            {
                hmlt_old.zero();
                ovlp_old.zero();
                for (int i = 0; i < num_bands; i++)
                {
                    hmlt_old(i, i) = eval[i];
                    ovlp_old(i, i) = complex_one;
                }

                /* need to recompute hpsi and opsi */
                if (converge_by_energy)
                {
                    hpsi.transform_from(hphi, N, evec, num_bands);
                    opsi.transform_from(ophi, N, evec, num_bands);
                }
 
                /* update basis functions */
                phi.copy_from(psi, 0, num_bands);
                /* update hphi and ophi */
                hphi.copy_from(hpsi, 0, num_bands);
                ophi.copy_from(opsi, 0, num_bands);
                /* number of basis functions that we already have */
                N = num_bands;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        phi.copy_from(res, 0, n, N);
    }

    kp__->beta_projectors().deallocate_workspace();

    kp__->set_fv_eigen_values(&eval[0]);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU)
    {
        psi.copy_to_host(0, num_bands);
        psi.deallocate_on_device();
    }
    #endif
    kp__->comm().barrier();
}

};

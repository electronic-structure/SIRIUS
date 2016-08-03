// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file diag_pseudo_potential_davidson.cpp
 *   
 *  \brief Implementation of Davidson iterative solver.
 */

#include "band.h"

namespace sirius {

template <typename T>
void Band::diag_pseudo_potential_davidson(K_point* kp__,
                                          int ispn__,
                                          Hloc_operator& h_op__,
                                          D_operator<T>& d_op__,
                                          Q_operator<T>& q_op__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_davidson");

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #endif

    /* get diagonal elements for preconditioning */
    auto h_diag = get_h_diag(kp__, ispn__, h_op__.v0(ispn__), d_op__);
    auto o_diag = get_o_diag(kp__, q_op__);

    auto pu = ctx_.processing_unit();

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto& itso = ctx_.iterative_solver_input_section();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions<false>(ispn__);

    bool converge_by_energy = (itso.converge_by_energy_ == 1);
    
    assert(num_bands * 2 < kp__->num_gkvec()); // iterative subspace size can't be smaller than this

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    /* allocate wave-functions */
    Wave_functions<false>  phi(kp__->num_gkvec_loc(), num_phi, pu);
    Wave_functions<false> hphi(kp__->num_gkvec_loc(), num_phi, pu);
    Wave_functions<false> ophi(kp__->num_gkvec_loc(), num_phi, pu);
    Wave_functions<false> hpsi(kp__->num_gkvec_loc(), num_bands, pu);
    Wave_functions<false> opsi(kp__->num_gkvec_loc(), num_bands, pu);
    /* residuals */
    Wave_functions<false> res(kp__->num_gkvec_loc(), num_bands, pu);

    /* allocate Hamiltonian and overlap */
    matrix<T> hmlt(num_phi, num_phi);
    matrix<T> ovlp(num_phi, num_phi);
    matrix<T> hmlt_old(num_phi, num_phi);
    matrix<T> ovlp_old(num_phi, num_phi);

    #ifdef __GPU
    if (gen_evp_solver_->type() == ev_magma)
    {
        hmlt.pin_memory();
        ovlp.pin_memory();
    }
    #endif

    matrix<T> evec(num_phi, num_phi);

    int bs = ctx_.cyclic_block_size();

    dmatrix<T> hmlt_dist;
    dmatrix<T> ovlp_dist;
    dmatrix<T> evec_dist;
    if (kp__->comm().size() == 1)
    {
        hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(&evec(0, 0), num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    }
    else
    {
        hmlt_dist = dmatrix<T>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        ovlp_dist = dmatrix<T>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
        evec_dist = dmatrix<T>(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    }

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) eval[i] = kp__->band_energy(i);
    std::vector<double> eval_old(num_bands);
    
    kp__->beta_projectors().prepare();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
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

        ovlp.allocate_on_device();
    }
    #endif

    /* trial basis functions */
    phi.copy_from(psi, 0, num_bands);

    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    #if (__VERBOSITY > 2)
    if (kp__->comm().rank() == 0)
    {
        DUMP("iterative solver tolerance: %18.12f", ctx_.iterative_solver_tolerance());
    }
    #endif
    
    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o<T>(kp__, ispn__, N, n, phi, hphi, ophi, h_op__, d_op__, q_op__);
        
        orthogonalize<T>(kp__, N, n, phi, hphi, ophi, ovlp);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_h_o<T>(kp__, N, n, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);

        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve generalized eigen-value problem with the size N */
        diag_h_o<T>(kp__, N, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
        
        #if (__VERBOSITY > 2)
        if (kp__->comm().rank() == 0)
        {
            DUMP("step: %i, current subspace size: %i, maximum subspace size: %i", k, N, num_phi);
            for (int i = 0; i < num_bands; i++) DUMP("eval[%i]=%20.16f, diff=%20.16f", i, eval[i], std::abs(eval[i] - eval_old[i]));
        }
        #endif

        /* check if occupied bands have converged */
        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++)
        {
            if (kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) > 1e-2 &&
                std::abs(eval_old[i] - eval[i]) > ctx_.iterative_solver_tolerance()) 
            {
                occ_band_converged = false;
            }
        }

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1 && !occ_band_converged)
        {
            /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
            n = residuals<T>(kp__, ispn__, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag);
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1) || occ_band_converged)
        {   
            runtime::Timer t1("sirius::Band::diag_pseudo_potential_davidson|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            psi.transform_from<T>(phi, N, evec, num_bands);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1) || occ_band_converged)
            {
                break;
            }
            else /* otherwise, set Psi as a new trial basis */
            {
                #if (__VERBOSITY > 2)
                if (kp__->comm().rank() == 0)
                {
                    DUMP("subspace size limit reached");
                }
                #endif
                hmlt_old.zero();
                ovlp_old.zero();
                for (int i = 0; i < num_bands; i++)
                {
                    hmlt_old(i, i) = eval[i];
                    ovlp_old(i, i) = 1.0;
                }

                /* need to compute all hpsi and opsi states (not only unconverged) */
                if (converge_by_energy)
                {
                    hpsi.transform_from<T>(hphi, N, evec, num_bands);
                    opsi.transform_from<T>(ophi, N, evec, num_bands);
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

    kp__->beta_projectors().dismiss();

    for (int j = 0; j < ctx_.num_fv_states(); j++)
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU)
    {
        psi.copy_to_host(0, num_bands);
        psi.deallocate_on_device();
    }
    #endif
    kp__->comm().barrier();
}

/* explicit instantiation for general k-point solver */
template void Band::diag_pseudo_potential_davidson<double_complex>(K_point* kp__,
                                                                   int ispn__,
                                                                   Hloc_operator& h_op__,
                                                                   D_operator<double_complex>& d_op__,
                                                                   Q_operator<double_complex>& q_op__) const;
/* explicit instantiation for gamma-point solver */
template void Band::diag_pseudo_potential_davidson<double>(K_point* kp__,
                                                           int ispn__,
                                                           Hloc_operator& h_op__,
                                                           D_operator<double>& d_op__,
                                                           Q_operator<double>& q_op__) const;
};

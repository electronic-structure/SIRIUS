#include "band.h"

namespace sirius {

void Band::solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3]) const
{
    PROFILE_WITH_TIMER("sirius::Band::solve_sv");

    if (!ctx_.need_sv()) {
        kp->bypass_sv();
        return;
    }

    if (kp->num_ranks() > 1 && !std_evp_solver()->parallel()) {
        TERMINATE("eigen-value solver is not parallel");
    }

    ctx_.fft().prepare(ctx_.gvec().partition());

    /* number of h|\psi> components */
    int nhpsi = (ctx_.num_mag_dims() == 3) ? 3 : ctx_.num_spins();

    /* size of the first-variational state */
    int fvsz = kp->wf_size();

    std::vector<double> band_energies(ctx_.num_bands());

    /* product of the second-variational Hamiltonian and a first-variational wave-function */
    std::vector<Wave_functions<true>*> hpsi;
    for (int i = 0; i < nhpsi; i++) {
        hpsi.push_back(new Wave_functions<true>(kp->wf_size(), ctx_.num_fv_states(), ctx_.cyclic_block_size(),
                                                ctx_.blacs_grid(), ctx_.blacs_grid_slice()));
    }

    /* compute product of magnetic field and wave-function */
    if (ctx_.num_spins() == 2) {
        apply_magnetic_field(kp->fv_states<true>(), kp->gkvec(), effective_magnetic_field, hpsi);
    }
    else {
        hpsi[0]->set_num_swapped(ctx_.num_fv_states());
        std::memset((*hpsi[0])[0], 0, kp->wf_size() * hpsi[0]->spl_num_col().local_size() * sizeof(double_complex));
    }

    //== if (ctx_.uj_correction())
    //== {
    //==     apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
    //==     if (ctx_.num_mag_dims() == 3) 
    //==     {
    //==         apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
    //==         if (ctx_.std_evp_solver()->parallel()) apply_uj_correction<du>(kp->fv_states_col(), hpsi);
    //==     }
    //== }

    //== if (ctx_.so_correction()) apply_so_correction(kp->fv_states_col(), hpsi);

    for (auto e: hpsi) {
        e->swap_backward(0, ctx_.num_fv_states());
    }

    if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
        #ifdef __GPU
        kp->fv_states<true>().coeffs().allocate(memory_t::device);
        kp->fv_states<true>().coeffs().copy_to_device();
        #endif
    }
 
    #ifdef __GPU
    double_complex alpha = complex_one;
    double_complex beta = complex_zero;
    #endif


    int nfv = ctx_.num_fv_states();
    int bs = ctx_.cyclic_block_size();

    if (ctx_.num_mag_dims() != 3) {
        dmatrix<double_complex> h(nfv, nfv, ctx_.blacs_grid(), bs, bs);
        //h.allocate(alloc_mode);

        if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
            #ifdef __GPU
            h.allocate(memory_t::device);
            #endif
        }

        /* perform one or two consecutive diagonalizations */
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
                #ifdef __GPU
                runtime::Timer t4("sirius::Band::solve_sv|zgemm");
                hpsi[ispn]->coeffs().allocate(memory_t::device);
                hpsi[ispn]->coeffs().copy_to_device();
                linalg<GPU>::gemm(2, 0, nfv, nfv, fvsz,
                                  &alpha, 
                                  kp->fv_states<true>().coeffs().at<GPU>(), kp->fv_states<true>().coeffs().ld(),
                                  hpsi[ispn]->coeffs().at<GPU>(), hpsi[ispn]->coeffs().ld(),
                                  &beta,
                                  h.at<GPU>(), h.ld());
                h.copy_to_host();
                hpsi[ispn]->coeffs().deallocate_on_device();
                double tval = t4.stop();
                DUMP("effective zgemm performance: %12.6f GFlops", 
                     8e-9 * ctx_.num_fv_states() * ctx_.num_fv_states() * fvsz / tval);
                #else
                TERMINATE_NO_GPU
                #endif
            } else {
                /* compute <wf_i | h * wf_j> for up-up or dn-dn block */
                linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().prime(), hpsi[ispn]->prime(),
                                  complex_zero, h);
            }
            
            for (int i = 0; i < nfv; i++) {
                h.add(i, i, kp->fv_eigen_value(i));
            }
        
            runtime::Timer t1("sirius::Band::solve_sv|stdevp");
            std_evp_solver()->solve(nfv, h.at<CPU>(), h.ld(), &band_energies[ispn * nfv],
                                    kp->sv_eigen_vectors(ispn).at<CPU>(), kp->sv_eigen_vectors(ispn).ld());
        }
    } else {
        int nb = ctx_.num_bands();
        dmatrix<double_complex> h(nb, nb, ctx_.blacs_grid(), bs, bs);

        /* compute <wf_i | h * wf_j> for up-up block */
        linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().prime(), 0, 0, hpsi[0]->prime(), 0, 0,
                          complex_zero, h, 0, 0);
        /* compute <wf_i | h * wf_j> for dn-dn block */
        linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().prime(), 0, 0, hpsi[1]->prime(), 0, 0,
                          complex_zero, h, nfv, nfv);
        /* compute <wf_i | h * wf_j> for up-dn block */
        linalg<CPU>::gemm(2, 0, nfv, nfv, fvsz, complex_one, kp->fv_states<true>().prime(), 0, 0, hpsi[2]->prime(), 0, 0,
                          complex_zero, h, 0, nfv);
        
        for (int i = 0; i < nfv; i++) {
            h.add(i,       i,       kp->fv_eigen_value(i));
            h.add(i + nfv, i + nfv, kp->fv_eigen_value(i));
        }
        runtime::Timer t1("sirius::Band::solve_sv|stdevp");
        std_evp_solver()->solve(nb, h.at<CPU>(), h.ld(), &band_energies[0],
                                kp->sv_eigen_vectors(0).at<CPU>(), kp->sv_eigen_vectors(0).ld());
        
    }
 
    if (ctx_.processing_unit() == GPU && kp->num_ranks() == 1) {
        #ifdef __GPU
        kp->fv_states<true>().coeffs().deallocate_on_device();
        #endif
    }
 
    for (auto e: hpsi) {
        delete e;
    }

    ctx_.fft().dismiss();

    kp->set_band_energies(&band_energies[0]);
}

};

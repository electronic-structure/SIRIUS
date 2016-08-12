#include "density.h"

namespace sirius {

void Density::generate_valence(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence");

    double wt = 0.0;
    double occ_val = 0.0;
    for (int ik = 0; ik < ks__.num_kpoints(); ik++) {
        wt += ks__[ik]->weight();
        for (int j = 0; j < ctx_.num_bands(); j++) {
            occ_val += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
        }
    }

    if (std::abs(wt - 1.0) > 1e-12) {
        TERMINATE("K_point weights don't sum to one");
    }

    if (std::abs(occ_val - unit_cell_.num_valence_electrons()) > 1e-8) {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << occ_val << std::endl
          << "  required : " << unit_cell_.num_valence_electrons() << std::endl
          << "  difference : " << std::abs(occ_val - unit_cell_.num_valence_electrons());
        WARNING(s);
    }

    /* swap wave functions */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            if (ctx_.full_potential()) {
                kp->spinor_wave_functions<true>(ispn).swap_forward(0, kp->num_occupied_bands(ispn));
            } else {
                kp->spinor_wave_functions<false>(ispn).swap_forward(0, kp->num_occupied_bands(ispn), kp->gkvec_fft_distr());
            }
        }
    }

    /* density matrix is required for lapw, uspp and paw */
    if (ctx_.esm_type() == electronic_structure_method_t::full_potential_lapwlo ||
        ctx_.esm_type() == electronic_structure_method_t::ultrasoft_pseudopotential ||
        ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
        generate_density_matrix(ks__);
    }

    //== for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
    //==     printf("ia=%i\n", ia);
    //==     int nbf = unit_cell_.atom(ia).mt_basis_size();

    //==     for (int xi1 = 0; xi1 < nbf; xi1++) {
    //==         for (int xi2 = 0; xi2 < nbf; xi2++) {
    //==             printf("%12.8f ", density_matrix_(xi1, xi2, 0, ia).real());
    //==         }
    //==         printf("\n");
    //==     }
    //== }

    //== printf("\n");
    //==if (ctx_.esm_type() == electronic_structure_method_t::paw_pseudopotential) {
    //==    symmetrize_density_matrix();
    //==}

    //==printf("=== Density matrix ===\n");
    //==for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
    //==    printf("ia=%i\n", ia);
    //==    int nbf = unit_cell_.atom(ia).mt_basis_size();

    //==    for (int xi1 = 0; xi1 < nbf; xi1++) {
    //==        for (int xi2 = 0; xi2 < nbf; xi2++) {
    //==            printf("%12.8f ", density_matrix_(xi1, xi2, 0, ia).real());
    //==        }
    //==        printf("\n");
    //==    }
    //==}

    /* zero density and magnetization */
    zero();

    /* interstitial part is independent of basis type */
    generate_valence_density_it(ks__);

    /* for muffin-tin part */
    switch (ctx_.esm_type())
    {
        case full_potential_lapwlo:
        {
            generate_valence_density_mt(ks__);
            break;
        }
        case full_potential_pwlo:
        {
            STOP();
        }
        default:
        {
            break;
        }
    }
    
    //== double nel = 0;
    //== for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
    //== {
    //==     nel += rho_->f_rg(ir);
    //== }
    //== ctx_.mpi_grid().communicator(1 << _dim_row_).allreduce(&nel, 1);
    //== nel = nel * unit_cell_.omega() / ctx_.fft().size();
    //== printf("number of electrons: %f\n", nel);

    ctx_.fft().prepare(ctx_.gvec_fft_distr());
    
    /* get rho(G) and mag(G)
     * they are required to symmetrize density and magnetization */
    rho_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++)
    {
        magnetization_[j]->fft_transform(-1);
    }

    //== printf("number of electrons: %f\n", rho_->f_pw(0).real() * unit_cell_.omega());
    //== STOP();

    ctx_.fft().dismiss();

    if (ctx_.esm_type() == ultrasoft_pseudopotential)
    {
        augment(ks__);
    }

    if (ctx_.esm_type() == paw_pseudopotential)
    {
        augment(ks__);
        symmetrize_density_matrix();
        //generate_paw_loc_density();
    }
}

};

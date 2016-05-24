#include "density.h"

namespace sirius {

void Density::generate_valence(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence");

    double wt = 0.0;
    double ot = 0.0;
    for (int ik = 0; ik < ks__.num_kpoints(); ik++)
    {
        wt += ks__[ik]->weight();
        for (int j = 0; j < ctx_.num_bands(); j++) ot += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
    }

    if (std::abs(wt - 1.0) > 1e-12) TERMINATE("K_point weights don't sum to one");

    if (std::abs(ot - unit_cell_.num_valence_electrons()) > 1e-8)
    {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << ot << std::endl
          << "  required : " << unit_cell_.num_valence_electrons() << std::endl
          << "  difference : " << std::abs(ot - unit_cell_.num_valence_electrons());
        WARNING(s);
    }

    /* swap wave functions */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
        {
            if (ctx_.full_potential())
            {
                kp->spinor_wave_functions<true>(ispn).swap_forward(0, kp->num_occupied_bands(ispn));
            }
            else
            {
                kp->spinor_wave_functions<false>(ispn).swap_forward(0, kp->num_occupied_bands(ispn), kp->gkvec_fft_distr());
            }
        }
    }

    /* zero density and magnetization */
    zero();

    ctx_.fft().prepare();

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
    
    #if (__VERIFICATION > 0)
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        if (rho_->f_rg(ir) < 0) {
            TERMINATE("density is wrong");
        }
    }
    #endif

    //== double nel = 0;
    //== for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
    //== {
    //==     nel += rho_->f_rg(ir);
    //== }
    //== ctx_.mpi_grid().communicator(1 << _dim_row_).allreduce(&nel, 1);
    //== nel = nel * unit_cell_.omega() / ctx_.fft().size();
    //== printf("number of electrons: %f\n", nel);
    
    /* get rho(G) and mag(G) */
    rho_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) magnetization_[j]->fft_transform(-1);

    //== printf("number of electrons: %f\n", rho_->f_pw(0).real() * unit_cell_.omega());
    //== STOP();

    ctx_.fft().dismiss();

    if (ctx_.esm_type() == ultrasoft_pseudopotential) augment(ks__);
}

};

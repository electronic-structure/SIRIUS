#include "density.h"

namespace sirius {

void Density::generate_valence(K_set& ks__)
{
    Timer t("sirius::Density::generate_valence", ctx_.comm());
    
    double wt = 0.0;
    double ot = 0.0;
    for (int ik = 0; ik < ks__.num_kpoints(); ik++)
    {
        wt += ks__[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++) ot += ks__[ik]->weight() * ks__[ik]->band_occupancy(j);
    }

    if (std::abs(wt - 1.0) > 1e-12) error_local(__FILE__, __LINE__, "K_point weights don't sum to one");

    if (std::abs(ot - unit_cell_.num_valence_electrons()) > 1e-8)
    {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << ot << std::endl
          << "  required : " << unit_cell_.num_valence_electrons() << std::endl
          << "  difference : " << fabs(ot - unit_cell_.num_valence_electrons());
        warning_local(__FILE__, __LINE__, s);
    }
    
    /* zero density and magnetization */
    zero();

    /* interstitial part is independent of basis type */
    generate_valence_density_it(ks__);

    /* for muffin-tin part */
    switch (parameters_.esm_type())
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
    for (int ir = 0; ir < ctx_.fft(0)->local_size(); ir++)
    {
        if (rho_->f_it(ir) < 0) TERMINATE("density is wrong");
    }
    #endif
    //== double nel = 0;
    //== for (int ir = 0; ir < ctx_.fft(0)->local_size(); ir++)
    //== {
    //==     nel += rho_->f_it(ir);
    //== }
    //== ctx_.mpi_grid().communicator(1 << _dim_row_).allreduce(&nel, 1);
    //== nel = nel * unit_cell_.omega() / ctx_.fft(0)->size();
    //== printf("number of electrons: %f\n", nel);
    
    /* get rho(G) */
    rho_->fft_transform(-1);

    //== printf("number of electrons: %f\n", rho_->f_pw(0).real() * unit_cell_.omega());
    //== STOP();

    if (parameters_.esm_type() == ultrasoft_pseudopotential) augment(ks__);
}

};

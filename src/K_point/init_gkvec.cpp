#include "k_point.h"

namespace sirius {

void K_point::init_gkvec_ylm_and_len(int lmax__, int num_gkvec__, std::vector<gklo_basis_descriptor>& desc__)
{
    gkvec_ylm_ = mdarray<double_complex, 2>(Utils::lmmax(lmax__), num_gkvec__);

    #pragma omp parallel for default(shared)
    for (int i = 0; i < num_gkvec__; i++)
    {
        int igk = desc__[i].igk;

        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(gkvec<cartesian>(igk));
        
        SHT::spherical_harmonics(lmax__, vs[1], vs[2], &gkvec_ylm_(0, i));
    }
}

void K_point::init_gkvec_phase_factors(int num_gkvec__, std::vector<gklo_basis_descriptor>& desc__)
{
    gkvec_phase_factors_ = mdarray<double_complex, 2>(num_gkvec__, unit_cell_.num_atoms());

    #pragma omp parallel for default(shared)
    for (int i = 0; i < num_gkvec__; i++)
    {
        int igk = desc__[i].igk;

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        {
            double phase = twopi * (gkvec<fractional>(igk) * unit_cell_.atom(ia)->position());

            gkvec_phase_factors_(i, ia) = std::exp(double_complex(0.0, phase));
        }
    }
}

void K_point::init_gkvec()
{
    //== int lmax = - 1;
    //== switch (parameters_.esm_type())
    //== {
    //==     case full_potential_lapwlo:
    //==     {
    //==         lmax = parameters_.lmax_apw();
    //==         break;
    //==     }
    //==     case full_potential_pwlo:
    //==     {
    //==         lmax = parameters_.lmax_pw();
    //==         break;
    //==     }
    //==     case ultrasoft_pseudopotential:
    //==     case norm_conserving_pseudopotential:
    //==     {
    //==         if (num_gkvec() != wf_size()) TERMINATE("wrong size of wave-functions");
    //==         lmax = parameters_.lmax_beta();
    //==         break;
    //==     }
    //== }
    
    //== init_gkvec_ylm_and_len(lmax, num_gkvec_row(), gklo_basis_descriptors_row_);
    init_gkvec_phase_factors(num_gkvec_row(), gklo_basis_descriptors_row_);
}

};

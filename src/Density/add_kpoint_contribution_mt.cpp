#include "density.h"

namespace sirius {

void Density::add_kpoint_contribution_mt(K_point* kp__,
                                         std::vector< std::pair<int, double> >& occupied_bands__, 
                                         mdarray<double_complex, 4>& mt_complex_density_matrix__)
{
    Timer t("sirius::Density::add_kpoint_contribution_mt");
    
    if (occupied_bands__.size() == 0) return;
   
    mdarray<double_complex, 3> wf1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands__.size(), parameters_.num_spins());
    mdarray<double_complex, 3> wf2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands__.size(), parameters_.num_spins());

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
        int mt_basis_size = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        
        for (int i = 0; i < (int)occupied_bands__.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                for (int j = 0; j < mt_basis_size; j++)
                {
                    wf1(j, i, ispn) = conj(kp__->spinor_wave_function(offset_wf + j, occupied_bands__[i].first, ispn));
                    wf2(j, i, ispn) = kp__->spinor_wave_function(offset_wf + j, occupied_bands__[i].first, ispn) * occupied_bands__[i].second;
                }
            }
        }

        for (int j = 0; j < (int)mt_complex_density_matrix__.size(2); j++)
        {
            linalg<CPU>::gemm(0, 1, mt_basis_size, mt_basis_size, (int)occupied_bands__.size(), complex_one, 
                              &wf1(0, 0, dmat_spins_[j].first), wf1.ld(), 
                              &wf2(0, 0, dmat_spins_[j].second), wf2.ld(), complex_one, 
                              mt_complex_density_matrix__.at<CPU>(0, 0, j, ia), mt_complex_density_matrix__.ld());
        }
    }
}

void Density::add_kpoint_contribution_pp(K_point* kp__, 
                                         std::vector< std::pair<int, double> >& occupied_bands__, 
                                         mdarray<double_complex, 4>& pp_complex_density_matrix__)
{
    Timer t("sirius::Density::add_kpoint_contribution_pp");

    int nbnd = num_occupied_bands(kp__);

    if (nbnd == 0) return;

    auto uc = parameters_.unit_cell();

    /* compute <beta|Psi> */
    Timer t1("sirius::Density::add_kpoint_contribution_pp|beta_psi");
    matrix<double_complex> beta_psi(uc->mt_basis_size(), nbnd);
    linalg<CPU>::gemm(2, 0, uc->mt_basis_size(), nbnd, kp__->num_gkvec_loc(), complex_one, 
                      kp__->beta_gk(), kp__->fv_states_slab(), complex_zero, beta_psi);
    kp__->comm().allreduce(&beta_psi(0, 0), (int)beta_psi.size());
    t1.stop();

    splindex<block> spl_bands(nbnd, kp__->comm().size(), kp__->comm().rank());

    if (spl_bands.local_size())
    {
        #pragma omp parallel
        {
            /* auxiliary arrays */
            mdarray<double_complex, 2> bp1(uc->max_mt_basis_size(), spl_bands.local_size());
            mdarray<double_complex, 2> bp2(uc->max_mt_basis_size(), spl_bands.local_size());
            #pragma omp for
            for (int ia = 0; ia < uc->num_atoms(); ia++)
            {   
                /* number of beta functions for a given atom */
                int nbf = uc->atom(ia)->mt_basis_size();

                for (int i = 0; i < (int)spl_bands.local_size(); i++)
                {
                    int j = (int)spl_bands[i];
                    for (int xi = 0; xi < nbf; xi++)
                    {
                        bp1(xi, i) = beta_psi(uc->atom(ia)->offset_lo() + xi, j);
                        bp2(xi, i) = conj(bp1(xi, i)) * kp__->band_occupancy(j) * kp__->weight();
                    }
                }

                linalg<CPU>::gemm(0, 1, nbf, nbf, (int)spl_bands.local_size(), complex_one, &bp1(0, 0), bp1.ld(),
                                  &bp2(0, 0), bp2.ld(), complex_one, &pp_complex_density_matrix__(0, 0, 0, ia), 
                                  pp_complex_density_matrix__.ld());
            }
        }
    }
}

};

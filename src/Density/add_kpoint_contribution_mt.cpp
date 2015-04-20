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

};

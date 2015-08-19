#include "k_point.h"

namespace sirius {

void K_point:: generate_beta_gk(int num_atoms__,
                                mdarray<double, 2>& atom_pos__,
                                mdarray<int, 2> const& beta_desc__,
                                matrix<double_complex>& beta_gk__)
{
    Timer t("sirius::K_point::generate_beta_gk");

    if (parameters_.processing_unit() == CPU)
    {
        /* create beta projectors */
        #pragma omp parallel
        for (int i = 0; i < num_atoms__; i++)
        {
            int ia = beta_desc__(3, i);
            #pragma omp for
            for (int xi = 0; xi < beta_desc__(0, i); xi++)
            {
                for (int igk_row = 0; igk_row < num_gkvec_row(); igk_row++)
                {
                    beta_gk__(igk_row, beta_desc__(1, i) + xi) = 
                        beta_gk_t_(igk_row, beta_desc__(2, i) + xi) * std::conj(gkvec_phase_factor(igk_row, ia));
                }
            }
        }
    }
    if (parameters_.processing_unit() == GPU)
    {
        #ifdef __GPU
        /* create beta projectors directly on GPU */
        create_beta_gk_gpu(num_atoms__,
                           num_gkvec_row(),
                           beta_desc__.at<GPU>(),
                           beta_gk_t_.at<GPU>(),
                           gkvec_row_.at<GPU>(),
                           atom_pos__.at<GPU>(),
                           beta_gk__.at<GPU>());
        #else
        TERMINATE_NO_GPU
        #endif
    }
}

};

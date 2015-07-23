#include "density.h"

namespace sirius {

void Density::generate_valence_density_it(K_set& ks)
{
    Timer t("sirius::Density::generate_valence_density_it", ctx_.comm());

    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        auto occupied_bands = ks[ik]->get_occupied_bands_list();
        add_k_point_contribution_it(ks[ik], occupied_bands);
    }
    
    /* reduce arrays; assume that each rank did it's own fraction of the density */
    ctx_.comm().allreduce(&rho_->f_it<global>(0), fft_->size()); 
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        ctx_.comm().allreduce(&magnetization_[j]->f_it<global>(0), fft_->size()); 
}

};

#include "density.h"

namespace sirius {

void Density::generate_valence_density_it(K_set& ks__)
{
    Timer t("sirius::Density::generate_valence_density_it", ctx_.comm());

    /* add k-point contribution */
    for (int ikloc = 0; ikloc < (int)ks__.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];
        occupied_bands_descriptor occupied_bands;

        if (fft_->parallel())
        {
            occupied_bands = kp->get_occupied_bands_list(kp->blacs_grid().comm_col());
        }
        else
        {
            occupied_bands = kp->get_occupied_bands_list(kp->blacs_grid_slice().comm_col());
        }

        if (!parameters_.full_potential())
        {
            if (kp->num_ranks() > 1 && !fft_->parallel())
            {
                linalg<CPU>::gemr2d(kp->wf_size(), occupied_bands.num_occupied_bands(),
                                    kp->fv_states(), 0, 0,
                                    kp->fv_states_slice(), 0, 0,
                                    kp->blacs_grid().context());
            }
            if (fft_->parallel())
            {
                Timer t1("fft|comm");
                linalg<CPU>::gemr2d(kp->wf_size(), occupied_bands.num_occupied_bands(),
                                    kp->fv_states(), 0, 0,
                                    kp->spinor_wave_functions(0), 0, 0,
                                    kp->blacs_grid().context());
                t1.stop();
            }
        }

        if (fft_->parallel())
        {
            add_k_point_contribution_it_pfft(ks__[ik], occupied_bands);
        }
        else
        {
            add_k_point_contribution_it(ks__[ik], occupied_bands);
        }
    }

    /* reduce arrays; assume that each rank did it's own fraction of the density */
    if (fft_->parallel())
    {
        ctx_.mpi_grid().communicator(1 << _dim_k_ | 1 << _dim_col_).allreduce(&rho_->f_it(0), fft_->local_size()); 
    }
    else
    {
        ctx_.comm().allreduce(&rho_->f_it(0), fft_->size()); 
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            ctx_.comm().allreduce(&magnetization_[j]->f_it(0), fft_->size()); 
    }
}

};

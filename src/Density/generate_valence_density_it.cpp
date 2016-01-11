#include "density.h"

namespace sirius {

void Density::generate_valence_density_it(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence_density_it");

    ctx_.fft_ctx().prepare();

    /* add k-point contribution */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];
        
        if (ctx_.full_potential())
        {
            add_k_point_contribution_it<true>(kp);
        }
        else
        {
            add_k_point_contribution_it<false>(kp);
        }
    }

    /* reduce arrays; assume that each rank did it's own fraction of the density */
    auto& comm = (ctx_.fft(0)->parallel()) ? ctx_.mpi_grid().communicator(1 << _dim_k_ | 1 << _dim_col_)
                                           : ctx_.comm();

    comm.allreduce(&rho_->f_rg(0), ctx_.fft(0)->local_size()); 
    for (int j = 0; j < ctx_.num_mag_dims(); j++)
        comm.allreduce(&magnetization_[j]->f_rg(0), ctx_.fft(0)->local_size()); 

    ctx_.fft_ctx().dismiss();
}

};

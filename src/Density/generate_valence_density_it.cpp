#include "density.h"

namespace sirius {

void Density::generate_valence_density_it(K_set& ks__)
{
    PROFILE_WITH_TIMER("sirius::Density::generate_valence_density_it");

    /* add k-point contribution */
    for (int ikloc = 0; ikloc < ks__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = ks__.spl_num_kpoints(ikloc);
        auto kp = ks__[ik];
        
        if (ctx_.full_potential()) {
            add_k_point_contribution_rg<true>(kp);
        } else {
            add_k_point_contribution_rg<false>(kp);
        }
    }

    /* reduce arrays; assume that each rank did its own fraction of the density */
    auto& comm = (ctx_.fft().parallel()) ? ctx_.mpi_grid().communicator(1 << _mpi_dim_k_ | 1 << _mpi_dim_k_col_)
                                         : ctx_.comm();

    comm.allreduce(&rho_->f_rg(0), ctx_.fft().local_size()); 
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        comm.allreduce(&magnetization_[j]->f_rg(0), ctx_.fft().local_size()); 
    }

    #if (__VERIFICATION > 0)
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        if (rho_->f_rg(ir) < 0) {
            TERMINATE("density is wrong");
        }
    }
    #endif
}

};

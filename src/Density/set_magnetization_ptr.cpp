#include "density.h"

namespace sirius {

void Density::set_magnetization_ptr(double* magmt, double* magir)
{
    if (ctx_.num_mag_dims() == 0) return;
    assert(ctx_.num_spins() == 2);

    // set temporary array wrapper
    mdarray<double, 4> magmt_tmp(magmt, ctx_.lmmax_rho(), unit_cell_.max_num_mt_points(), 
                                 unit_cell_.num_atoms(), ctx_.num_mag_dims());
    mdarray<double, 2> magir_tmp(magir, ctx_.fft().size(), ctx_.num_mag_dims());
    
    if (ctx_.num_mag_dims() == 1)
    {
        // z component is the first and only one
        magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[0]->set_rg_ptr(&magir_tmp(0, 0));
    }

    if (ctx_.num_mag_dims() == 3)
    {
        // z component is the first
        magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 2));
        magnetization_[0]->set_rg_ptr(&magir_tmp(0, 2));
        // x component is the second
        magnetization_[1]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[1]->set_rg_ptr(&magir_tmp(0, 0));
        // y component is the third
        magnetization_[2]->set_mt_ptr(&magmt_tmp(0, 0, 0, 1));
        magnetization_[2]->set_rg_ptr(&magir_tmp(0, 1));
    }
}

}

#include "atom.h"

namespace sirius {

void Atom::init(int offset_aw__, int offset_lo__, int offset_wf__)
{
    assert(offset_aw__ >= 0);
    
    offset_aw_ = offset_aw__;
    offset_lo_ = offset_lo__;
    offset_wf_ = offset_wf__;

    lmax_pot_ = type().parameters().lmax_pot();
    num_mag_dims_ = type().parameters().num_mag_dims();

    if (type().parameters().full_potential())
    {
        int lmmax = Utils::lmmax(lmax_pot_);

        h_radial_integrals_ = mdarray<double, 3>(lmmax, type().indexr().size(), type().indexr().size());
        
        b_radial_integrals_ = mdarray<double, 4>(lmmax, type().indexr().size(), type().indexr().size(), num_mag_dims_);
        
        occupation_matrix_ = mdarray<double_complex, 4>(16, 16, 2, 2);
        
        uj_correction_matrix_ = mdarray<double_complex, 4>(16, 16, 2, 2);
    }

    if (!type().parameters().full_potential())
    {
        int nbf = type().mt_lo_basis_size();
        d_mtrx_ = mdarray<double_complex, 3>(nbf, nbf, num_mag_dims_ + 1);
        d_mtrx_.zero();

        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            int lm2 = type().indexb(xi2).lm;
            int idxrf2 = type().indexb(xi2).idxrf;
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                int lm1 = type().indexb(xi1).lm;
                int idxrf1 = type().indexb(xi1).idxrf;
                if (lm1 == lm2) d_mtrx_(xi1, xi2, 0) = type().uspp().d_mtrx_ion(idxrf1, idxrf2);
            }
        }
    }
}

}

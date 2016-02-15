#include "unit_cell.h"

namespace sirius {

std::vector<double_complex> Unit_cell::make_periodic_function(mdarray<double, 2>& form_factors__, Gvec const& gvec__) const
{
    PROFILE_WITH_TIMER("sirius::Unit_cell::make_periodic_function");

    assert((int)form_factors__.size(0) == num_atom_types());
    
    std::vector<double_complex> f_pw(gvec__.num_gvec(), double_complex(0, 0));

    double fourpi_omega = fourpi / omega();

    splindex<block> spl_ngv(gvec__.num_gvec(), comm_.size(), comm_.rank());

    #pragma omp parallel for
    for (int igloc = 0; igloc < spl_ngv.local_size(); igloc++)
    {
        int ig = spl_ngv[igloc];
        int igs = gvec__.shell(ig);

        for (int ia = 0; ia < num_atoms(); ia++)
        {            
            int iat = atom(ia).type_id();
            double_complex z = std::exp(double_complex(0.0, twopi * (gvec__[ig] * atom(ia).position())));
            f_pw[ig] += fourpi_omega * std::conj(z) * form_factors__(iat, igs);
        }
    }

    comm_.allgather(&f_pw[0], spl_ngv.global_offset(), spl_ngv.local_size());

    return std::move(f_pw);
}

}

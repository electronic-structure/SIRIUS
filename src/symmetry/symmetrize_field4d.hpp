#ifndef __SYMMETRIZE_FIELD4D_HPP__
#define __SYMMETRIZE_FIELD4D_HPP__

#include "symmetrize_mt_function.hpp"
#include "symmetrize_pw_function.hpp"

namespace sirius {

inline void
symmetrize_field4d(Field4D& f__)
{
    auto& ctx = f__.ctx();

    /* quick exit: the only symmetry operation is identity */
    if (ctx.unit_cell().symmetry().size() == 1) {
        return;
    }

    /* symmetrize PW components */
    symmetrize_pw_function(ctx.unit_cell().symmetry(), ctx.remap_gvec(), ctx.sym_phase_factors(),
        ctx.num_mag_dims(), f__.pw_components());

    if (ctx.full_potential()) {
        symmetrize_mt_function(ctx.unit_cell().symmetry(), ctx.comm(), ctx.num_mag_dims(), f__.mt_components());
    }
}

}

#endif

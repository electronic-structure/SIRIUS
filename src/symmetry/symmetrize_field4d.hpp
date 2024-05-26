/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file symmetrize_field4d.hpp
 *
 *  \brief Symmetrize density and potential fields (scalar + vector).
 */

#ifndef __SYMMETRIZE_FIELD4D_HPP__
#define __SYMMETRIZE_FIELD4D_HPP__

#include "symmetrize_mt_function.hpp"
#include "symmetrize_pw_function.hpp"
#include "function3d/field4d.hpp"

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
    symmetrize_pw_function(ctx.unit_cell().symmetry(), ctx.remap_gvec(), ctx.sym_phase_factors(), ctx.num_mag_dims(),
                           f__.pw_components());

    if (ctx.full_potential()) {
        symmetrize_mt_function(ctx.unit_cell(), ctx.rotm(), ctx.mpi_grid_mt_sym(), ctx.num_mag_dims(),
                               f__.mt_components());
    }
}

} // namespace sirius

#endif

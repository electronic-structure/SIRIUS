/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __GENERATE_ALM_BLOCK_HPP__
#define __GENERATE_ALM_BLOCK_HPP__

#include "context/simulation_context.hpp"
#include "lapw/matching_coefficients.hpp"

namespace sirius {

/// Generate matching coefficients for a block of atoms.
template <bool conjugate, typename T>
auto
generate_alm_block(Simulation_context const& ctx__, int atom_begin__, int num_atoms__,
                   Matching_coefficients const& alm__)
{
    PROFILE("sirius::generate_alm_block");

    int num_mt_aw{0};
    std::vector<int> mt_aw_offset(num_atoms__);
    for (int ia = 0; ia < num_atoms__; ia++) {
        mt_aw_offset[ia] = num_mt_aw;
        num_mt_aw += ctx__.unit_cell().atom(atom_begin__ + ia).mt_aw_basis_size();
    }

    mdarray<std::complex<T>, 2> result;
    switch (ctx__.processing_unit()) {
        case device_t::CPU: {
            result = mdarray<std::complex<T>, 2>({alm__.gkvec().count(), num_mt_aw}, get_memory_pool(memory_t::host),
                                                 mdarray_label("alm_block"));
            break;
        }
        case device_t::GPU: {
            result = mdarray<std::complex<T>, 2>({alm__.gkvec().count(), num_mt_aw},
                                                 get_memory_pool(memory_t::host_pinned), mdarray_label("alm_block"));
            result.allocate(get_memory_pool(memory_t::device));
            break;
        }
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < num_atoms__; i++) {
            auto& atom = ctx__.unit_cell().atom(atom_begin__ + i);
            auto& type = atom.type();
            /* wrap matching coefficients of a single atom */
            mdarray<std::complex<T>, 2> alm_atom;
            switch (ctx__.processing_unit()) {
                case device_t::CPU: {
                    alm_atom = mdarray<std::complex<T>, 2>({alm__.gkvec().count(), type.mt_aw_basis_size()},
                                                           result.at(memory_t::host, 0, mt_aw_offset[i]),
                                                           mdarray_label("alm_atom"));
                    break;
                }
                case device_t::GPU: {
                    alm_atom = mdarray<std::complex<T>, 2>({alm__.gkvec().count(), type.mt_aw_basis_size()},
                                                           result.at(memory_t::host, 0, mt_aw_offset[i]),
                                                           result.at(memory_t::device, 0, mt_aw_offset[i]),
                                                           mdarray_label("alm_atom"));
                    break;
                }
            }
            /* generate LAPW matching coefficients on the CPU */
            alm__.template generate<conjugate>(atom, alm_atom);
            if (ctx__.processing_unit() == device_t::GPU) {
                alm_atom.copy_to(memory_t::device, acc::stream_id(tid));
            }
        }
        if (ctx__.processing_unit() == device_t::GPU) {
            acc::sync_stream(acc::stream_id(tid));
        }
    }
    return result;
}

} // namespace sirius

#endif

#ifndef __GENERATE_ALM_BLOCK_HPP__
#define __GENERATE_ALM_BLOCK_HPP__

#include "context/simulation_context.hpp"
#include "lapw/matching_coefficients.hpp"

namespace sirius {

/// Generate matching coefficients for a block of atoms.
template <bool conjugate, typename T>
auto generate_alm_block(Simulation_context const& ctx__, int atom_begin__, int num_atoms__,
        Matching_coefficients const& alm__)
{
    PROFILE("sirius::generate_alm_block");

    int num_mt_aw{0};
    std::vector<int> mt_aw_offset(num_atoms__);
    for (int ia = 0; ia < num_atoms__; ia++) {
        mt_aw_offset[ia] = num_mt_aw;
        num_mt_aw += ctx__.unit_cell().atom(atom_begin__ + ia).mt_aw_basis_size();
    }

    sddk::mdarray<std::complex<T>, 2> result;
    switch (ctx__.processing_unit()) {
        case sddk::device_t::CPU: {
            result = sddk::mdarray<std::complex<T>, 2>(alm__.gkvec().count(), num_mt_aw,
                    ctx__.mem_pool(sddk::memory_t::host), "alm_block");
            break;
        }
        case sddk::device_t::GPU: {
            result = sddk::mdarray<std::complex<T>, 2>(alm__.gkvec().count(), num_mt_aw,
                    ctx__.mem_pool(sddk::memory_t::host_pinned), "alm_block");
            result.allocate(ctx__.mem_pool(sddk::memory_t::device));
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
            sddk::mdarray<std::complex<T>, 2> alm_atom;
            switch (ctx__.processing_unit()) {
                case sddk::device_t::CPU: {
                    alm_atom = sddk::mdarray<std::complex<T>, 2>(result.at(sddk::memory_t::host, 0, mt_aw_offset[i]),
                                                                 alm__.gkvec().count(), type.mt_aw_basis_size(), "alm_atom");
                    break;
                }
                case sddk::device_t::GPU: {
                    alm_atom = sddk::mdarray<std::complex<T>, 2>(result.at(sddk::memory_t::host, 0, mt_aw_offset[i]),
                                                                 result.at(sddk::memory_t::device, 0, mt_aw_offset[i]),
                                                                 alm__.gkvec().count(), type.mt_aw_basis_size(), "alm_atom");
                    break;
                }
            }
            /* generate conjugated LAPW matching coefficients on the CPU */
            alm__.template generate<conjugate>(atom, alm_atom);
            if (ctx__.processing_unit() == sddk::device_t::GPU) {
                alm_atom.copy_to(sddk::memory_t::device, stream_id(tid));
            }

        }
        if (ctx__.processing_unit() == sddk::device_t::GPU) {
            acc::sync_stream(stream_id(tid));
        }
    }
    return result;
}

}

#endif

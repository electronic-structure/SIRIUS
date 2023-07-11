#ifndef __SYMMETRIZE_FORCES_HPP__
#define __SYMMETRIZE_FORCES_HPP__

#include "crystal_symmetry.hpp"

namespace sirius {

inline void
symmetrize_forces(Unit_cell const& uc__, sddk::mdarray<double, 2>& f__)
{
    auto& sym = uc__.symmetry();

    if (sym.size() == 1) {
        return;
    }

    sddk::mdarray<double, 2> sym_forces(3, uc__.spl_num_atoms().local_size());
    sym_forces.zero();

    for (int isym = 0; isym < sym.size(); isym++) {
        auto const& Rc = sym[isym].spg_op.Rc;

        for (int ia = 0; ia < uc__.num_atoms(); ia++) {
            r3::vector<double> force_ia(&f__(0, ia));
            int ja        = sym[isym].spg_op.sym_atom[ia];
            auto location = uc__.spl_num_atoms().location(ja);
            if (location.rank == uc__.comm().rank()) {
                auto force_ja = dot(Rc, force_ia);
                for (int x : {0, 1, 2}) {
                    sym_forces(x, location.local_index) += force_ja[x];
                }
            }
        }
    }

    double alpha = 1.0 / double(sym.size());
    for (int ia = 0; ia < uc__.spl_num_atoms().local_size(); ia++) {
        for (int x: {0, 1, 2}) {
            sym_forces(x, ia) *= alpha;
        }
    }
    double* sbuf = uc__.spl_num_atoms().local_size() ? sym_forces.at(sddk::memory_t::host) : nullptr;
    uc__.comm().allgather(sbuf, f__.at(sddk::memory_t::host), 3 * uc__.spl_num_atoms().local_size(),
        3 * uc__.spl_num_atoms().global_offset());
}

}

#endif

#include "hubbard.hpp"

namespace sirius {

Hubbard::Hubbard(Simulation_context& ctx__)
    : ctx_(ctx__)
    , unit_cell_(ctx__.unit_cell())
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    int indexb_max = -1;

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        if (ctx_.unit_cell().atom(ia).type().hubbard_correction()) {
            indexb_max = std::max(indexb_max, static_cast<int>(ctx_.unit_cell().atom(ia).type().indexb_hub().size()));
        }
    }

    max_number_of_orbitals_per_atom_ = indexb_max;

    hubbard_potential_ = sddk::mdarray<double_complex, 4>(indexb_max, indexb_max, 4, ctx_.unit_cell().num_atoms(),
            memory_t::host, "hubbard_potential_");

    auto r = ctx_.unit_cell().num_hubbard_wf();

    number_of_hubbard_orbitals_ = r.first;
    offset_ = r.second;
}

} // namespace sirius

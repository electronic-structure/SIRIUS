#include "hubbard.hpp"

namespace sirius {

Hubbard::Hubbard(Simulation_context& ctx__)
    : ctx_(ctx__)
    , unit_cell_(ctx__.unit_cell())
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    auto r                      = ctx_.unit_cell().num_hubbard_wf();
    number_of_hubbard_orbitals_ = r.first;
}

} // namespace sirius

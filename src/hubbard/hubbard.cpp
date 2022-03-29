#include "hubbard.hpp"

namespace sirius {

Hubbard::Hubbard(Simulation_context& ctx__)
    : ctx_(ctx__)
    , unit_cell_(ctx__.unit_cell())
{
    if (!ctx_.hubbard_correction()) {
        return;
    }
}

} // namespace sirius

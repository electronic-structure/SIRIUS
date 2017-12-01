#ifndef __HAMILTONIAN_H__
#define __HAMILTONIAN_H__
#include "simulation_context.h"
#include "hubbard.hpp"
#include "potential.h"

namespace sirius {

class Hamiltonian
{
 private:
  /// Simulation context.
  Simulation_context& ctx_;

  /// Alias for the unit cell.
  Unit_cell& unit_cell_;

  /// alias for the potential
  Potential &potential_;

  /// Alias for the hubbard potential (note it is a pointer)
  std::unique_ptr<Hubbard_potential> U_;

 public:
  Hubbard_potential& U() const
    {
      return *U_;
    }

  Potential &potential() const
    {
      return potential_;
    }

  Unit_cell &unit_cell() const
    {
      return unit_cell_;
    }

  Simulation_context& ctx() const
    {
      return ctx_;
    }

  Hamiltonian(Simulation_context& ctx__, Potential &potential__)
    : ctx_(ctx__),
    unit_cell_(ctx__.unit_cell()),
    potential_(potential__)
    {
      if(ctx_.hubbard_correction()) {
        U_ = std::unique_ptr<Hubbard_potential>(new Hubbard_potential(ctx_));
      }
    }
};
}
#endif
